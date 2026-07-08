"""Единый evaluation pipeline для Stage 7 embedding encoder-ов.

Скрипт оценивает encoder-ы, обученные разными objective:
- softmax cross-entropy encoder из Stage 6;
- Siamese encoder с contrastive loss;
- Triplet encoder с triplet loss.

Порог аутентификации подбирается только на validation split.
Test split используется только для финальной оценки FAR, FRR, EER
и per-user ошибок.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.embedding_model.generate_metric_learning_pairs import (
    DEFAULT_FEATURES_FILE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    add_deterministic_split,
    get_feature_columns,
    load_feature_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "embedding_model"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "embedding_model"

DEFAULT_TARGET_FAR = 0.01
DEFAULT_BATCH_SIZE = 512
DEFAULT_TRAIN_FRACTION = 0.64
DEFAULT_VALIDATION_FRACTION = 0.16

SUPPORTED_DISTANCES = ("cosine", "euclidean", "manhattan")
SUPPORTED_POLICIES = ("global", "per_user", "guarded")


@dataclass(frozen=True)
class EncoderConfig:
    """Пути к encoder-у и scaler-у для одного варианта Stage 7."""

    name: str
    encoder_path: Path
    scaler_path: Path


@dataclass(frozen=True)
class SplitFeatures:
    """Нормализованные признаки одного split-а."""

    split_name: str
    x: np.ndarray
    user_ids: np.ndarray
    sample_ids: np.ndarray


@dataclass(frozen=True)
class EmbeddingSplit:
    """Embedding-векторы одного split-а."""

    split_name: str
    embeddings: np.ndarray
    user_ids: np.ndarray
    sample_ids: np.ndarray


@dataclass(frozen=True)
class TemplateSet:
    """Набор средних embedding-шаблонов пользователей."""

    user_ids: list[str]
    template_matrix: np.ndarray
    samples_count: dict[str, int]


@dataclass(frozen=True)
class TrialDistanceSet:
    """Distance matrix для проверки samples против всех user templates."""

    split_name: str
    distance_name: str
    distance_matrix: np.ndarray
    sample_user_ids: np.ndarray
    sample_ids: np.ndarray
    template_user_ids: list[str]

    @property
    def user_index(self) -> dict[str, int]:
        """Вернуть отображение user_id -> индекс колонки template matrix."""
        return {user_id: index for index, user_id in enumerate(self.template_user_ids)}


@dataclass(frozen=True)
class ThresholdPolicy:
    """Набор порогов для одного distance/policy варианта."""

    distance_name: str
    policy_name: str
    thresholds: dict[str, float]
    selected_users: list[str]


ENCODER_CONFIGS = {
    "softmax": EncoderConfig(
        name="softmax",
        encoder_path=DEFAULT_MODELS_DIR / "encoder.keras",
        scaler_path=DEFAULT_MODELS_DIR / "embedding_scaler.pkl",
    ),
    "siamese": EncoderConfig(
        name="siamese",
        encoder_path=DEFAULT_MODELS_DIR / "siamese_encoder.keras",
        scaler_path=DEFAULT_MODELS_DIR / "metric_learning_scaler.pkl",
    ),
    "triplet": EncoderConfig(
        name="triplet",
        encoder_path=DEFAULT_MODELS_DIR / "triplet_encoder.keras",
        scaler_path=DEFAULT_MODELS_DIR / "metric_learning_scaler.pkl",
    ),
}


def load_and_scale_splits(
    features_file: Path,
    scaler_path: Path,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[list[SplitFeatures], list[str]]:
    """Загрузить таблицу признаков и применить уже сохранённый scaler.

    Scaler не обучается заново. Это важно для корректного сравнения encoder-ов:
    каждый encoder оценивается с тем preprocessing, с которым он обучался.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    df = load_feature_table(features_file)
    df = add_deterministic_split(
        df=df,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    feature_columns = get_feature_columns(df)
    scaler = joblib.load(scaler_path)

    split_features = []
    for split_name in [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT]:
        split_df = df[df["split"] == split_name].copy().reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"Split is empty: {split_name}")

        x = split_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
        x_scaled = scaler.transform(x).astype(np.float32)
        split_features.append(
            SplitFeatures(
                split_name=split_name,
                x=x_scaled,
                user_ids=split_df["user_id"].astype(str).to_numpy(),
                sample_ids=split_df["sample_id"].astype(str).to_numpy(),
            )
        )

    return split_features, feature_columns


def load_encoder_model(encoder_path: Path) -> Any:
    """Загрузить Keras encoder без компиляции.

    Перед загрузкой импортируются Stage 7 model modules. Это регистрирует
    пользовательские Keras-слои `L2NormalizeLayer`, `EuclideanDistanceLayer`
    и `TripletDistanceLayer`, необходимые для корректной десериализации.
    """
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    import importlib

    import tensorflow as tf

    importlib.import_module("src.embedding_model.siamese_model")
    importlib.import_module("src.embedding_model.triplet_model")

    try:
        return tf.keras.models.load_model(encoder_path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(encoder_path, compile=False)


def extract_embedding_splits(
    encoder: Any,
    split_features: list[SplitFeatures],
    batch_size: int,
) -> list[EmbeddingSplit]:
    """Преобразовать признаки train/validation/test в embedding-векторы."""
    embedding_splits = []

    for split in split_features:
        embeddings = encoder.predict(split.x, batch_size=batch_size, verbose=0)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(f"Encoder output must be 2D, got shape: {embeddings.shape}")

        embedding_splits.append(
            EmbeddingSplit(
                split_name=split.split_name,
                embeddings=embeddings,
                user_ids=split.user_ids,
                sample_ids=split.sample_ids,
            )
        )

    return embedding_splits


def build_mean_templates(train_split: EmbeddingSplit) -> TemplateSet:
    """Построить средний embedding-template для каждого пользователя."""
    user_ids = sorted(np.unique(train_split.user_ids).astype(str).tolist())
    templates = []
    samples_count: dict[str, int] = {}

    for user_id in user_ids:
        mask = train_split.user_ids == user_id
        user_embeddings = train_split.embeddings[mask]

        if user_embeddings.size == 0:
            raise ValueError(f"No enrollment embeddings for user: {user_id}")

        templates.append(np.mean(user_embeddings, axis=0))
        samples_count[user_id] = int(user_embeddings.shape[0])

    return TemplateSet(
        user_ids=user_ids,
        template_matrix=np.vstack(templates).astype(np.float32),
        samples_count=samples_count,
    )


def calculate_distance_matrix(
    embeddings: np.ndarray,
    templates: np.ndarray,
    distance_name: str,
) -> np.ndarray:
    """Рассчитать distance matrix `samples x users`."""
    if embeddings.ndim != 2 or templates.ndim != 2:
        raise ValueError("Embeddings and templates must be 2D arrays.")

    if embeddings.shape[1] != templates.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"{embeddings.shape[1]} != {templates.shape[1]}"
        )

    if distance_name == "euclidean":
        diff = embeddings[:, None, :] - templates[None, :, :]
        return np.linalg.norm(diff, axis=2).astype(np.float32)

    if distance_name == "manhattan":
        diff = np.abs(embeddings[:, None, :] - templates[None, :, :])
        return np.sum(diff, axis=2).astype(np.float32)

    if distance_name == "cosine":
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        template_norms = np.linalg.norm(templates, axis=1, keepdims=True).T
        denominator = np.maximum(embedding_norms * template_norms, np.finfo(np.float32).eps)
        similarity = (embeddings @ templates.T) / denominator
        return (1.0 - similarity).astype(np.float32)

    raise ValueError(f"Unsupported distance: {distance_name}")


def build_trial_distances(
    embedding_split: EmbeddingSplit,
    templates: TemplateSet,
    distance_name: str,
) -> TrialDistanceSet:
    """Построить расстояния samples split-а до всех пользовательских шаблонов."""
    distance_matrix = calculate_distance_matrix(
        embeddings=embedding_split.embeddings,
        templates=templates.template_matrix,
        distance_name=distance_name,
    )

    return TrialDistanceSet(
        split_name=embedding_split.split_name,
        distance_name=distance_name,
        distance_matrix=distance_matrix,
        sample_user_ids=embedding_split.user_ids,
        sample_ids=embedding_split.sample_ids,
        template_user_ids=templates.user_ids,
    )


def split_genuine_impostor_distances(trials: TrialDistanceSet) -> tuple[np.ndarray, np.ndarray]:
    """Разделить расстояния на genuine и impostor попытки."""
    user_index = trials.user_index
    genuine_distances = []
    impostor_distances = []

    for row_index, actual_user_id in enumerate(trials.sample_user_ids):
        actual_user_id = str(actual_user_id)
        if actual_user_id not in user_index:
            raise ValueError(f"No template for user: {actual_user_id}")

        genuine_column = user_index[actual_user_id]
        row_distances = trials.distance_matrix[row_index]
        genuine_distances.append(row_distances[genuine_column])

        impostor_mask = np.ones(row_distances.shape[0], dtype=bool)
        impostor_mask[genuine_column] = False
        impostor_distances.extend(row_distances[impostor_mask].tolist())

    return (
        np.asarray(genuine_distances, dtype=np.float32),
        np.asarray(impostor_distances, dtype=np.float32),
    )


def calculate_eer_and_auc(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
) -> dict[str, float]:
    """Рассчитать EER и ROC AUC для distance-based verification."""
    if genuine_distances.size == 0 or impostor_distances.size == 0:
        raise ValueError("Both genuine and impostor distances are required.")

    labels = np.concatenate(
        [np.ones(genuine_distances.shape[0]), np.zeros(impostor_distances.shape[0])]
    )
    scores = -np.concatenate([genuine_distances, impostor_distances])

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    roc_auc = float(roc_auc_score(labels, scores))

    return {"eer": eer, "roc_auc": roc_auc}


def calculate_distance_diagnostics(trials: TrialDistanceSet) -> dict[str, float | str]:
    """Сформировать summary диагностики genuine/impostor расстояний."""
    genuine_distances, impostor_distances = split_genuine_impostor_distances(trials)
    eer_auc = calculate_eer_and_auc(genuine_distances, impostor_distances)
    genuine_mean = float(np.mean(genuine_distances))
    impostor_mean = float(np.mean(impostor_distances))

    return {
        "split": trials.split_name,
        "distance": trials.distance_name,
        "roc_auc": eer_auc["roc_auc"],
        "eer": eer_auc["eer"],
        "genuine_mean": genuine_mean,
        "impostor_mean": impostor_mean,
        "margin": impostor_mean - genuine_mean,
        "genuine_trials": int(genuine_distances.shape[0]),
        "impostor_trials": int(impostor_distances.shape[0]),
    }


def select_distance_threshold_at_far(impostor_distances: np.ndarray, target_far: float) -> float:
    """Выбрать максимальный distance-threshold с FAR не выше target FAR.

    Для distance-based verification попытка принимается при `distance <= threshold`.
    Поэтому FAR равен доле impostor distances, которые оказались меньше
    или равны threshold.
    """
    if not 0.0 < target_far < 1.0:
        raise ValueError("target_far must be in (0, 1).")

    if impostor_distances.size == 0:
        raise ValueError("Impostor distances are required for threshold selection.")

    sorted_distances = np.sort(np.asarray(impostor_distances, dtype=np.float64))
    allowed_accepts = int(np.floor(target_far * sorted_distances.shape[0]))

    if allowed_accepts <= 0:
        return float(np.nextafter(sorted_distances[0], -np.inf))

    selected_threshold = float(np.nextafter(sorted_distances[0], -np.inf))
    for candidate in np.unique(sorted_distances):
        accepted = int(np.searchsorted(sorted_distances, candidate, side="right"))
        if accepted <= allowed_accepts:
            selected_threshold = float(candidate)
        else:
            break

    return selected_threshold


def select_global_policy(
    validation_trials: TrialDistanceSet,
    target_far: float,
) -> ThresholdPolicy:
    """Подобрать общий порог на validation split."""
    _, impostor_distances = split_genuine_impostor_distances(validation_trials)
    threshold = select_distance_threshold_at_far(impostor_distances, target_far)

    return ThresholdPolicy(
        distance_name=validation_trials.distance_name,
        policy_name="global",
        thresholds={user_id: threshold for user_id in validation_trials.template_user_ids},
        selected_users=[],
    )


def select_per_user_policy(
    validation_trials: TrialDistanceSet,
    target_far: float,
) -> ThresholdPolicy:
    """Подобрать индивидуальные пороги по claimed_user_id на validation split."""
    thresholds = {}

    for column_index, claimed_user_id in enumerate(validation_trials.template_user_ids):
        impostor_mask = validation_trials.sample_user_ids != claimed_user_id
        impostor_distances = validation_trials.distance_matrix[impostor_mask, column_index]
        thresholds[claimed_user_id] = select_distance_threshold_at_far(
            impostor_distances,
            target_far,
        )

    return ThresholdPolicy(
        distance_name=validation_trials.distance_name,
        policy_name="per_user",
        thresholds=thresholds,
        selected_users=list(thresholds.keys()),
    )


def select_guarded_policy(
    validation_trials: TrialDistanceSet,
    global_policy: ThresholdPolicy,
    per_user_policy: ThresholdPolicy,
    max_validation_frr_increase: float,
) -> ThresholdPolicy:
    """Построить guarded per-user policy на validation split.

    Индивидуальный порог применяется только если он строже глобального
    и не увеличивает validation FRR пользователя сверх допустимого зазора.
    """
    if max_validation_frr_increase < 0:
        raise ValueError("max_validation_frr_increase must be non-negative.")

    global_user_stats = calculate_per_user_error_stats(
        validation_trials,
        global_policy.thresholds,
    )
    candidate_user_stats = calculate_per_user_error_stats(
        validation_trials,
        per_user_policy.thresholds,
    )

    guarded_thresholds = dict(global_policy.thresholds)
    selected_users = []

    for user_id in validation_trials.template_user_ids:
        global_threshold = global_policy.thresholds[user_id]
        candidate_threshold = per_user_policy.thresholds[user_id]

        if candidate_threshold >= global_threshold:
            continue

        global_stats = global_user_stats[user_id]
        candidate_stats = candidate_user_stats[user_id]
        frr_limit = global_stats["frr"] + max_validation_frr_increase

        if candidate_stats["far"] <= global_stats["far"] and candidate_stats["frr"] <= frr_limit:
            guarded_thresholds[user_id] = candidate_threshold
            selected_users.append(user_id)

    return ThresholdPolicy(
        distance_name=validation_trials.distance_name,
        policy_name="guarded",
        thresholds=guarded_thresholds,
        selected_users=selected_users,
    )


def evaluate_threshold_policy(
    trials: TrialDistanceSet,
    thresholds: dict[str, float],
) -> dict[str, float | int]:
    """Рассчитать FAR/FRR для заданной политики порогов."""
    user_index = trials.user_index
    false_accepts = 0
    false_rejects = 0
    genuine_trials = 0
    impostor_trials = 0

    for row_index, actual_user_id in enumerate(trials.sample_user_ids):
        actual_user_id = str(actual_user_id)
        if actual_user_id not in user_index:
            raise ValueError(f"No template for user: {actual_user_id}")

        for claimed_user_id, column_index in user_index.items():
            threshold = thresholds[claimed_user_id]
            accepted = trials.distance_matrix[row_index, column_index] <= threshold

            if claimed_user_id == actual_user_id:
                genuine_trials += 1
                if not accepted:
                    false_rejects += 1
            else:
                impostor_trials += 1
                if accepted:
                    false_accepts += 1

    far = false_accepts / impostor_trials if impostor_trials else 0.0
    frr = false_rejects / genuine_trials if genuine_trials else 0.0

    return {
        "genuine_trials": genuine_trials,
        "impostor_trials": impostor_trials,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "far": far,
        "frr": frr,
        "balanced_error": (far + frr) / 2.0,
    }


def calculate_per_user_error_stats(
    trials: TrialDistanceSet,
    thresholds: dict[str, float],
) -> dict[str, dict[str, float | int]]:
    """Рассчитать FAR/FRR по каждому claimed/actual пользователю."""
    user_index = trials.user_index
    stats = {
        user_id: {
            "impostor_trials": 0,
            "genuine_trials": 0,
            "false_accepts": 0,
            "false_rejects": 0,
            "far": 0.0,
            "frr": 0.0,
        }
        for user_id in trials.template_user_ids
    }

    for row_index, actual_user_id in enumerate(trials.sample_user_ids):
        actual_user_id = str(actual_user_id)
        for claimed_user_id, column_index in user_index.items():
            threshold = thresholds[claimed_user_id]
            accepted = trials.distance_matrix[row_index, column_index] <= threshold

            if claimed_user_id == actual_user_id:
                stats[claimed_user_id]["genuine_trials"] += 1
                if not accepted:
                    stats[claimed_user_id]["false_rejects"] += 1
            else:
                stats[claimed_user_id]["impostor_trials"] += 1
                if accepted:
                    stats[claimed_user_id]["false_accepts"] += 1

    for user_stats in stats.values():
        impostor_trials = int(user_stats["impostor_trials"])
        genuine_trials = int(user_stats["genuine_trials"])
        user_stats["far"] = (
            int(user_stats["false_accepts"]) / impostor_trials if impostor_trials else 0.0
        )
        user_stats["frr"] = (
            int(user_stats["false_rejects"]) / genuine_trials if genuine_trials else 0.0
        )

    return stats


def build_policy_summary_row(
    encoder_name: str,
    trials: TrialDistanceSet,
    policy: ThresholdPolicy,
    target_far: float,
) -> dict[str, float | int | str]:
    """Сформировать строку summary для threshold policy."""
    summary = evaluate_threshold_policy(trials, policy.thresholds)
    per_user_stats = calculate_per_user_error_stats(trials, policy.thresholds)
    far_values = [float(user_stats["far"]) for user_stats in per_user_stats.values()]
    frr_values = [float(user_stats["frr"]) for user_stats in per_user_stats.values()]

    return {
        "encoder": encoder_name,
        "split": trials.split_name,
        "distance": policy.distance_name,
        "policy": policy.policy_name,
        "target_far": target_far,
        "far": float(summary["far"]),
        "frr": float(summary["frr"]),
        "balanced_error": float(summary["balanced_error"]),
        "false_accepts": int(summary["false_accepts"]),
        "false_rejects": int(summary["false_rejects"]),
        "genuine_trials": int(summary["genuine_trials"]),
        "impostor_trials": int(summary["impostor_trials"]),
        "max_user_far": float(np.max(far_values)),
        "max_user_frr": float(np.max(frr_values)),
        "mean_user_far": float(np.mean(far_values)),
        "mean_user_frr": float(np.mean(frr_values)),
        "selected_users_count": len(policy.selected_users),
        "selected_users": ",".join(policy.selected_users),
    }


def save_templates(templates: TemplateSet, output_path: Path) -> None:
    """Сохранить embedding templates в JSON для аудита evaluation pipeline."""
    payload = {
        user_id: {
            "embedding": templates.template_matrix[index].astype(float).tolist(),
            "samples_count": templates.samples_count[user_id],
            "template_method": "mean_embedding_train_split",
        }
        for index, user_id in enumerate(templates.user_ids)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_thresholds(
    policies: list[ThresholdPolicy],
    output_path: Path,
) -> None:
    """Сохранить подобранные validation thresholds в JSON."""
    payload: dict[str, dict[str, Any]] = {}

    for policy in policies:
        key = f"{policy.distance_name}_{policy.policy_name}"
        payload[key] = {
            "distance": policy.distance_name,
            "policy": policy.policy_name,
            "thresholds": policy.thresholds,
            "selected_users": policy.selected_users,
            "threshold_selection_split": VALIDATION_SPLIT,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_per_user_stats(
    encoder_name: str,
    trials: TrialDistanceSet,
    policy: ThresholdPolicy,
    output_path: Path,
) -> None:
    """Сохранить per-user FAR/FRR diagnostics для одной политики."""
    per_user_stats = calculate_per_user_error_stats(trials, policy.thresholds)
    rows = []

    for user_id, stats in per_user_stats.items():
        rows.append(
            {
                "encoder": encoder_name,
                "split": trials.split_name,
                "distance": policy.distance_name,
                "policy": policy.policy_name,
                "user_id": user_id,
                **stats,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def evaluate_one_encoder(
    config: EncoderConfig,
    features_file: Path,
    output_root: Path,
    distances: list[str],
    target_far: float,
    batch_size: int,
    train_fraction: float,
    validation_fraction: float,
    max_validation_frr_increase: float,
) -> None:
    """Выполнить полный evaluation cycle для одного encoder-а."""
    output_dir = output_root / f"{config.name}_encoder"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_features, feature_columns = load_and_scale_splits(
        features_file=features_file,
        scaler_path=config.scaler_path,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    encoder = load_encoder_model(config.encoder_path)
    embedding_splits = extract_embedding_splits(encoder, split_features, batch_size)
    split_by_name = {split.split_name: split for split in embedding_splits}
    templates = build_mean_templates(split_by_name[TRAIN_SPLIT])
    save_templates(templates, output_dir / "user_templates.json")

    diagnostics_rows = []
    policy_rows = []
    selected_policies = []

    for distance_name in distances:
        validation_trials = build_trial_distances(
            embedding_split=split_by_name[VALIDATION_SPLIT],
            templates=templates,
            distance_name=distance_name,
        )
        test_trials = build_trial_distances(
            embedding_split=split_by_name[TEST_SPLIT],
            templates=templates,
            distance_name=distance_name,
        )

        diagnostics_rows.append(calculate_distance_diagnostics(validation_trials))
        diagnostics_rows.append(calculate_distance_diagnostics(test_trials))

        global_policy = select_global_policy(validation_trials, target_far)
        per_user_policy = select_per_user_policy(validation_trials, target_far)
        guarded_policy = select_guarded_policy(
            validation_trials=validation_trials,
            global_policy=global_policy,
            per_user_policy=per_user_policy,
            max_validation_frr_increase=max_validation_frr_increase,
        )
        distance_policies = [global_policy, per_user_policy, guarded_policy]
        selected_policies.extend(distance_policies)

        for policy in distance_policies:
            policy_rows.append(
                build_policy_summary_row(config.name, validation_trials, policy, target_far)
            )
            policy_rows.append(
                build_policy_summary_row(config.name, test_trials, policy, target_far)
            )

            save_per_user_stats(
                encoder_name=config.name,
                trials=test_trials,
                policy=policy,
                output_path=output_dir
                / f"per_user_{distance_name}_{policy.policy_name}_test.csv",
            )

    diagnostics_path = output_dir / "embedding_distance_diagnostics.csv"
    policy_path = output_dir / "embedding_threshold_policy.csv"
    thresholds_path = output_dir / "user_thresholds.json"
    metadata_path = output_dir / "evaluation_metadata.json"

    pd.DataFrame(diagnostics_rows).to_csv(diagnostics_path, index=False)
    pd.DataFrame(policy_rows).to_csv(policy_path, index=False)
    save_thresholds(selected_policies, thresholds_path)

    metadata = {
        "encoder": config.name,
        "encoder_path": str(config.encoder_path),
        "scaler_path": str(config.scaler_path),
        "features_file": str(features_file),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_far": target_far,
        "threshold_selection_split": VALIDATION_SPLIT,
        "test_split_usage": "final_evaluation_only",
        "distances": distances,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Encoder evaluation completed: {config.name}")
    print(f"Report directory: {output_dir}")
    print(f"Diagnostics: {diagnostics_path}")
    print(f"Threshold policy: {policy_path}")
    print(f"Thresholds: {thresholds_path}")


def resolve_encoder_names(requested_names: list[str]) -> list[str]:
    """Развернуть `all` в список поддерживаемых encoder-ов."""
    if not requested_names or "all" in requested_names:
        return list(ENCODER_CONFIGS.keys())

    unknown_names = sorted(set(requested_names) - set(ENCODER_CONFIGS.keys()))
    if unknown_names:
        raise ValueError(f"Unknown encoder names: {unknown_names}")

    return requested_names


def resolve_distances(requested_distances: list[str]) -> list[str]:
    """Развернуть список distance metrics."""
    if not requested_distances or "all" in requested_distances:
        return list(SUPPORTED_DISTANCES)

    unknown_distances = sorted(set(requested_distances) - set(SUPPORTED_DISTANCES))
    if unknown_distances:
        raise ValueError(f"Unknown distances: {unknown_distances}")

    return requested_distances


def build_arg_parser() -> argparse.ArgumentParser:
    """Создать CLI parser для generalized evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 7 encoders with a shared distance-based protocol."
    )
    parser.add_argument("--features-file", type=Path, default=DEFAULT_FEATURES_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--encoder-name",
        action="append",
        choices=["all", *ENCODER_CONFIGS.keys()],
        default=None,
        help="Encoder to evaluate. Can be repeated. Default: all.",
    )
    parser.add_argument(
        "--distance",
        action="append",
        choices=["all", *SUPPORTED_DISTANCES],
        default=None,
        help="Distance metric. Can be repeated. Default: all.",
    )
    parser.add_argument("--target-far", type=float, default=DEFAULT_TARGET_FAR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--max-validation-frr-increase", type=float, default=0.02)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip encoder variants with missing model/scaler files.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    encoder_names = resolve_encoder_names(args.encoder_name or ["all"])
    distances = resolve_distances(args.distance or ["all"])

    for encoder_name in encoder_names:
        config = ENCODER_CONFIGS[encoder_name]
        missing_encoder = not config.encoder_path.exists()
        missing_scaler = not config.scaler_path.exists()
        if args.skip_missing and (missing_encoder or missing_scaler):
            print(f"Skip missing encoder/scaler for: {encoder_name}")
            continue

        evaluate_one_encoder(
            config=config,
            features_file=args.features_file,
            output_root=args.output_root,
            distances=distances,
            target_far=args.target_far,
            batch_size=args.batch_size,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            max_validation_frr_increase=args.max_validation_frr_increase,
        )


if __name__ == "__main__":
    main()
