"""Тесты контроля качества артефактов stage 6 embedding-based verification.

Файл размещается в каталоге:

    tests/embedding_model/test_stage6_embedding_artifacts.py

Назначение тестов:

    1. Проверить наличие ключевых файлов stage 6.
    2. Проверить, что embedding-классификатор обучался на 31 признаке.
    3. Проверить структуру CSV-отчётов.
    4. Зафиксировать итоговый вывод stage 6: embedding-based verification
       уступает softmax baseline v2 по FRR и balanced error при сопоставимом FAR.
    5. Проверить, что итоговый Markdown-отчёт содержит обязательные разделы.

Тесты не переобучают модель и не изменяют артефакты проекта.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def get_project_root() -> Path:
    """Получить корень проекта из расположения тестового файла."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture()
def project_root() -> Path:
    """Вернуть корень проекта."""
    return get_project_root()


@pytest.fixture()
def embedding_dirs(project_root: Path) -> dict[str, Path]:
    """Вернуть основные каталоги stage 6."""
    return {
        "models": project_root / "models" / "embedding_model",
        "users": project_root / "users" / "embedding_model",
        "reports": project_root / "reports" / "embedding_model",
        "processed": project_root / "data" / "processed" / "embedding_model",
    }


@pytest.mark.parametrize(
    "directory_key, filename",
    [
        ("models", "embedding_classifier.keras"),
        ("models", "encoder.keras"),
        ("models", "embedding_scaler.pkl"),
        ("models", "embedding_label_encoder.pkl"),
        ("users", "user_templates_embedding.json"),
        ("users", "user_thresholds_embedding.json"),
        ("processed", "cmu_embedding_split.json"),
        ("processed", "embeddings_train.csv"),
        ("processed", "embeddings_validation.csv"),
        ("processed", "embeddings_test.csv"),
        ("reports", "embedding_classifier_metrics.csv"),
        ("reports", "embedding_feature_columns.txt"),
        ("reports", "embedding_distance_diagnostics.csv"),
        ("reports", "embedding_threshold_policy.csv"),
        ("reports", "embedding_enrollment_size_experiment.csv"),
        ("reports", "embedding_vs_softmax_comparison.csv"),
        ("reports", "embedding_vs_softmax_summary.md"),
        ("reports", "stage6_embedding_report.md"),
    ],
)
def test_required_stage6_artifacts_exist(
    embedding_dirs: dict[str, Path],
    directory_key: str,
    filename: str,
) -> None:
    """Проверить наличие и непустой размер ключевых файлов stage 6."""
    path = embedding_dirs[directory_key] / filename

    assert path.exists(), f"Артефакт stage 6 не найден: {path}"
    assert path.stat().st_size > 0, f"Артефакт stage 6 пуст: {path}"


def test_embedding_feature_columns_are_31_without_service_columns(
    embedding_dirs: dict[str, Path],
) -> None:
    """Проверить, что в embedding-модель попал ровно 31 timing feature."""
    path = embedding_dirs["reports"] / "embedding_feature_columns.txt"

    feature_columns = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    assert len(feature_columns) == 31
    assert not any(column.startswith("Unnamed") for column in feature_columns)
    assert "user_id" not in feature_columns


def test_classifier_metrics_have_expected_splits(
    embedding_dirs: dict[str, Path],
) -> None:
    """Проверить структуру отчёта качества embedding-классификатора."""
    path = embedding_dirs["reports"] / "embedding_classifier_metrics.csv"
    metrics = pd.read_csv(path)

    assert set(metrics["split"]) == {"train", "validation", "test"}
    assert {"samples", "accuracy", "macro_f1", "weighted_f1"} <= set(metrics.columns)

    test_row = metrics.loc[metrics["split"] == "test"].iloc[0]
    assert int(test_row["samples"]) == 4080
    assert float(test_row["accuracy"]) >= 0.90


def test_distance_diagnostics_selects_cosine_as_best_test_metric(
    embedding_dirs: dict[str, Path],
) -> None:
    """Проверить, что cosine distance даёт лучший test EER."""
    path = embedding_dirs["reports"] / "embedding_distance_diagnostics.csv"
    diagnostics = pd.read_csv(path)

    test_rows = diagnostics.loc[diagnostics["split"] == "test"].copy()
    best_row = test_rows.loc[test_rows["eer"].idxmin()]

    assert best_row["distance_metric"] == "cosine"
    assert float(best_row["eer"]) < 0.06
    assert float(best_row["roc_auc"]) > 0.98


def test_threshold_policy_has_expected_shape_and_best_policy(
    embedding_dirs: dict[str, Path],
) -> None:
    """Проверить структуру threshold policy и лучший test-вариант."""
    path = embedding_dirs["reports"] / "embedding_threshold_policy.csv"
    policy = pd.read_csv(path)

    assert len(policy) == 18
    assert set(policy["split"]) == {"validation", "test"}
    assert set(policy["distance_metric"]) == {"euclidean", "cosine", "manhattan"}
    assert set(policy["policy"]) == {"global", "per_user", "guarded"}

    test_rows = policy.loc[policy["split"] == "test"].copy()
    best_row = test_rows.loc[test_rows["balanced_error"].idxmin()]

    assert best_row["distance_metric"] == "cosine"
    assert best_row["policy"] == "per_user"
    assert float(best_row["far"]) < 0.011
    assert float(best_row["frr"]) > 0.10


def test_enrollment_size_experiment_shows_frr_reduction(
    embedding_dirs: dict[str, Path],
) -> None:
    """Проверить, что рост числа enrollment samples снижает FRR."""
    path = embedding_dirs["reports"] / "embedding_enrollment_size_experiment.csv"
    experiment = pd.read_csv(path)

    target_rows = experiment.loc[
        (experiment["split"] == "test")
        & (experiment["distance_metric"] == "cosine")
        & (experiment["policy"] == "per_user")
    ].copy()

    assert set(target_rows["enrollment_samples"]) >= {5, 10, 20, 30, 50}

    by_n = target_rows.set_index("enrollment_samples")

    assert float(by_n.loc[50, "frr"]) < float(by_n.loc[5, "frr"])
    assert float(by_n.loc[50, "eer"]) < float(by_n.loc[5, "eer"])
    assert float(by_n.loc[50, "far"]) < 0.011


def test_embedding_is_worse_than_softmax_baseline_by_frr(
    embedding_dirs: dict[str, Path],
) -> None:
    """Зафиксировать итоговый научный вывод stage 6."""
    path = embedding_dirs["reports"] / "embedding_vs_softmax_comparison.csv"
    comparison = pd.read_csv(path)

    baseline = comparison.loc[
        (comparison["approach"] == "softmax_based_verification")
        & (comparison["policy"] == "global")
    ].iloc[0]

    embedding_rows = comparison.loc[comparison["approach"] == "embedding_based_verification"].copy()
    best_embedding = embedding_rows.loc[embedding_rows["balanced_error"].idxmin()]

    assert float(best_embedding["far"]) <= float(baseline["far"]) + 0.002
    assert float(best_embedding["frr"]) > float(baseline["frr"])
    assert float(best_embedding["balanced_error"]) > float(baseline["balanced_error"])


def test_stage6_report_contains_required_sections(
    embedding_dirs: dict[str, Path],
) -> None:
    """Проверить наличие обязательных разделов итогового Markdown-отчёта."""
    path = embedding_dirs["reports"] / "stage6_embedding_report.md"
    text = path.read_text(encoding="utf-8")

    required_fragments = [
        "# Stage 6. Embedding-based verification",
        "## 2. Качество embedding-классификатора",
        "## 3. Диагностика embedding-расстояний",
        "## 4. Threshold policy",
        "## 5. Эксперимент с количеством enrollment samples",
        "## 6. Сравнение с softmax baseline v2",
        "## 8. Итоговый вывод",
        "softmax baseline v2 остаётся основной рабочей моделью",
    ]

    for fragment in required_fragments:
        assert fragment in text
