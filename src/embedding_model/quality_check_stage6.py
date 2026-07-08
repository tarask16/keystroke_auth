"""Контроль качества артефактов stage 6 embedding-based verification.

Файл размещается в каталоге:

    src/embedding_model/quality_check_stage6.py

Назначение файла:

    1. Проверить наличие ключевых артефактов stage 6.
    2. Проверить согласованность основных CSV-отчётов.
    3. Подтвердить лучший embedding-вариант.
    4. Подтвердить итоговый вывод о сравнении с softmax baseline v2.

Скрипт не переобучает модель и не изменяет существующие артефакты.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from src.embedding_model.embedding import EmbeddingModelPaths

REQUIRED_ARTIFACTS: Final[tuple[str, ...]] = (
    "models/embedding_model/embedding_classifier.keras",
    "models/embedding_model/encoder.keras",
    "models/embedding_model/embedding_scaler.pkl",
    "models/embedding_model/embedding_label_encoder.pkl",
    "users/embedding_model/user_templates_embedding.json",
    "users/embedding_model/user_thresholds_embedding.json",
    "data/processed/embedding_model/cmu_embedding_split.json",
    "data/processed/embedding_model/embeddings_train.csv",
    "data/processed/embedding_model/embeddings_validation.csv",
    "data/processed/embedding_model/embeddings_test.csv",
    "reports/embedding_model/embedding_classifier_metrics.csv",
    "reports/embedding_model/embedding_feature_columns.txt",
    "reports/embedding_model/embedding_distance_diagnostics.csv",
    "reports/embedding_model/embedding_threshold_policy.csv",
    "reports/embedding_model/embedding_enrollment_size_experiment.csv",
    "reports/embedding_model/embedding_vs_softmax_comparison.csv",
    "reports/embedding_model/embedding_vs_softmax_summary.md",
    "reports/embedding_model/stage6_embedding_report.md",
)


def assert_file_exists(path: Path) -> None:
    """Проверить, что файл существует и не пустой."""
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Файл пуст: {path}")


def load_csv(path: Path) -> pd.DataFrame:
    """Загрузить CSV-файл и проверить, что он не пустой."""
    assert_file_exists(path)
    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"CSV-файл пуст: {path}")

    return dataframe


def check_required_artifacts(project_root: Path) -> None:
    """Проверить наличие обязательных артефактов stage 6."""
    for relative_path in REQUIRED_ARTIFACTS:
        assert_file_exists(project_root / relative_path)


def check_feature_columns(paths: EmbeddingModelPaths) -> None:
    """Проверить список признаков embedding-модели."""
    path = paths.reports_dir / "embedding_feature_columns.txt"
    feature_columns = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    if len(feature_columns) != 31:
        raise ValueError(f"Ожидалось 31 признаков, получено: {len(feature_columns)}.")

    service_columns = [
        column for column in feature_columns if column.startswith("Unnamed") or column == "user_id"
    ]

    if service_columns:
        raise ValueError(
            f"В список признаков попали служебные колонки: {', '.join(service_columns)}."
        )


def check_classifier_metrics(paths: EmbeddingModelPaths) -> None:
    """Проверить метрики embedding-классификатора."""
    metrics = load_csv(paths.reports_dir / "embedding_classifier_metrics.csv")

    expected_splits = {"train", "validation", "test"}
    actual_splits = set(metrics["split"])

    if actual_splits != expected_splits:
        raise ValueError(f"Некорректные split-ы classifier metrics: {actual_splits}")

    test_row = metrics.loc[metrics["split"] == "test"].iloc[0]

    if float(test_row["accuracy"]) < 0.90:
        raise ValueError("Test accuracy embedding-классификатора ниже 90%.")


def check_distance_diagnostics(paths: EmbeddingModelPaths) -> None:
    """Проверить диагностику embedding-расстояний."""
    diagnostics = load_csv(paths.reports_dir / "embedding_distance_diagnostics.csv")
    test_rows = diagnostics.loc[diagnostics["split"] == "test"].copy()
    best_row = test_rows.loc[test_rows["eer"].idxmin()]

    if best_row["distance_metric"] != "cosine":
        raise ValueError(
            "Лучшая test distance metric отличается от ожидаемой cosine: "
            f"{best_row['distance_metric']}."
        )

    if float(best_row["roc_auc"]) <= 0.98:
        raise ValueError("ROC AUC лучшей embedding-метрики слишком низкий.")


def check_threshold_policy(paths: EmbeddingModelPaths) -> None:
    """Проверить отчёт threshold policy."""
    policy = load_csv(paths.reports_dir / "embedding_threshold_policy.csv")

    if len(policy) != 18:
        raise ValueError(f"Ожидалось 18 строк threshold policy, получено: {len(policy)}.")

    test_rows = policy.loc[policy["split"] == "test"].copy()
    best_row = test_rows.loc[test_rows["balanced_error"].idxmin()]

    if best_row["distance_metric"] != "cosine" or best_row["policy"] != "per_user":
        raise ValueError(
            "Лучшая threshold policy отличается от cosine + per_user: "
            f"{best_row['distance_metric']} + {best_row['policy']}."
        )


def check_enrollment_experiment(paths: EmbeddingModelPaths) -> None:
    """Проверить эксперимент с количеством enrollment samples."""
    experiment = load_csv(paths.reports_dir / "embedding_enrollment_size_experiment.csv")

    target_rows = experiment.loc[
        (experiment["split"] == "test")
        & (experiment["distance_metric"] == "cosine")
        & (experiment["policy"] == "per_user")
    ].copy()

    required_sizes = {5, 10, 20, 30, 50}
    actual_sizes = set(target_rows["enrollment_samples"])

    if not required_sizes <= actual_sizes:
        raise ValueError(
            f"В enrollment experiment отсутствуют N: {sorted(required_sizes - actual_sizes)}."
        )

    by_n = target_rows.set_index("enrollment_samples")

    if float(by_n.loc[50, "frr"]) >= float(by_n.loc[5, "frr"]):
        raise ValueError("FRR при N=50 не ниже FRR при N=5.")


def check_comparison(paths: EmbeddingModelPaths) -> None:
    """Проверить итоговое сравнение с softmax baseline v2."""
    comparison = load_csv(paths.reports_dir / "embedding_vs_softmax_comparison.csv")

    baseline = comparison.loc[
        (comparison["approach"] == "softmax_based_verification")
        & (comparison["policy"] == "global")
    ].iloc[0]

    embedding_rows = comparison.loc[comparison["approach"] == "embedding_based_verification"].copy()
    best_embedding = embedding_rows.loc[embedding_rows["balanced_error"].idxmin()]

    if float(best_embedding["balanced_error"]) <= float(baseline["balanced_error"]):
        raise ValueError(
            "Итоговый вывод нарушен: embedding не должен быть лучше baseline "
            "в текущем эксперименте."
        )

    if float(best_embedding["frr"]) <= float(baseline["frr"]):
        raise ValueError("Итоговый вывод нарушен: embedding FRR не выше baseline FRR.")


def check_stage6_report(paths: EmbeddingModelPaths) -> None:
    """Проверить итоговый Markdown-отчёт stage 6."""
    path = paths.reports_dir / "stage6_embedding_report.md"
    assert_file_exists(path)

    text = path.read_text(encoding="utf-8")
    required_fragments = [
        "# Stage 6. Embedding-based verification",
        "## 6. Сравнение с softmax baseline v2",
        "softmax baseline v2 остаётся основной рабочей моделью",
        "Siamese / contrastive / triplet loss",
    ]

    missing_fragments = [fragment for fragment in required_fragments if fragment not in text]

    if missing_fragments:
        raise ValueError(
            f"В stage6_embedding_report.md отсутствуют фрагменты: {missing_fragments}."
        )


def main() -> None:
    """Выполнить контроль качества stage 6."""
    paths = EmbeddingModelPaths.from_source_dir()
    project_root = paths.project_root

    checks = [
        ("наличие артефактов", lambda: check_required_artifacts(project_root)),
        ("список 31 признака", lambda: check_feature_columns(paths)),
        ("метрики classifier", lambda: check_classifier_metrics(paths)),
        ("диагностика расстояний", lambda: check_distance_diagnostics(paths)),
        ("threshold policy", lambda: check_threshold_policy(paths)),
        ("enrollment experiment", lambda: check_enrollment_experiment(paths)),
        ("сравнение с baseline v2", lambda: check_comparison(paths)),
        ("итоговый Markdown-отчёт", lambda: check_stage6_report(paths)),
    ]

    print("Контроль качества stage 6.")

    for name, check in checks:
        check()
        print(f"[OK] {name}")

    print("Все проверки stage 6 пройдены.")


if __name__ == "__main__":
    main()
