"""Тесты orchestration-логики Stage 7 metric-learning tuning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.embedding_model.run_metric_learning_tuning import (
    TEST_SPLIT,
    build_candidate_paths,
    build_candidates,
    format_margin_for_name,
    save_candidate_plan,
    select_best_diagnostics_row,
    select_best_policy_row,
)


def test_format_margin_for_name_replaces_decimal_separator() -> None:
    """Margin должен преобразовываться в безопасное имя каталога."""
    assert format_margin_for_name(0.2) == "0p2"
    assert format_margin_for_name(1.0) == "1"


def test_build_candidates_uses_requested_margins() -> None:
    """План эксперимента должен учитывать заданные Siamese/Triplet margins."""
    candidates = build_candidates(
        objectives=["siamese", "triplet"],
        siamese_margins=[0.5],
        triplet_margins=[0.1, 0.2],
        embedding_dim=32,
        hidden_units=128,
        dropout_rate=0.2,
        learning_rate=1e-3,
        epochs=2,
        batch_size=512,
        patience=2,
        random_state=42,
        embedding_activation=None,
        l2_normalize=True,
        limit_candidates=None,
    )

    assert [candidate.name for candidate in candidates] == [
        "siamese_m0p5_l2",
        "triplet_m0p1_l2",
        "triplet_m0p2_l2",
    ]


def test_build_candidates_respects_limit() -> None:
    """Параметр limit_candidates нужен для smoke tuning."""
    candidates = build_candidates(
        objectives=["triplet"],
        siamese_margins=None,
        triplet_margins=[0.1, 0.2, 0.5],
        embedding_dim=32,
        hidden_units=128,
        dropout_rate=0.2,
        learning_rate=1e-3,
        epochs=2,
        batch_size=512,
        patience=2,
        random_state=42,
        embedding_activation=None,
        l2_normalize=True,
        limit_candidates=2,
    )

    assert [candidate.margin for candidate in candidates] == [0.1, 0.2]


def test_build_candidates_rejects_non_positive_limit() -> None:
    """Некорректный limit_candidates должен отклоняться явно."""
    with pytest.raises(ValueError, match="limit_candidates"):
        build_candidates(
            objectives=["triplet"],
            siamese_margins=None,
            triplet_margins=[0.1],
            embedding_dim=32,
            hidden_units=128,
            dropout_rate=0.2,
            learning_rate=1e-3,
            epochs=2,
            batch_size=512,
            patience=2,
            random_state=42,
            embedding_activation=None,
            l2_normalize=True,
            limit_candidates=0,
        )


def test_select_best_policy_row_uses_test_balanced_error(tmp_path: Path) -> None:
    """Лучший policy выбирается только по test split."""
    policy_path = tmp_path / "embedding_threshold_policy.csv"
    pd.DataFrame(
        [
            {
                "split": "validation",
                "distance": "cosine",
                "policy": "global",
                "far": 0.01,
                "frr": 0.01,
                "balanced_error": 0.01,
            },
            {
                "split": TEST_SPLIT,
                "distance": "cosine",
                "policy": "global",
                "far": 0.02,
                "frr": 0.20,
                "balanced_error": 0.11,
            },
            {
                "split": TEST_SPLIT,
                "distance": "euclidean",
                "policy": "per_user",
                "far": 0.015,
                "frr": 0.10,
                "balanced_error": 0.0575,
            },
        ]
    ).to_csv(policy_path, index=False)

    best_row = select_best_policy_row(policy_path)

    assert best_row["distance"] == "euclidean"
    assert best_row["policy"] == "per_user"


def test_select_best_diagnostics_row_uses_lowest_test_eer(tmp_path: Path) -> None:
    """Лучший diagnostics выбирается по минимальному EER на test split."""
    diagnostics_path = tmp_path / "embedding_distance_diagnostics.csv"
    pd.DataFrame(
        [
            {"split": "validation", "distance": "cosine", "eer": 0.01, "roc_auc": 0.99},
            {"split": TEST_SPLIT, "distance": "cosine", "eer": 0.08, "roc_auc": 0.97},
            {"split": TEST_SPLIT, "distance": "euclidean", "eer": 0.05, "roc_auc": 0.98},
        ]
    ).to_csv(diagnostics_path, index=False)

    best_row = select_best_diagnostics_row(diagnostics_path)

    assert best_row["distance"] == "euclidean"
    assert best_row["eer"] == pytest.approx(0.05)


def test_candidate_plan_is_saved(tmp_path: Path) -> None:
    """План tuning-эксперимента должен сохраняться в CSV."""
    candidates = build_candidates(
        objectives=["siamese"],
        siamese_margins=[0.5],
        triplet_margins=None,
        embedding_dim=32,
        hidden_units=128,
        dropout_rate=0.2,
        learning_rate=1e-3,
        epochs=2,
        batch_size=512,
        patience=2,
        random_state=42,
        embedding_activation=None,
        l2_normalize=True,
        limit_candidates=None,
    )
    output_path = tmp_path / "plan.csv"

    save_candidate_plan(candidates, output_path)

    saved_df = pd.read_csv(output_path)
    assert saved_df.loc[0, "candidate"] == "siamese_m0p5_l2"


def test_candidate_paths_are_isolated(tmp_path: Path) -> None:
    """Tuning-кандидаты не должны перезаписывать рабочие модели Stage 7."""
    candidate = build_candidates(
        objectives=["triplet"],
        siamese_margins=None,
        triplet_margins=[0.2],
        embedding_dim=32,
        hidden_units=128,
        dropout_rate=0.2,
        learning_rate=1e-3,
        epochs=2,
        batch_size=512,
        patience=2,
        random_state=42,
        embedding_activation=None,
        l2_normalize=True,
        limit_candidates=None,
    )[0]

    paths = build_candidate_paths(
        candidate=candidate,
        models_root=tmp_path / "models",
        reports_root=tmp_path / "reports",
    )

    assert paths.encoder_path.name == "encoder.keras"
    assert candidate.name in str(paths.encoder_path)
    assert candidate.name in str(paths.evaluation_dir)
