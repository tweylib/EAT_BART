from __future__ import annotations

import json

import pytest
import torch

from eat_bart.training.eat_learning_diagnostic import (
    compare_condition_summaries,
    compare_generation_rows,
    probability_statistics,
    relative_parameter_update,
    resolve_diagnostic_checkpoints,
    resolve_initialization_source,
    shuffle_emotion_features,
)


def test_resolve_diagnostic_checkpoints_uses_best_and_last(tmp_path) -> None:
    first = tmp_path / "checkpoint-10"
    last = tmp_path / "checkpoint-40"
    first.mkdir()
    last.mkdir()
    (first / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 10,
                "best_model_checkpoint": "/old/location/checkpoint-10",
            }
        ),
        encoding="utf-8",
    )
    (last / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 40,
                "best_model_checkpoint": "/old/location/checkpoint-10",
            }
        ),
        encoding="utf-8",
    )

    best, final = resolve_diagnostic_checkpoints(tmp_path)

    assert best == first
    assert final == last


def test_initialization_source_falls_back_to_best_eat_checkpoint(tmp_path) -> None:
    best = tmp_path / "checkpoint-10"
    best.mkdir()
    missing_baseline = tmp_path / "missing-baseline"

    source, kind = resolve_initialization_source(
        {
            "baseline_checkpoint_path": str(missing_baseline),
            "baseline_artifact_name": "bart_baseline_comparable",
        },
        best,
    )

    assert source == best
    assert kind == "best_eat_checkpoint_frozen_bart_fallback"


def test_shuffle_emotion_features_preserves_vectors_and_zero_special_tokens() -> None:
    features = torch.tensor(
        [[[0.0], [1.0], [2.0], [3.0], [0.0]], [[0.0], [4.0], [5.0], [0.0], [0.0]]]
    )
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])

    shuffled = shuffle_emotion_features(features, mask, seed=7)

    assert shuffled[:, 0].eq(0).all()
    assert shuffled[0, 4].eq(0).all()
    assert shuffled[1, 3:].eq(0).all()
    assert sorted(shuffled[0, 1:4, 0].tolist()) == [1.0, 2.0, 3.0]
    assert sorted(shuffled[1, 1:3, 0].tolist()) == [4.0, 5.0]
    assert torch.equal(shuffled, shuffle_emotion_features(features, mask, seed=7))


def test_probability_statistics_identifies_uniform_emotion_attention() -> None:
    standard_scores = torch.tensor([[[[4.0, 0.0], [0.0, 4.0]]]])
    emotion_scores = torch.zeros_like(standard_scores)
    mask = torch.ones(1, 2, dtype=torch.long)

    statistics, count = probability_statistics(standard_scores, emotion_scores, mask)

    assert count == 2
    assert statistics["normalized_emotion_entropy"].item() / count == pytest.approx(1.0)
    assert statistics["emotion_kl_from_uniform"].item() / count == pytest.approx(0.0)
    assert statistics["emotion_standard_js"].item() > 0.0
    assert statistics["emotion_standard_total_variation"].item() > 0.0


def test_relative_parameter_update() -> None:
    before = torch.tensor([3.0, 4.0])
    after = torch.tensor([6.0, 8.0])

    assert relative_parameter_update(before, after) == pytest.approx(1.0)


def test_compare_condition_summaries_reports_signed_loss_deltas() -> None:
    base_metrics = {
        "mean_head_normalized_emotion_entropy": 0.8,
        "mean_head_emotion_kl_from_uniform": 0.2,
        "mean_head_emotion_standard_js": 0.3,
        "mean_head_emotion_standard_total_variation": 0.4,
    }
    summaries = {
        "initial_aligned": {"teacher_forced_loss_subset": 2.0, **base_metrics},
        "best_aligned": {"teacher_forced_loss_subset": 1.5, **base_metrics},
        "best_shuffled": {"teacher_forced_loss_subset": 1.7, **base_metrics},
        "final_aligned": {"teacher_forced_loss_subset": 1.6, **base_metrics},
    }

    comparison = compare_condition_summaries(summaries)

    assert comparison["best_loss_minus_initial_loss"] == pytest.approx(-0.5)
    assert comparison["final_loss_minus_best_loss"] == pytest.approx(0.1)
    assert comparison["shuffled_loss_minus_aligned_loss"] == pytest.approx(0.2)


def test_compare_generation_rows_uses_exact_token_sequences() -> None:
    best = [
        {
            "question": "q1",
            "reference_response": "r1",
            "generated_response": "same",
            "generated_token_ids": "1 2",
        },
        {
            "question": "q2",
            "reference_response": "r2",
            "generated_response": "first",
            "generated_token_ids": "3 4",
        },
    ]
    final = [
        {
            "question": "q1",
            "reference_response": "r1",
            "generated_response": "same",
            "generated_token_ids": "1 2",
        },
        {
            "question": "q2",
            "reference_response": "r2",
            "generated_response": "second",
            "generated_token_ids": "3 5",
        },
    ]

    summary, rows = compare_generation_rows(best, final)

    assert summary["different_token_sequences"] == 1.0
    assert summary["different_token_sequence_rate"] == pytest.approx(0.5)
    assert rows[0]["same_token_sequence"] is True
    assert rows[1]["same_token_sequence"] is False
