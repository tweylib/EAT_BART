from __future__ import annotations

import csv
import json

import pytest

from eat_bart.training.metrics import (
    compute_generation_metrics,
    extract_validation_loss,
    score_generation_csv,
)


def test_compute_generation_metrics_scores_exact_match_high() -> None:
    metrics = compute_generation_metrics(
        predictions=["therapy can help with anxiety"],
        references=["therapy can help with anxiety"],
    )

    assert metrics["num_examples"] == 1.0
    assert metrics["empty_prediction_rate"] == 0.0
    assert "bertscore_f1" in metrics
    assert metrics["rouge_l_f1"] == pytest.approx(1.0)
    assert metrics["bleu_4"] == pytest.approx(1.0)
    assert metrics["distinct_2"] == pytest.approx(1.0)


def test_compute_generation_metrics_treats_nan_string_as_empty() -> None:
    metrics = compute_generation_metrics(
        predictions=["NaN", ""],
        references=["a useful response", "another useful response"],
    )

    assert metrics["empty_prediction_rate"] == pytest.approx(1.0)
    assert metrics["avg_prediction_tokens"] == pytest.approx(0.0)
    assert metrics["rouge_l_f1"] == pytest.approx(0.0)


def test_compute_generation_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_generation_metrics(predictions=["one"], references=["one", "two"])


def test_score_generation_csv_writes_metrics_file(tmp_path) -> None:
    input_path = tmp_path / "generations.csv"
    output_path = tmp_path / "metrics.csv"
    with input_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["question", "reference_response", "generated_response"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "question": "How can I manage anxiety?",
                "reference_response": "Therapy can help with anxiety.",
                "generated_response": "Therapy can help with anxiety.",
            }
        )

    metrics = score_generation_csv(input_path=input_path, output_path=output_path)

    assert output_path.exists()
    assert metrics["num_examples"] == 1.0


def test_extract_validation_loss_reads_best_and_last_checkpoint_values(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoint-10"
    checkpoint_dir.mkdir()
    state_path = checkpoint_dir / "trainer_state.json"
    state_path.write_text(
        json.dumps(
            {
                "log_history": [
                    {"eval_loss": 2.5, "epoch": 1},
                    {"eval_loss": 1.5, "epoch": 2},
                    {"eval_loss": 1.8, "epoch": 3},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert extract_validation_loss(tmp_path, mode="best") == pytest.approx(1.5)
    assert extract_validation_loss(tmp_path, mode="last") == pytest.approx(1.8)
