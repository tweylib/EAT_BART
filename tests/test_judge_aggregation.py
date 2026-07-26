from __future__ import annotations

import csv

import pytest

from eat_bart.training.judge_aggregation import aggregate_judge_summaries


def test_aggregate_judge_summaries_averages_completed_judges(tmp_path) -> None:
    llama_path = tmp_path / "llama.csv"
    qwen_path = tmp_path / "qwen.csv"
    _write_summary(
        llama_path,
        {
            "num_judged_examples": "10",
            "llm_empathy": "4",
            "llm_coherence": "3",
            "llm_safety": "5",
        },
    )
    _write_summary(
        qwen_path,
        {
            "num_judged_examples": "10",
            "llm_empathy": "2",
            "llm_coherence": "5",
            "llm_safety": "5",
        },
    )

    output_path = tmp_path / "aggregate.csv"
    aggregate = aggregate_judge_summaries(
        judges=[
            {"name": "llama", "summary_path": str(llama_path)},
            {"name": "qwen", "summary_path": str(qwen_path)},
        ],
        output_path=output_path,
    )

    assert output_path.exists()
    assert aggregate["num_completed_judges"] == pytest.approx(2.0)
    assert aggregate["llm_empathy_mean_across_judges"] == pytest.approx(3.0)
    assert aggregate["llm_coherence_mean_across_judges"] == pytest.approx(4.0)
    assert aggregate["llm_safety_mean_across_judges"] == pytest.approx(5.0)


def _write_summary(path, row: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
