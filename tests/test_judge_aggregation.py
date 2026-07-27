from __future__ import annotations

import csv

import pytest

from eat_bart.training.judge_aggregation import aggregate_judge_summaries


def test_aggregate_judge_summaries_averages_completed_judges(tmp_path) -> None:
    llama_path = tmp_path / "llama.csv"
    second_judge_path = tmp_path / "second_judge.csv"
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
        second_judge_path,
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
            {"name": "second-judge", "summary_path": str(second_judge_path)},
        ],
        output_path=output_path,
    )

    assert output_path.exists()
    assert aggregate["num_eligible_judges"] == pytest.approx(2.0)
    assert aggregate["llm_empathy_mean_across_judges"] == pytest.approx(3.0)
    assert aggregate["llm_coherence_mean_across_judges"] == pytest.approx(4.0)
    assert aggregate["llm_safety_mean_across_judges"] == pytest.approx(5.0)


def test_aggregate_judge_summaries_weights_partial_judges_by_completed_examples(tmp_path) -> None:
    llama_path = tmp_path / "llama.csv"
    gpt_oss_path = tmp_path / "gpt_oss.csv"
    _write_summary(
        llama_path,
        {
            "num_requested_examples": "100",
            "num_judged_examples": "100",
            "llm_empathy": "4",
            "llm_coherence": "4",
            "llm_safety": "5",
        },
    )
    _write_summary(
        gpt_oss_path,
        {
            "num_requested_examples": "100",
            "num_judged_examples": "12",
            "llm_empathy": "2",
            "llm_coherence": "2",
            "llm_safety": "4",
        },
    )

    output_path = tmp_path / "aggregate.csv"
    aggregate = aggregate_judge_summaries(
        judges=[
            {"name": "llama", "summary_path": str(llama_path)},
            {"name": "gpt-oss", "summary_path": str(gpt_oss_path)},
        ],
        output_path=output_path,
    )

    assert aggregate["num_eligible_judges"] == pytest.approx(2.0)
    assert aggregate["total_judged_examples"] == pytest.approx(112.0)
    assert aggregate["eligible_judges"] == "llama;gpt-oss"
    assert aggregate["skipped_judges"] == ""
    assert aggregate["llm_empathy_mean_across_judges"] == pytest.approx(3.0)
    assert aggregate["llm_empathy_weighted_by_examples"] == pytest.approx(
        ((4.0 * 100.0) + (2.0 * 12.0)) / 112.0
    )


def _write_summary(path, row: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
