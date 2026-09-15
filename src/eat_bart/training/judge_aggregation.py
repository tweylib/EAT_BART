"""Aggregate multiple LLM judge summaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

JUDGE_METRIC_COLUMNS = ["llm_empathy", "llm_coherence", "llm_safety"]


def aggregate_judge_summaries(
    judges: list[dict[str, str]],
    output_path: str | Path,
    min_completion_rate: float = 0.0,
    min_judged_examples: int = 1,
    require_all_judges: bool = False,
) -> dict[str, float | str]:
    """Average judge summary metrics across completed judge runs."""
    rows = []
    for judge in judges:
        summary = _read_one_row_csv(judge["summary_path"])
        rows.append({"judge": judge["name"], **summary})

    aggregate = _aggregate_rows(
        rows,
        min_completion_rate=min_completion_rate,
        min_judged_examples=min_judged_examples,
    )
    if require_all_judges and aggregate["num_eligible_judges"] != float(len(judges)):
        raise ValueError(
            "Not all configured judges met the completion requirements: "
            f"eligible={int(aggregate['num_eligible_judges'])}, configured={len(judges)}."
        )
    _write_aggregate(output_path, aggregate)
    return aggregate


def _aggregate_rows(
    rows: list[dict[str, Any]],
    min_completion_rate: float = 0.0,
    min_judged_examples: int = 1,
) -> dict[str, float | str]:
    eligible_rows = [
        row
        for row in rows
        if _is_eligible_judge(
            row,
            min_completion_rate=min_completion_rate,
            min_judged_examples=min_judged_examples,
        )
    ]
    skipped_rows = [
        row
        for row in rows
        if not _is_eligible_judge(
            row,
            min_completion_rate=min_completion_rate,
            min_judged_examples=min_judged_examples,
        )
    ]
    aggregate: dict[str, float | str] = {
        "num_judges": float(len(rows)),
        "num_eligible_judges": float(len(eligible_rows)),
        "total_judged_examples": sum(
            float(row.get("num_judged_examples", 0.0)) for row in eligible_rows
        ),
        "min_completion_rate": float(min_completion_rate),
        "min_judged_examples": float(min_judged_examples),
        "judges": ";".join(str(row["judge"]) for row in rows),
        "eligible_judges": ";".join(str(row["judge"]) for row in eligible_rows),
        "skipped_judges": ";".join(str(row["judge"]) for row in skipped_rows),
        "judge_completion_rates": ";".join(
            f"{row['judge']}:{_completion_rate(row):.4f}" for row in rows
        ),
    }
    for metric in JUDGE_METRIC_COLUMNS:
        values = [float(row[metric]) for row in eligible_rows if metric in row]
        aggregate[f"{metric}_mean_across_judges"] = _mean(values)
        aggregate[f"{metric}_weighted_by_examples"] = _weighted_mean_by_judged_examples(
            rows=eligible_rows,
            metric=metric,
        )

    return aggregate


def _is_eligible_judge(
    row: dict[str, Any],
    min_completion_rate: float,
    min_judged_examples: int,
) -> bool:
    judged_examples = float(row.get("num_judged_examples", 0.0))
    completion_rate = _completion_rate(row)
    return judged_examples >= min_judged_examples and completion_rate >= min_completion_rate


def _completion_rate(row: dict[str, Any]) -> float:
    judged_examples = float(row.get("num_judged_examples", 0.0))
    requested_examples = float(row.get("num_requested_examples", judged_examples))
    return judged_examples / requested_examples if requested_examples > 0 else 0.0


def _weighted_mean_by_judged_examples(rows: list[dict[str, Any]], metric: str) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        if metric not in row:
            continue
        weight = float(row.get("num_judged_examples", 0.0))
        weighted_sum += float(row[metric]) * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    return weighted_sum / total_weight


def _read_one_row_csv(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 1:
        raise ValueError(f"Expected exactly one summary row in {path}, found {len(rows)}.")

    return rows[0]


def _write_aggregate(output_path: str | Path, aggregate: dict[str, float | str]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate))
        writer.writeheader()
        writer.writerow(aggregate)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)
