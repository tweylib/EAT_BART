"""Aggregate multiple LLM judge summaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

JUDGE_METRIC_COLUMNS = ["llm_empathy", "llm_coherence", "llm_safety"]


def aggregate_judge_summaries(
    judges: list[dict[str, str]],
    output_path: str | Path,
) -> dict[str, float | str]:
    """Average judge summary metrics across completed judge runs."""
    rows = []
    for judge in judges:
        summary = _read_one_row_csv(judge["summary_path"])
        rows.append({"judge": judge["name"], **summary})

    aggregate = _aggregate_rows(rows)
    _write_aggregate(output_path, aggregate)
    return aggregate


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    completed_rows = [
        row
        for row in rows
        if float(row.get("num_judged_examples", 0.0)) > 0.0
    ]
    aggregate: dict[str, float | str] = {
        "num_judges": float(len(rows)),
        "num_completed_judges": float(len(completed_rows)),
        "judges": ";".join(str(row["judge"]) for row in rows),
        "completed_judges": ";".join(str(row["judge"]) for row in completed_rows),
    }
    for metric in JUDGE_METRIC_COLUMNS:
        values = [float(row[metric]) for row in completed_rows if metric in row]
        aggregate[f"{metric}_mean_across_judges"] = _mean(values)

    return aggregate


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
