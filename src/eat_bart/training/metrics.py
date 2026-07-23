"""Metrics for response generation experiments."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def score_generation_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
    prediction_column: str = "generated_response",
    reference_column: str = "reference_response",
) -> dict[str, float]:
    """Score a generation CSV and optionally write one-row metrics CSV."""
    rows = _read_generation_rows(input_path)
    predictions = [row.get(prediction_column, "") for row in rows]
    references = [row.get(reference_column, "") for row in rows]
    metrics = compute_generation_metrics(predictions=predictions, references=references)

    if output_path is not None:
        _write_metrics_csv(output_path, metrics)

    return metrics


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute lightweight text-generation metrics.

    The metrics are intentionally dependency-free so Kaggle runs can score outputs
    without needing additional downloads.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length.")

    if not predictions:
        return {
            "num_examples": 0.0,
            "empty_prediction_rate": 0.0,
            "avg_prediction_tokens": 0.0,
            "avg_reference_tokens": 0.0,
            "rouge_1_f1": 0.0,
            "rouge_2_f1": 0.0,
            "rouge_l_f1": 0.0,
            "bleu_4": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
        }

    prediction_tokens = [_tokenize(text) for text in predictions]
    reference_tokens = [_tokenize(text) for text in references]

    return {
        "num_examples": float(len(predictions)),
        "empty_prediction_rate": _mean(
            1.0 if len(tokens) == 0 else 0.0 for tokens in prediction_tokens
        ),
        "avg_prediction_tokens": _mean(len(tokens) for tokens in prediction_tokens),
        "avg_reference_tokens": _mean(len(tokens) for tokens in reference_tokens),
        "rouge_1_f1": _mean(
            _ngram_f1(prediction, reference, n=1)
            for prediction, reference in zip(prediction_tokens, reference_tokens, strict=True)
        ),
        "rouge_2_f1": _mean(
            _ngram_f1(prediction, reference, n=2)
            for prediction, reference in zip(prediction_tokens, reference_tokens, strict=True)
        ),
        "rouge_l_f1": _mean(
            _rouge_l_f1(prediction, reference)
            for prediction, reference in zip(prediction_tokens, reference_tokens, strict=True)
        ),
        "bleu_4": _corpus_bleu(prediction_tokens, reference_tokens, max_order=4),
        "distinct_1": _distinct_n(prediction_tokens, n=1),
        "distinct_2": _distinct_n(prediction_tokens, n=2),
    }


def _read_generation_rows(input_path: str | Path) -> list[dict[str, str]]:
    with Path(input_path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_metrics_csv(output_path: str | Path, metrics: dict[str, float]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def _tokenize(text: Any) -> list[str]:
    if text is None:
        return []

    normalized = str(text).strip().lower()
    if normalized.lower() == "nan":
        return []

    return TOKEN_PATTERN.findall(normalized)


def _ngram_f1(prediction: list[str], reference: list[str], n: int) -> float:
    prediction_ngrams = _ngram_counts(prediction, n)
    reference_ngrams = _ngram_counts(reference, n)
    if not prediction_ngrams or not reference_ngrams:
        return 0.0

    overlap = sum((prediction_ngrams & reference_ngrams).values())
    precision = overlap / sum(prediction_ngrams.values())
    recall = overlap / sum(reference_ngrams.values())
    return _f1(precision, recall)


def _rouge_l_f1(prediction: list[str], reference: list[str]) -> float:
    if not prediction or not reference:
        return 0.0

    lcs_length = _lcs_length(prediction, reference)
    precision = lcs_length / len(prediction)
    recall = lcs_length / len(reference)
    return _f1(precision, recall)


def _corpus_bleu(
    prediction_tokens: list[list[str]],
    reference_tokens: list[list[str]],
    max_order: int,
) -> float:
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    prediction_length = 0
    reference_length = 0

    for prediction, reference in zip(prediction_tokens, reference_tokens, strict=True):
        prediction_length += len(prediction)
        reference_length += len(reference)
        for order in range(1, max_order + 1):
            prediction_ngrams = _ngram_counts(prediction, order)
            reference_ngrams = _ngram_counts(reference, order)
            matches_by_order[order - 1] += sum((prediction_ngrams & reference_ngrams).values())
            possible_matches_by_order[order - 1] += max(len(prediction) - order + 1, 0)

    if prediction_length == 0:
        return 0.0

    precisions = [
        (matches_by_order[index] + 1.0) / (possible_matches_by_order[index] + 1.0)
        for index in range(max_order)
    ]
    geo_mean = _geometric_mean(precisions)
    brevity_penalty = (
        1.0
        if prediction_length > reference_length
        else pow(2.718281828459045, 1.0 - reference_length / prediction_length)
    )
    return brevity_penalty * geo_mean


def _distinct_n(tokenized_texts: list[list[str]], n: int) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for tokens in tokenized_texts:
        all_ngrams.extend(_ngrams(tokens, n))

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(_ngrams(tokens, n))


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []

    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous_row = [0] * (len(right) + 1)
    for left_token in left:
        current_row = [0]
        for column, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current_row.append(previous_row[column - 1] + 1)
            else:
                current_row.append(max(previous_row[column], current_row[-1]))
        previous_row = current_row

    return previous_row[-1]


def _f1(precision: float, recall: float) -> float:
    if precision == 0.0 or recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def _geometric_mean(values: list[float]) -> float:
    product = 1.0
    for value in values:
        product *= value

    return product ** (1.0 / len(values))


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0

    return sum(float(value) for value in items) / len(items)
