"""Dataset wrappers for mental-health response generation."""

from __future__ import annotations

import csv
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_TAG_WITH_SPACES_PATTERN = re.compile(r"<\s*/?\s*[^>]+?\s*>")
CSS_DECLARATION_PATTERN = re.compile(
    r"\b[a-zA-Z-]+\s*:\s*[^;<>]+(?:;|(?=\s+[a-zA-Z-]+\s*:)|$)"
)
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class QuestionResponseExample:
    """One mental-health question/response pair."""

    question: str
    response: str


class MentalHealthResponseDataset(Dataset[dict[str, str]]):
    """Dataset of mental-health question/response pairs."""

    def __init__(self, examples: list[QuestionResponseExample]) -> None:
        self.examples = examples

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        question_column: str = "question",
        response_column: str = "response",
        limit: int | None = None,
    ) -> "MentalHealthResponseDataset":
        """Load question/response pairs from a CSV file."""
        examples = load_question_response_csv(
            path=path,
            question_column=question_column,
            response_column=response_column,
            limit=limit,
        )
        return cls(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str]:
        example = self.examples[index]
        return {"question": example.question, "response": example.response}


def load_question_response_csv(
    path: str | Path,
    question_column: str = "question",
    response_column: str = "response",
    limit: int | None = None,
) -> list[QuestionResponseExample]:
    """Load question/response rows from CSV."""
    _raise_csv_field_size_limit()
    dataset_path = Path(path)
    examples: list[QuestionResponseExample] = []

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        _validate_columns(fieldnames, question_column, response_column)

        for row in reader:
            question = _clean_text(row[question_column])
            response = _clean_text(row[response_column])
            if not question or not response:
                continue

            examples.append(QuestionResponseExample(question=question, response=response))
            if limit is not None and len(examples) >= limit:
                break

    return examples


def _validate_columns(
    fieldnames: list[str],
    question_column: str,
    response_column: str,
) -> None:
    missing_columns = [
        column for column in (question_column, response_column) if column not in fieldnames
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        available = ", ".join(fieldnames)
        raise ValueError(f"Dataset is missing columns: {missing}. Available columns: {available}")


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value)).replace("\r\n", "\n")
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = HTML_TAG_WITH_SPACES_PATTERN.sub(" ", text)
    text = CSS_DECLARATION_PATTERN.sub(" ", text)
    text = _normalize_detached_punctuation(text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _normalize_detached_punctuation(text: str) -> str:
    """Undo common tokenizer-like spacing in dirty HTML-derived text."""
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([])}])", r"\1", text)
    text = re.sub(r"([A-Za-z])\s+'\s+([A-Za-z])", r"\1'\2", text)
    text = re.sub(r"\bpost\s*-\s*traumatic\b", "post-traumatic", text, flags=re.IGNORECASE)
    return text


def split_dataset(
    dataset: Dataset[Any],
    validation_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[Subset[Any], Subset[Any], Subset[Any]]:
    """Split a dataset into train, validation, and test subsets."""
    if not 0.0 <= validation_size < 1.0:
        raise ValueError("validation_size must be in [0.0, 1.0).")
    if not 0.0 <= test_size < 1.0:
        raise ValueError("test_size must be in [0.0, 1.0).")
    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.0.")

    dataset_size = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    test_count = int(dataset_size * test_size)
    validation_count = int(dataset_size * validation_size)
    train_count = dataset_size - validation_count - test_count

    train_indices = indices[:train_count]
    validation_indices = indices[train_count : train_count + validation_count]
    test_indices = indices[train_count + validation_count :]

    return Subset(dataset, train_indices), Subset(dataset, validation_indices), Subset(dataset, test_indices)


def _raise_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10
