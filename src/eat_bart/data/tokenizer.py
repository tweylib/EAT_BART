"""Tokenizer loading helpers."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


def load_bart_tokenizer(
    model_name: str,
    local_files_only: bool = False,
    add_prefix_space: bool = False,
) -> Any:
    """Load the standard tokenizer for BART text inputs."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
        add_prefix_space=add_prefix_space,
    )
    return tokenizer
