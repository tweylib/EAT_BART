"""Tokenizer loading helpers."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


def load_bart_tokenizer(
    model_name: str,
    local_files_only: bool = False,
    add_prefix_space: bool = True,
) -> Any:
    """Load a fast BART tokenizer configured for pre-tokenized word inputs."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=local_files_only,
        add_prefix_space=add_prefix_space,
    )
    if not tokenizer.is_fast:
        raise ValueError("EAT-BART emotion alignment requires a fast Hugging Face tokenizer.")

    return tokenizer
