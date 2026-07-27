"""Batch collation for the standard BART baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.models.bart.modeling_bart import shift_tokens_right


@dataclass
class BartDataCollator:
    """Tokenize and collate question/response examples for standard BART."""

    tokenizer: Any
    max_source_length: int = 256
    max_target_length: int = 128
    decoder_start_token_id: int | None = None

    def __call__(self, examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """Return tensors for BART training.

        input_ids shape: [batch_size, source_seq_len]
        labels shape: [batch_size, target_seq_len]
        """
        questions = [example["question"] for example in examples]
        responses = [example["response"] for example in examples]

        source_encoded = self.tokenizer(
            questions,
            max_length=self.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target_encoded = self.tokenizer(
            responses,
            max_length=self.max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        pad_token_id = self._get_pad_token_id()
        decoder_start_token_id = self._get_decoder_start_token_id()

        labels = target_encoded["input_ids"].clone()
        labels[labels == pad_token_id] = -100

        decoder_input_ids = shift_tokens_right(
            labels,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )
        decoder_attention_mask = decoder_input_ids.ne(pad_token_id).long()

        return {
            "input_ids": source_encoded["input_ids"],
            "attention_mask": source_encoded["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }

    def _get_pad_token_id(self) -> int:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id.")
        return int(pad_token_id)

    def _get_decoder_start_token_id(self) -> int:
        if self.decoder_start_token_id is not None:
            return self.decoder_start_token_id
        if self.tokenizer.eos_token_id is None:
            raise ValueError("BART decoder_start_token_id is required when tokenizer has no eos_token_id.")
        return int(self.tokenizer.eos_token_id)
