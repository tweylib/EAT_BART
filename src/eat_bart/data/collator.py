"""Batch collation for EAT-BART."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.models.bart.modeling_bart import shift_tokens_right

from eat_bart.data.emotion_features import SubwordStrategy, tokenize_texts_with_emotion_features
from eat_bart.data.emotion_lexicon import LexiconEntry
from eat_bart.data.contextual_emotion import tokenize_and_align_contextual_emotion


@dataclass
class EATBartDataCollator:
    """Collate question/response examples for EAT-BART."""

    tokenizer: Any
    lexicon: dict[str, LexiconEntry]
    max_source_length: int = 256
    max_target_length: int = 128
    subword_strategy: SubwordStrategy = "single"
    decoder_start_token_id: int | None = None
    emotion_feature_source: str = "lexicon"
    emotion_tokenizer: Any | None = None
    contextual_emotion_cache: dict[str, torch.Tensor] | None = None

    def __call__(self, examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """Return tensors for BART training.

        input_ids shape: [batch_size, source_seq_len]
        labels shape: [batch_size, target_seq_len]
        encoder_emotion_features shape: [batch_size, source_seq_len, 8]
        decoder_emotion_features shape: [batch_size, target_seq_len, 8]
        """
        questions = [example["question"] for example in examples]
        responses = [example["response"] for example in examples]

        contextual_batch: dict[str, torch.Tensor] = {}
        if self.emotion_feature_source == "goemotions_contextual":
            if self.contextual_emotion_cache is not None:
                source_encoded = self.tokenizer(
                    questions, max_length=self.max_source_length, padding=True,
                    truncation=True, return_tensors="pt",
                )
                seq_len = source_encoded["input_ids"].size(1)
                rows = []
                for text_value in questions:
                    if text_value not in self.contextual_emotion_cache:
                        raise KeyError("Question is missing from the contextual emotion cache.")
                    cached = self.contextual_emotion_cache[text_value]
                    row = torch.zeros(seq_len, cached.size(-1), dtype=cached.dtype)
                    copy_length = min(seq_len, cached.size(0))
                    row[:copy_length] = cached[:copy_length]
                    rows.append(row)
                encoder_emotion_features = None
                contextual_batch = {"aligned_emotion_hidden_states": torch.stack(rows)}
            elif self.emotion_tokenizer is None:
                raise ValueError("goemotions_contextual requires an emotion tokenizer.")
            else:
                source_encoded, emotion_encoded, alignment, diagnostics = tokenize_and_align_contextual_emotion(
                questions, self.tokenizer, self.emotion_tokenizer, self.max_source_length
                )
                if diagnostics.alignment_failures:
                    raise ValueError(f"Failed to align {diagnostics.alignment_failures} valid BART tokens.")
                encoder_emotion_features = None
                contextual_batch = {
                    "emotion_input_ids": emotion_encoded["input_ids"],
                    "emotion_attention_mask": emotion_encoded["attention_mask"],
                    "bart_to_emotion_alignment": alignment,
                }
        elif self.emotion_feature_source == "lexicon":
            source_encoded, encoder_emotion_features = tokenize_texts_with_emotion_features(
                texts=questions, tokenizer=self.tokenizer, lexicon=self.lexicon,
                max_length=self.max_source_length, padding=True, truncation=True,
                strategy=self.subword_strategy,
            )
        else:
            raise ValueError(f"Unsupported emotion_feature_source: {self.emotion_feature_source}")
        target_encoded, target_emotion_features = tokenize_texts_with_emotion_features(
            texts=responses,
            tokenizer=self.tokenizer,
            lexicon=self.lexicon,
            max_length=self.max_target_length,
            padding=True,
            truncation=True,
            strategy=self.subword_strategy,
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
        decoder_emotion_features = shift_emotion_features_right(
            target_emotion_features=target_emotion_features,
            decoder_input_ids=decoder_input_ids,
            pad_token_id=pad_token_id,
        )

        batch = {
            "input_ids": source_encoded["input_ids"],
            "attention_mask": source_encoded["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "decoder_emotion_features": decoder_emotion_features,
        }
        if encoder_emotion_features is not None:
            batch["encoder_emotion_features"] = encoder_emotion_features
        batch.update(contextual_batch)
        return batch

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


def shift_emotion_features_right(
    target_emotion_features: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Shift target emotion features to align with decoder_input_ids.

    target_emotion_features shape: [batch_size, target_seq_len, 8]
    decoder_input_ids shape: [batch_size, target_seq_len]
    return shape: [batch_size, target_seq_len, 8]
    """
    shifted = torch.zeros_like(target_emotion_features)
    shifted[:, 1:, :] = target_emotion_features[:, :-1, :]

    # pad_mask shape: [batch_size, target_seq_len, 1]
    pad_mask = decoder_input_ids.eq(pad_token_id).unsqueeze(-1)
    return shifted.masked_fill(pad_mask, 0.0)
