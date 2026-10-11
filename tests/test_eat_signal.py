import csv

import pytest
import torch
from transformers import BartConfig

from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.eat_bart_model import build_eat_bart_model_from_config
from eat_bart.training.eat_signal import (
    calculate_decoder_eat_signal,
    write_eat_signal_csv,
)


def test_calculate_decoder_eat_signal_uses_only_causal_token_pairs(tmp_path) -> None:
    model = build_eat_bart_model_from_config(
        BartConfig(
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            vocab_size=99,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            decoder_start_token_id=2,
            dropout=0.0,
            attention_dropout=0.0,
        ),
        EATAttentionConfig(2, 8, 4, 0.1, "probability_mix"),
        modify_encoder_self_attention=False,
        modify_decoder_self_attention=True,
    )
    batch = {
        "input_ids": torch.tensor([[0, 5, 6, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "decoder_input_ids": torch.tensor([[2, 0, 7, 8]]),
        "decoder_attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[0, 7, 8, 2]]),
        "decoder_emotion_features": torch.randn(1, 4, 8),
    }

    rows, summary = calculate_decoder_eat_signal(model, [batch])

    assert len(rows) == 2
    assert all(row["side"] == "decoder" for row in rows)
    assert all(row["r_h"] > 0 for row in rows)
    assert summary["overall_r_h"] > 0

    output_path = write_eat_signal_csv(tmp_path / "r_h.csv", rows)
    with output_path.open(encoding="utf-8", newline="") as file:
        saved_rows = list(csv.DictReader(file))
    assert len(saved_rows) == 2
    assert {int(row["head"]) for row in saved_rows} == {0, 1}


def test_decoder_probability_compose_signal_measures_probability_delta() -> None:
    model = build_eat_bart_model_from_config(
        BartConfig(
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            vocab_size=99,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            decoder_start_token_id=2,
            dropout=0.0,
            attention_dropout=0.0,
        ),
        EATAttentionConfig(2, 8, 4, 0.05, "probability_compose"),
        modify_encoder_self_attention=False,
        modify_decoder_self_attention=True,
    )
    batch = {
        "input_ids": torch.tensor([[0, 5, 6, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "decoder_input_ids": torch.tensor([[2, 0, 7, 8]]),
        "decoder_attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[0, 7, 8, 2]]),
        "decoder_emotion_features": torch.randn(1, 4, 8),
    }

    rows, summary = calculate_decoder_eat_signal(model, [batch])

    assert all(row["signal_space"] == "probability_delta" for row in rows)
    assert summary["overall_r_h"] > 0


def test_decoder_signal_rejects_invalid_batch_limit() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        calculate_decoder_eat_signal(torch.nn.Linear(1, 1), [], max_batches=0)
