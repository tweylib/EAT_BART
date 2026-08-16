import csv

import pytest
import torch
from transformers import BartConfig

from eat_bart.modeling.eat_bart_model import build_eat_bart_model_from_config
from eat_bart.training.eat_signal import calculate_encoder_eat_signal, write_eat_signal_csv


def test_calculate_encoder_eat_signal_reports_per_head_ratios(tmp_path) -> None:
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
        modify_decoder_self_attention=False,
    )
    batch = {
        "input_ids": torch.tensor([[0, 5, 6, 2, 1], [0, 7, 8, 9, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
        "encoder_emotion_features": torch.tensor(
            [
                [
                    [0.0] * 8,
                    [0.2, 0.0, 0.1, 0.0, 0.3, 0.0, 0.4, 0.0],
                    [0.0] * 8,
                    [0.0] * 8,
                    [0.0] * 8,
                ],
                [
                    [0.0] * 8,
                    [0.0, 0.1, 0.0, 0.2, 0.0, 0.3, 0.0, 0.4],
                    [0.4, 0.0, 0.3, 0.0, 0.2, 0.0, 0.1, 0.0],
                    [0.0] * 8,
                    [0.0] * 8,
                ],
            ]
        ),
    }

    rows, summary = calculate_encoder_eat_signal(model, [batch])

    assert len(rows) == 2
    assert {row["head"] for row in rows} == {0, 1}
    assert all(row["r_h"] > 0 for row in rows)
    assert all(row["mean_abs_attention_score"] > 0 for row in rows)
    assert summary["overall_r_h"] > 0
    assert all(row["overall_r_h"] == pytest.approx(summary["overall_r_h"]) for row in rows)
    assert summary["emotion_token_coverage"] == pytest.approx(3 / 9)

    output_path = write_eat_signal_csv(tmp_path / "r_h.csv", rows)
    with output_path.open(encoding="utf-8", newline="") as file:
        saved_rows = list(csv.DictReader(file))
    assert len(saved_rows) == 2
    assert {int(row["head"]) for row in saved_rows} == {0, 1}
