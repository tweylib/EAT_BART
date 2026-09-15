from __future__ import annotations

import torch
from transformers import BartConfig

from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.eat_bart_model import build_eat_bart_model_from_config
from eat_bart.training.train import freeze_bart_for_eat_only_training


def test_only_w1_and_w2_remain_trainable() -> None:
    config = BartConfig(
        vocab_size=99, d_model=16, encoder_layers=2, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
    )
    model = build_eat_bart_model_from_config(
        config, EATAttentionConfig(2, 768, 8, 0.1, "probability_mix"), True, False
    )
    freeze_bart_for_eat_only_training(model)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all(name.endswith(("w1_s", "w2_s")) for name in trainable)
    assert len([name for name in trainable if name.endswith("w1_s")]) == 2
    assert len([name for name in trainable if name.endswith("w2_s")]) == 2
