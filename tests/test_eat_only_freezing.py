from __future__ import annotations

import torch
from transformers import BartConfig

from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.eat_bart_model import build_eat_bart_model_from_config
from eat_bart.training.train import (
    freeze_bart_for_eat_only_training,
    snapshot_eat_weights,
    validate_eat_weight_update,
    validate_eat_placement,
)


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


def test_decoder_only_freezing_and_placement_leave_exactly_decoder_w_trainable() -> None:
    config = BartConfig(
        vocab_size=99, d_model=16, encoder_layers=1, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
    )
    model = build_eat_bart_model_from_config(
        config,
        EATAttentionConfig(2, 768, 8, 0.1, "probability_mix"),
        modify_encoder_self_attention=False,
        modify_decoder_self_attention=True,
    )

    validate_eat_placement(model, modify_encoder=False, modify_decoder=True)
    freeze_bart_for_eat_only_training(model)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert len(trainable) == 4
    assert all("model.decoder.layers" in name for name in trainable)
    assert all(name.endswith(("w1_s", "w2_s")) for name in trainable)


def test_eat_weight_update_verification_rejects_stuck_weights() -> None:
    config = BartConfig(
        vocab_size=99, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
    )
    model = build_eat_bart_model_from_config(
        config,
        modify_encoder_self_attention=False,
        modify_decoder_self_attention=True,
    )
    initial = snapshot_eat_weights(model)

    with torch.no_grad():
        model.model.decoder.layers[0].self_attn.emotion_interaction.w1_s.add_(0.01)

    assert validate_eat_weight_update(model, initial) > 0.0

    stuck = snapshot_eat_weights(model)
    try:
        validate_eat_weight_update(model, stuck)
    except ValueError as error:
        assert "did not update" in str(error)
    else:
        raise AssertionError("Expected unchanged EAT weights to fail verification.")
