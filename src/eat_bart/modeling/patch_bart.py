"""Utilities for replacing BART self-attention modules."""

from __future__ import annotations

from transformers.models.bart.modeling_bart import BartAttention

from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.eat_bart_attention import EATBartAttention
from eat_bart.modeling.eat_bart_forward import enable_eat_forwarding


def patch_bart_self_attention(
    model: object,
    eat_config: EATAttentionConfig | None = None,
    modify_encoder_self_attention: bool = True,
    modify_decoder_self_attention: bool = True,
) -> object:
    """Patch encoder and decoder self-attention while leaving cross-attention unchanged."""
    enable_eat_forwarding(model)
    bart_model = getattr(model, "model", model)
    encoder = getattr(bart_model, "encoder")
    decoder = getattr(bart_model, "decoder")

    if modify_encoder_self_attention:
        for layer in encoder.layers:
            layer.self_attn = _convert_self_attention(layer.self_attn, eat_config)

    if modify_decoder_self_attention:
        for layer in decoder.layers:
            layer.self_attn = _convert_self_attention(layer.self_attn, eat_config)

    return model


def _convert_self_attention(
    attention: BartAttention,
    eat_config: EATAttentionConfig | None,
) -> EATBartAttention:
    """Convert one BART self-attention module to EAT self-attention."""
    if isinstance(attention, EATBartAttention):
        return attention

    return EATBartAttention.from_bart_attention(attention, eat_config=eat_config)
