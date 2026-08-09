"""BART attention subclass/wrapper with EAT score injection."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.bart.modeling_bart import BartAttention

from eat_bart.modeling.eat_attention import EATAttentionConfig, EmotionInteraction


class EATBartAttention(BartAttention):
    """Modified BART self-attention with emotion-aware score injection.

    This class is intended for encoder and decoder self-attention only.
    Cross-attention must remain the original Hugging Face BartAttention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Any | None = None,
        layer_idx: int | None = None,
        eat_config: EATAttentionConfig | None = None,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_decoder=is_decoder,
            bias=bias,
            is_causal=is_causal,
            config=config,
            layer_idx=layer_idx,
        )
        self.emotion_interaction = EmotionInteraction(
            eat_config or EATAttentionConfig(num_heads=num_heads)
        )

    @classmethod
    def from_bart_attention(
        cls,
        attention: BartAttention,
        eat_config: EATAttentionConfig | None = None,
    ) -> "EATBartAttention":
        """Create EAT attention from an existing Hugging Face BART attention module."""
        module = cls(
            embed_dim=attention.embed_dim,
            num_heads=attention.num_heads,
            dropout=attention.dropout,
            is_decoder=attention.is_decoder,
            bias=attention.k_proj.bias is not None,
            is_causal=attention.is_causal,
            config=attention.config,
            layer_idx=attention.layer_idx,
            eat_config=eat_config,
        )
        module.q_proj.load_state_dict(attention.q_proj.state_dict())
        module.k_proj.load_state_dict(attention.k_proj.state_dict())
        module.v_proj.load_state_dict(attention.v_proj.state_dict())
        module.out_proj.load_state_dict(attention.out_proj.state_dict())
        return module

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        emotion_features: torch.Tensor | None = None,
        encoder_emotion_features: torch.Tensor | None = None,
        decoder_emotion_features: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run BART attention with optional EAT score injection.

        hidden_states shape: [batch_size, tgt_len, embed_dim]
        emotion_features shape: [batch_size, seq_len, 8]
        attention_mask shape: [batch_size, 1, tgt_len, src_len]
        """
        is_cross_attention = key_value_states is not None
        if is_cross_attention:
            raise ValueError("EATBartAttention must not be used for BART cross-attention.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # query_states shape: [batch_size, num_heads, tgt_len, head_dim]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        is_updated = False
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        current_states = hidden_states
        if past_key_values is not None and is_updated:
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            kv_shape = (*current_states.shape[:-1], -1, self.head_dim)
            # key_states shape: [batch_size, num_heads, src_len, head_dim]
            key_states = key_states.view(kv_shape).transpose(1, 2)
            # value_states shape: [batch_size, num_heads, src_len, head_dim]
            value_states = value_states.view(kv_shape).transpose(1, 2)

            if past_key_values is not None:
                key_states, value_states = curr_past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                )

        emotion_scores = None
        if emotion_features is None:
            emotion_features = decoder_emotion_features if self.is_decoder else encoder_emotion_features

        if emotion_features is None:
            emotion_features = getattr(self, "_eat_emotion_features", None)

        if emotion_features is not None:
            # emotion_scores shape: [batch_size, num_heads, tgt_len, src_len]
            emotion_scores = self.emotion_interaction(emotion_features)

        attn_output, attn_weights = eat_eager_attention_forward(
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            emotion_scores=emotion_scores,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # attn_output shape before projection: [batch_size, tgt_len, embed_dim]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


def eat_eager_attention_forward(
    module: EATBartAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    emotion_scores: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention with emotion scores added before mask application.

    query shape: [batch_size, num_heads, tgt_len, head_dim]
    key shape: [batch_size, num_heads, src_len, head_dim]
    value shape: [batch_size, num_heads, src_len, head_dim]
    return attn_output shape: [batch_size, tgt_len, num_heads, head_dim]
    """
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # attn_weights shape: [batch_size, num_heads, tgt_len, src_len]
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if emotion_scores is not None:
        attn_weights = module.emotion_interaction.combine_with_attention_scores(
            attention_scores=attn_weights,
            emotion_scores=emotion_scores,
        )

    if attention_mask is not None:
        mask = _attention_mask_to_bool(attention_mask)
        attn_weights = attn_weights.masked_fill(mask, torch.finfo(attn_weights.dtype).min)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # attn_output shape: [batch_size, num_heads, tgt_len, head_dim]
    attn_output = torch.matmul(attn_weights, value)
    # attn_output shape: [batch_size, tgt_len, num_heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _attention_mask_to_bool(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert BART attention masks to boolean masks for masked_fill.

    attention_mask shape: [batch_size, 1, tgt_len, src_len]
    return shape: [batch_size, 1, tgt_len, src_len]
    """
    if attention_mask.dtype == torch.bool:
        return attention_mask

    return attention_mask < 0
