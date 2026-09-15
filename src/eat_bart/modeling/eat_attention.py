"""Emotion-aware attention score utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

EATFormula = Literal["additive", "multiplicative", "probability_mix"]


@dataclass(frozen=True)
class EATAttentionConfig:
    """Configuration for emotion interaction scores."""

    num_heads: int
    emotion_dim: int = 8
    emotion_hidden_dim: int = 32
    alpha_init: float = 0.05
    formula: EATFormula = "additive"


class EmotionInteraction(nn.Module):
    """Compute per-head emotion interaction matrices."""

    def __init__(self, config: EATAttentionConfig) -> None:
        super().__init__()
        if config.formula not in ("additive", "multiplicative", "probability_mix"):
            raise ValueError(f"Unsupported EAT attention formula: {config.formula}")
        if not 0.0 <= config.alpha_init <= 1.0:
            raise ValueError("alpha must be in [0, 1].")

        self.config = config
        self.w1_s = nn.Parameter(
            torch.empty(config.num_heads, config.emotion_dim, config.emotion_hidden_dim)
        )
        self.w2_s = nn.Parameter(
            torch.empty(config.num_heads, config.emotion_dim, config.emotion_hidden_dim)
        )
        if config.formula == "probability_mix":
            self.register_buffer("alpha", torch.tensor(float(config.alpha_init)))
        else:
            self.alpha = nn.Parameter(torch.full((config.num_heads,), config.alpha_init))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize W1_s, W2_s, and alpha."""
        nn.init.xavier_uniform_(self.w1_s)
        nn.init.xavier_uniform_(self.w2_s)
        with torch.no_grad():
            self.alpha.fill_(self.config.alpha_init)

    def forward(self, emotion_features: torch.Tensor) -> torch.Tensor:
        """Compute S_h from emotion features.

        emotion_features shape: [batch_size, seq_len, 8]
        return shape: [batch_size, num_heads, seq_len, seq_len]
        """
        if emotion_features.dim() != 3:
            raise ValueError("emotion_features must have shape [batch_size, seq_len, emotion_dim].")
        if emotion_features.size(-1) != self.config.emotion_dim:
            raise ValueError(
                "emotion_features last dimension must match config.emotion_dim "
                f"({self.config.emotion_dim})."
            )

        # Cached contextual features may be float16 while frozen BART/EAT weights
        # remain float32 (for example during CPU evaluation).
        emotion_features = emotion_features.to(self.w1_s.dtype)
        # left shape: [batch_size, num_heads, seq_len, emotion_hidden_dim]
        left = torch.einsum("bse,hed->bhsd", emotion_features, self.w1_s)
        # right shape: [batch_size, num_heads, seq_len, emotion_hidden_dim]
        right = torch.einsum("bte,hed->bhtd", emotion_features, self.w2_s)
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.einsum("bhsd,bhtd->bhst", left, right)

        return scores / math.sqrt(self.config.emotion_hidden_dim)

    def combine_with_attention_scores(
        self,
        attention_scores: torch.Tensor,
        emotion_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Combine raw attention scores with S_h before masking.

        attention_scores shape: [batch_size, num_heads, tgt_len, src_len]
        emotion_scores shape: [batch_size, num_heads, tgt_len, src_len]
        return shape: [batch_size, num_heads, tgt_len, src_len]
        """
        if attention_scores.shape != emotion_scores.shape:
            raise ValueError("attention_scores and emotion_scores must have identical shapes.")

        # alpha shape after view: [1, num_heads, 1, 1]
        alpha = self.alpha.view(1, -1, 1, 1)

        if self.config.formula == "additive":
            return attention_scores + alpha * emotion_scores

        # Multiplicative ablation: A_eat = A * (I + alpha_h * S_h).
        # identity shape: [1, 1, tgt_len, src_len]
        identity = _attention_identity_like(attention_scores)
        return attention_scores * (identity + alpha * emotion_scores)


def _attention_identity_like(attention_scores: torch.Tensor) -> torch.Tensor:
    """Return an attention-position identity matrix broadcastable to attention scores.

    attention_scores shape: [batch_size, num_heads, tgt_len, src_len]
    return shape: [1, 1, tgt_len, src_len]
    """
    tgt_len = attention_scores.size(-2)
    src_len = attention_scores.size(-1)
    identity = torch.zeros(
        tgt_len,
        src_len,
        dtype=attention_scores.dtype,
        device=attention_scores.device,
    )
    diagonal_len = min(tgt_len, src_len)
    identity[torch.arange(diagonal_len), torch.arange(diagonal_len)] = 1.0

    return identity.view(1, 1, tgt_len, src_len)
