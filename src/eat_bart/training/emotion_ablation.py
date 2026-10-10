"""Evaluation-only transformations for aligned emotion features."""

from __future__ import annotations

from typing import Literal

import torch

EmotionFeatureCondition = Literal["aligned", "shuffled", "zero"]


def apply_emotion_feature_condition(
    features: torch.Tensor,
    attention_mask: torch.Tensor,
    condition: EmotionFeatureCondition | str = "aligned",
    *,
    seed: int = 1729,
    example_offset: int = 0,
) -> torch.Tensor:
    """Return aligned, within-example shuffled, or zero emotion features.

    Shuffling is deterministic per dataset example and therefore independent of
    evaluation batch size. Only nonzero vectors at valid BART token positions are
    permuted; padding and zero-valued special-token rows remain untouched.
    """
    if features.dim() != 3:
        raise ValueError("features must have shape [batch_size, seq_len, emotion_dim].")
    if attention_mask.shape != features.shape[:2]:
        raise ValueError("attention_mask must match the first two feature dimensions.")

    normalized_condition = str(condition).lower()
    if normalized_condition == "aligned":
        return features
    if normalized_condition == "zero":
        return torch.zeros_like(features)
    if normalized_condition != "shuffled":
        raise ValueError(
            "evaluation.emotion_feature_condition must be 'aligned', 'shuffled', or 'zero'."
        )

    shuffled = features.clone()
    features_cpu = features.detach().cpu()
    mask_cpu = attention_mask.detach().cpu().bool()
    for row in range(features.size(0)):
        emotional = features_cpu[row].abs().sum(dim=-1).gt(0)
        indices = torch.nonzero(mask_cpu[row] & emotional, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + int(example_offset) + row
        )
        permutation = indices[torch.randperm(indices.numel(), generator=generator)]
        shuffled[row, indices.to(shuffled.device)] = features[
            row, permutation.to(features.device)
        ]
    return shuffled
