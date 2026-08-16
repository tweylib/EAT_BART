"""Post-training diagnostics for the effective EAT attention contribution."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from eat_bart.modeling.eat_bart_attention import EATBartAttention


@dataclass
class _HeadAccumulator:
    attention_abs_sum: torch.Tensor
    emotion_abs_sum: torch.Tensor
    valid_pair_count: int = 0


def calculate_encoder_eat_signal(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    max_batches: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Calculate per-head r_h on valid source-token pairs.

    r_h = mean(abs(alpha_h * S_h)) / mean(abs(A_h)), where A_h is the
    scaled query-key score before emotion modulation and masking.
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1 when provided.")

    encoder = _get_encoder(model)
    attention_modules = {
        name: module
        for name, module in encoder.named_modules()
        if isinstance(module, EATBartAttention) and not module.is_decoder
    }
    if not attention_modules:
        raise ValueError("No encoder EAT attention modules were found for r_h analysis.")

    accumulators = {
        name: _HeadAccumulator(
            attention_abs_sum=torch.zeros(module.num_heads, dtype=torch.float64),
            emotion_abs_sum=torch.zeros(module.num_heads, dtype=torch.float64),
        )
        for name, module in attention_modules.items()
    }
    current_attention_mask: torch.Tensor | None = None

    def capture_scores(
        name: str,
        module: EATBartAttention,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        emotion_features = getattr(module, "_eat_emotion_features", None)
        if hidden_states is None or emotion_features is None or current_attention_mask is None:
            return

        # query/key shape: [batch_size, num_heads, seq_len, head_dim]
        hidden_shape = (*hidden_states.shape[:-1], module.num_heads, module.head_dim)
        query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # attention_scores shape: [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
        emotion_scores = module.emotion_interaction(emotion_features)
        # scaled_emotion_scores shape: [batch_size, num_heads, seq_len, seq_len]
        alpha = module.emotion_interaction.alpha.view(1, -1, 1, 1)
        scaled_emotion_scores = alpha * emotion_scores

        # valid_pairs shape: [batch_size, 1, seq_len, seq_len]
        valid_tokens = current_attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        valid_pairs = valid_tokens[:, None, :, None] & valid_tokens[:, None, None, :]
        valid_pairs_float = valid_pairs.to(dtype=attention_scores.dtype)

        accumulator = accumulators[name]
        accumulator.attention_abs_sum += (
            (attention_scores.abs() * valid_pairs_float)
            .sum(dim=(0, 2, 3))
            .detach()
            .cpu()
            .to(torch.float64)
        )
        accumulator.emotion_abs_sum += (
            (scaled_emotion_scores.abs() * valid_pairs_float)
            .sum(dim=(0, 2, 3))
            .detach()
            .cpu()
            .to(torch.float64)
        )
        accumulator.valid_pair_count += int(valid_pairs.sum().item())

    handles = []
    for name, module in attention_modules.items():
        handles.append(
            module.register_forward_pre_hook(
                lambda module, args, kwargs, name=name: capture_scores(
                    name, module, args, kwargs
                ),
                with_kwargs=True,
            )
        )

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    num_batches = 0
    num_examples = 0
    valid_token_count = 0
    emotional_token_count = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                current_attention_mask = batch["attention_mask"].to(device)
                emotion_features = batch["encoder_emotion_features"].to(device)
                input_ids = batch["input_ids"].to(device)

                # emotional_tokens shape: [batch_size, seq_len]
                valid_tokens = current_attention_mask.bool()
                emotional_tokens = emotion_features.abs().sum(dim=-1).gt(0)
                valid_token_count += int(valid_tokens.sum().item())
                emotional_token_count += int((emotional_tokens & valid_tokens).sum().item())
                num_batches += 1
                num_examples += int(input_ids.size(0))

                encoder(
                    input_ids=input_ids,
                    attention_mask=current_attention_mask,
                    encoder_emotion_features=emotion_features,
                )
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    if num_batches == 0:
        raise ValueError("The r_h dataloader produced no batches.")

    coverage = emotional_token_count / valid_token_count if valid_token_count else 0.0
    rows: list[dict[str, Any]] = []
    total_attention = 0.0
    total_emotion = 0.0
    total_head_pairs = 0
    for name, module in attention_modules.items():
        accumulator = accumulators[name]
        pair_count = accumulator.valid_pair_count
        if pair_count == 0:
            continue

        for head in range(module.num_heads):
            attention_sum = float(accumulator.attention_abs_sum[head].item())
            emotion_sum = float(accumulator.emotion_abs_sum[head].item())
            mean_attention = attention_sum / pair_count
            mean_emotion = emotion_sum / pair_count
            ratio = mean_emotion / mean_attention if mean_attention else float("nan")
            rows.append(
                {
                    "side": "encoder",
                    "layer": _layer_index(name),
                    "head": head,
                    "alpha": float(
                        module.emotion_interaction.alpha[head].detach().cpu().item()
                    ),
                    "mean_abs_attention_score": mean_attention,
                    "mean_abs_scaled_emotion_score": mean_emotion,
                    "r_h": ratio,
                    "emotion_token_coverage": coverage,
                    "num_batches": num_batches,
                    "num_examples": num_examples,
                }
            )
            total_attention += attention_sum
            total_emotion += emotion_sum
            total_head_pairs += pair_count

    if total_head_pairs == 0:
        raise ValueError("No valid token pairs were observed during r_h analysis.")

    overall_mean_attention = total_attention / total_head_pairs
    overall_mean_emotion = total_emotion / total_head_pairs
    overall_ratio = (
        overall_mean_emotion / overall_mean_attention
        if overall_mean_attention
        else float("nan")
    )
    summary = {
        "overall_r_h": overall_ratio,
        "overall_mean_abs_attention_score": overall_mean_attention,
        "overall_mean_abs_scaled_emotion_score": overall_mean_emotion,
        "emotion_token_coverage": coverage,
        "num_batches": float(num_batches),
        "num_examples": float(num_examples),
    }
    for row in rows:
        row.update(
            overall_r_h=overall_ratio,
            overall_mean_abs_attention_score=overall_mean_attention,
            overall_mean_abs_scaled_emotion_score=overall_mean_emotion,
        )
    return rows, summary


def write_eat_signal_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write per-layer, per-head EAT signal diagnostics to CSV."""
    if not rows:
        raise ValueError("Cannot write an empty EAT signal report.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _get_encoder(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model.module if hasattr(model, "module") else model
    bart_model = getattr(unwrapped, "model", unwrapped)
    encoder = getattr(bart_model, "encoder", None)
    if encoder is None:
        raise ValueError("Could not find a BART encoder for r_h analysis.")
    return encoder


def _layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    if "layers" not in parts:
        raise ValueError(f"Could not determine encoder layer from module name: {module_name}")
    return int(parts[parts.index("layers") + 1])
