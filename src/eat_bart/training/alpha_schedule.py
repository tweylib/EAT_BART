"""Scheduled probability-mixture coefficients for EAT training."""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from eat_bart.modeling.eat_attention import PROBABILITY_FORMULAS, EmotionInteraction


def linear_warmup_alpha(
    step: int,
    warmup_steps: int,
    start_alpha: float,
    target_alpha: float,
) -> float:
    """Return alpha for a nonzero linear warmup, clamped at the target."""
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be at least 1.")
    progress = min(max(step, 0) / warmup_steps, 1.0)
    return start_alpha + (target_alpha - start_alpha) * progress


def set_probability_alpha(model: torch.nn.Module, value: float) -> int:
    """Set every fixed probability-formula alpha buffer and return its count."""
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, EmotionInteraction):
                continue
            if module.config.formula not in PROBABILITY_FORMULAS:
                continue
            if isinstance(module.alpha, torch.nn.Parameter):
                raise ValueError("Scheduled alpha must be a fixed buffer, not a Parameter.")
            module.alpha.fill_(value)
            count += 1
    return count


class LinearAlphaWarmupCallback(TrainerCallback):
    """Warm fixed EAT alpha from a nonzero start to its configured target."""

    def __init__(
        self,
        start_alpha: float,
        target_alpha: float,
        warmup_epochs: float,
    ) -> None:
        if not 0.0 < start_alpha <= target_alpha <= 1.0:
            raise ValueError(
                "Alpha warmup requires 0 < start_alpha <= target_alpha <= 1."
            )
        if warmup_epochs <= 0.0:
            raise ValueError("alpha warmup_epochs must be positive.")
        self.start_alpha = float(start_alpha)
        self.target_alpha = float(target_alpha)
        self.warmup_epochs = float(warmup_epochs)
        self.warmup_steps: int | None = None
        self.current_alpha = self.start_alpha

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **_: Any,
    ) -> TrainerControl:
        if self.warmup_epochs > float(args.num_train_epochs):
            raise ValueError(
                "alpha warmup_epochs cannot exceed training num_train_epochs."
            )
        if state.max_steps < 1:
            raise ValueError("Trainer state must provide a positive max_steps.")
        self.warmup_steps = max(
            1,
            math.ceil(
                state.max_steps * self.warmup_epochs / float(args.num_train_epochs)
            ),
        )
        self._apply(model, state.global_step)
        return control

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **_: Any,
    ) -> TrainerControl:
        self._apply(model, state.global_step)
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **_: Any,
    ) -> TrainerControl:
        # Update after the optimizer step so epoch-end evaluation/checkpointing
        # observes alpha at the completed global step.
        self._apply(model, state.global_step)
        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **_: Any,
    ) -> TrainerControl:
        if logs is not None:
            logs["eat_alpha"] = self.current_alpha
        # Trainer appends the log-history row immediately before dispatching
        # on_log, so update that row as well for checkpoint auditability.
        if state.log_history:
            state.log_history[-1]["eat_alpha"] = self.current_alpha
        return control

    def _apply(self, model: torch.nn.Module | None, step: int) -> None:
        if model is None:
            raise ValueError("Alpha warmup callback received no model.")
        if self.warmup_steps is None:
            raise RuntimeError("Alpha warmup was used before on_train_begin.")
        self.current_alpha = linear_warmup_alpha(
            step=step,
            warmup_steps=self.warmup_steps,
            start_alpha=self.start_alpha,
            target_alpha=self.target_alpha,
        )
        count = set_probability_alpha(model, self.current_alpha)
        if count == 0:
            raise ValueError("Alpha warmup found no probability-based EAT modules.")
