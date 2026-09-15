"""Optimizer helpers for differential EAT-BART learning rates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from transformers import Seq2SeqTrainer

ALPHA_PARAMETER_SUFFIX = ".emotion_interaction.alpha"
EAT_PARAMETER_MARKER = ".emotion_interaction."


class DifferentialLearningRateTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with separate rates for BART, EAT projections, and alpha."""

    def __init__(
        self,
        *args: Any,
        eat_learning_rate: float,
        alpha_learning_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.eat_learning_rate = eat_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
        """Create AdamW parameter groups while preserving Trainer scheduler behavior."""
        optimizer_model = self.model if model is None else model
        if self.optimizer is None:
            decay_parameter_names = self.get_decay_parameter_names(optimizer_model)
            parameter_groups = build_differential_parameter_groups(
                model=optimizer_model,
                bart_learning_rate=float(self.args.learning_rate),
                eat_learning_rate=self.eat_learning_rate,
                alpha_learning_rate=self.alpha_learning_rate,
                weight_decay=float(self.args.weight_decay),
                decay_parameter_names=decay_parameter_names,
            )
            self.optimizer = torch.optim.AdamW(
                parameter_groups,
                betas=(float(self.args.adam_beta1), float(self.args.adam_beta2)),
                eps=float(self.args.adam_epsilon),
            )

        return self.optimizer


def build_differential_parameter_groups(
    model: torch.nn.Module,
    bart_learning_rate: float,
    eat_learning_rate: float,
    alpha_learning_rate: float | None,
    weight_decay: float = 0.0,
    decay_parameter_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Split trainable parameters by ownership and optional weight-decay behavior."""
    _validate_learning_rate("bart_learning_rate", bart_learning_rate)
    _validate_learning_rate("eat_learning_rate", eat_learning_rate)
    if alpha_learning_rate is not None:
        _validate_learning_rate("alpha_learning_rate", alpha_learning_rate)

    decay_names = set(decay_parameter_names or ())
    grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    category_counts = {"bart": 0, "eat": 0, "alpha": 0}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        category = _parameter_category(name)
        use_decay = bool(weight_decay) and name in decay_names
        grouped.setdefault((category, use_decay), []).append(parameter)
        category_counts[category] += parameter.numel()

    required_categories = ("bart", "eat", "alpha") if alpha_learning_rate is not None else ("bart", "eat")
    missing_categories = [name for name in required_categories if category_counts[name] == 0]
    if missing_categories:
        missing = ", ".join(missing_categories)
        raise ValueError(f"Differential optimizer found no trainable parameters for: {missing}")

    learning_rates = {
        "bart": bart_learning_rate,
        "eat": eat_learning_rate,
        "alpha": alpha_learning_rate,
    }
    parameter_groups: list[dict[str, Any]] = []
    for (category, use_decay), parameters in grouped.items():
        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rates[category],
                "weight_decay": weight_decay if use_decay else 0.0,
                "group_name": f"{category}_{'decay' if use_decay else 'no_decay'}",
            }
        )

    return parameter_groups


def _parameter_category(name: str) -> str:
    if name.endswith(ALPHA_PARAMETER_SUFFIX):
        return "alpha"
    if EAT_PARAMETER_MARKER in name:
        return "eat"
    return "bart"


def _validate_learning_rate(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
