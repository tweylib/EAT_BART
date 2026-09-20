from types import SimpleNamespace

import pytest
import torch
from transformers import TrainerControl, TrainerState

from eat_bart.modeling.eat_attention import EATAttentionConfig, EmotionInteraction
from eat_bart.training.alpha_schedule import (
    LinearAlphaWarmupCallback,
    linear_warmup_alpha,
    set_probability_alpha,
)


class _ProbabilityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.interactions = torch.nn.ModuleList(
            [
                EmotionInteraction(
                    EATAttentionConfig(
                        num_heads=2,
                        emotion_dim=8,
                        emotion_hidden_dim=4,
                        alpha_init=0.05,
                        formula="probability_mix",
                    )
                )
                for _ in range(2)
            ]
        )


def test_linear_alpha_warmup_starts_nonzero_and_clamps_at_target() -> None:
    assert linear_warmup_alpha(0, 50, 0.005, 0.05) == pytest.approx(0.005)
    assert linear_warmup_alpha(10, 50, 0.005, 0.05) == pytest.approx(0.014)
    assert linear_warmup_alpha(50, 50, 0.005, 0.05) == pytest.approx(0.05)
    assert linear_warmup_alpha(500, 50, 0.005, 0.05) == pytest.approx(0.05)


def test_alpha_warmup_callback_uses_epoch_fraction_and_updates_all_layers() -> None:
    model = _ProbabilityModel()
    callback = LinearAlphaWarmupCallback(0.005, 0.05, warmup_epochs=5)
    args = SimpleNamespace(num_train_epochs=40)
    state = TrainerState(max_steps=400, global_step=0)
    control = TrainerControl()

    callback.on_train_begin(args, state, control, model=model)
    assert callback.warmup_steps == 50
    assert all(
        interaction.alpha.item() == pytest.approx(0.005)
        for interaction in model.interactions
    )

    state.global_step = 10
    callback.on_step_end(args, state, control, model=model)
    assert all(
        interaction.alpha.item() == pytest.approx(0.014)
        for interaction in model.interactions
    )

    state.global_step = 50
    callback.on_step_end(args, state, control, model=model)
    assert all(
        interaction.alpha.item() == pytest.approx(0.05)
        for interaction in model.interactions
    )

    logs = {}
    state.log_history.append({"loss": 1.0, "step": 50})
    callback.on_log(args, state, control, logs=logs)
    assert logs["eat_alpha"] == pytest.approx(0.05)
    assert state.log_history[-1]["eat_alpha"] == pytest.approx(0.05)


def test_set_probability_alpha_reports_updated_module_count() -> None:
    model = _ProbabilityModel()
    assert set_probability_alpha(model, 0.0125) == 2
    assert all(
        interaction.alpha.item() == pytest.approx(0.0125)
        for interaction in model.interactions
    )


def test_alpha_warmup_rejects_zero_start() -> None:
    with pytest.raises(ValueError, match="0 < start_alpha"):
        LinearAlphaWarmupCallback(0.0, 0.05, warmup_epochs=5)
