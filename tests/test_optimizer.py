import pytest
from transformers import BartConfig

from eat_bart.modeling.eat_bart_model import build_eat_bart_model_from_config
from eat_bart.training.optimizer import build_differential_parameter_groups


def test_differential_parameter_groups_assign_requested_learning_rates() -> None:
    model = build_eat_bart_model_from_config(
        BartConfig(
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            vocab_size=99,
        ),
        modify_decoder_self_attention=False,
    )

    groups = build_differential_parameter_groups(
        model=model,
        bart_learning_rate=1e-5,
        eat_learning_rate=5e-5,
        alpha_learning_rate=0.01,
    )
    learning_rate_by_parameter = {
        id(parameter): group["lr"]
        for group in groups
        for parameter in group["params"]
    }

    for name, parameter in model.named_parameters():
        if name.endswith("emotion_interaction.alpha"):
            assert learning_rate_by_parameter[id(parameter)] == pytest.approx(0.01)
        elif ".emotion_interaction." in name:
            assert learning_rate_by_parameter[id(parameter)] == pytest.approx(5e-5)
        else:
            assert learning_rate_by_parameter[id(parameter)] == pytest.approx(1e-5)


def test_differential_parameter_groups_reject_missing_eat_parameters() -> None:
    model = build_eat_bart_model_from_config(
        BartConfig(
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            vocab_size=99,
        ),
        modify_encoder_self_attention=False,
        modify_decoder_self_attention=False,
    )

    with pytest.raises(ValueError, match="eat, alpha"):
        build_differential_parameter_groups(
            model=model,
            bart_learning_rate=1e-5,
            eat_learning_rate=5e-5,
            alpha_learning_rate=0.01,
        )
