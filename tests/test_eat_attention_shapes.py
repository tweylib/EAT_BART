import torch
import pytest

from eat_bart.modeling.eat_attention import EATAttentionConfig, EmotionInteraction


def test_emotion_interaction_parameter_shapes() -> None:
    module = EmotionInteraction(EATAttentionConfig(num_heads=12))

    assert tuple(module.w1_s.shape) == (12, 8, 32)
    assert tuple(module.w2_s.shape) == (12, 8, 32)
    assert tuple(module.alpha.shape) == (12,)


def test_emotion_interaction_output_shape() -> None:
    module = EmotionInteraction(EATAttentionConfig(num_heads=2, emotion_hidden_dim=4))
    emotion_features = torch.randn(3, 5, 8)

    scores = module(emotion_features)

    assert tuple(scores.shape) == (3, 2, 5, 5)


def test_zero_emotion_features_produce_zero_scores() -> None:
    module = EmotionInteraction(EATAttentionConfig(num_heads=2, emotion_hidden_dim=4))
    emotion_features = torch.zeros(3, 5, 8)

    scores = module(emotion_features)

    assert torch.equal(scores, torch.zeros(3, 2, 5, 5))


def test_additive_formula_combines_before_masking() -> None:
    module = EmotionInteraction(EATAttentionConfig(num_heads=2, alpha_init=0.5))
    attention_scores = torch.ones(1, 2, 3, 3)
    emotion_scores = torch.full((1, 2, 3, 3), 2.0)

    combined = module.combine_with_attention_scores(attention_scores, emotion_scores)

    assert torch.equal(combined, torch.full((1, 2, 3, 3), 2.0))


def test_multiplicative_ablation_formula() -> None:
    module = EmotionInteraction(
        EATAttentionConfig(num_heads=1, alpha_init=0.5, formula="multiplicative")
    )
    attention_scores = torch.ones(1, 1, 3, 3)
    emotion_scores = torch.full((1, 1, 3, 3), 2.0)

    combined = module.combine_with_attention_scores(attention_scores, emotion_scores)

    expected = torch.tensor([[[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]]])
    assert torch.equal(combined, expected)


def test_tied_orthogonal_initialization_is_per_head_and_parameters_are_distinct() -> None:
    torch.manual_seed(23)
    module = EmotionInteraction(
        EATAttentionConfig(
            num_heads=3,
            emotion_dim=8,
            emotion_hidden_dim=4,
            formula="probability_mix",
            weight_initialization="tied_orthogonal",
        )
    )

    assert module.w1_s is not module.w2_s
    assert module.w1_s.data_ptr() != module.w2_s.data_ptr()
    assert torch.equal(module.w1_s, module.w2_s)
    identity = torch.eye(4)
    for head in range(3):
        gram = module.w1_s[head].transpose(0, 1) @ module.w1_s[head]
        assert torch.allclose(gram, identity, atol=1e-6, rtol=1e-6)
    assert not torch.equal(module.w1_s[0], module.w1_s[1])

    (module.w1_s.sum() + 2.0 * module.w2_s.sum()).backward()
    assert torch.equal(module.w1_s.grad, torch.ones_like(module.w1_s))
    assert torch.equal(module.w2_s.grad, torch.full_like(module.w2_s, 2.0))


def test_tied_orthogonal_requires_enough_input_dimensions() -> None:
    with pytest.raises(ValueError, match="emotion_dim >= emotion_hidden_dim"):
        EmotionInteraction(
            EATAttentionConfig(
                num_heads=2,
                emotion_dim=4,
                emotion_hidden_dim=8,
                weight_initialization="tied_orthogonal",
            )
        )


def test_default_initialization_remains_independent_xavier() -> None:
    module = EmotionInteraction(
        EATAttentionConfig(num_heads=2, emotion_dim=8, emotion_hidden_dim=4)
    )
    assert module.config.weight_initialization == "independent_xavier"
    assert not torch.equal(module.w1_s, module.w2_s)
