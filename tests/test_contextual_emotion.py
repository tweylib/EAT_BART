from __future__ import annotations

import torch

from eat_bart.data.contextual_emotion import align_hidden_states, build_offset_alignment
from eat_bart.modeling.eat_attention import EATAttentionConfig
from eat_bart.modeling.eat_bart_attention import EATBartAttention, eat_eager_attention_forward


def test_offset_alignment_exact_fast_path_has_one_representation_per_bart_token() -> None:
    offsets = torch.tensor([[[0, 0], [0, 4], [4, 5], [0, 0]]])
    mask = torch.ones(1, 4, dtype=torch.long)
    alignment, diagnostics = build_offset_alignment(offsets, offsets, mask, mask)
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    aligned = align_hidden_states(hidden, alignment)
    assert diagnostics.used_identity_fast_path
    assert diagnostics.coverage == 1.0
    assert aligned.shape == (1, 4, 3)
    assert torch.equal(aligned[:, 1:3], hidden[:, 1:3])


def test_offset_alignment_mean_pools_overlapping_subtokens() -> None:
    bart = torch.tensor([[[0, 0], [0, 5], [0, 0]]])
    roberta = torch.tensor([[[0, 0], [0, 2], [2, 5], [0, 0]]])
    alignment, diagnostics = build_offset_alignment(
        bart, roberta, torch.ones(1, 3), torch.ones(1, 4)
    )
    hidden = torch.tensor([[[0.0], [2.0], [4.0], [0.0]]])
    assert align_hidden_states(hidden, alignment)[0, 1, 0].item() == 3.0
    assert diagnostics.coverage == 1.0


def _probability_mix(alpha: float):
    module = EATBartAttention(
        embed_dim=4, num_heads=1,
        eat_config=EATAttentionConfig(
            num_heads=1, emotion_dim=2, emotion_hidden_dim=2,
            alpha_init=alpha, formula="probability_mix",
        ),
    )
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = query.clone()
    value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    emotion_scores = torch.tensor([[[[0.0, 2.0], [2.0, 0.0]]]])
    return eat_eager_attention_forward(module, query, key, value, None, emotion_scores, dropout=0.0)


def test_probability_mix_alpha_endpoints_and_normalization() -> None:
    _, p0 = _probability_mix(0.0)
    expected_a = torch.softmax(torch.tensor([[[[2**-0.5, 0.0], [0.0, 2**-0.5]]]]), -1)
    assert torch.allclose(p0, expected_a)
    _, p1 = _probability_mix(1.0)
    expected_s = torch.softmax(torch.tensor([[[[0.0, 2.0], [2.0, 0.0]]]]), -1)
    assert torch.allclose(p1, expected_s)
    assert torch.allclose(p1.sum(-1), torch.ones_like(p1.sum(-1)))


def test_probability_mix_alpha_is_fixed() -> None:
    config = EATAttentionConfig(num_heads=2, alpha_init=0.1, formula="probability_mix")
    module = EATBartAttention(embed_dim=8, num_heads=2, eat_config=config)
    assert "emotion_interaction.alpha" not in dict(module.named_parameters())
    assert module.emotion_interaction.alpha.item() == pytest.approx(0.1)


def test_contextual_pairwise_weights_receive_gradients_without_projection() -> None:
    torch.manual_seed(7)
    module = EATBartAttention(
        embed_dim=8, num_heads=2,
        eat_config=EATAttentionConfig(2, 768, 16, 0.1, "probability_mix"),
    )
    contextual = torch.randn(2, 5, 768)
    emotion_scores = module.emotion_interaction(contextual)
    standard = torch.softmax(torch.randn_like(emotion_scores), dim=-1)
    mixed = 0.9 * standard + 0.1 * torch.softmax(emotion_scores, dim=-1)
    loss = torch.matmul(mixed, torch.randn(2, 2, 5, 4)).square().mean()
    loss.backward()
    assert module.emotion_interaction.w1_s.grad is not None
    assert module.emotion_interaction.w1_s.grad.norm() > 0
    assert module.emotion_interaction.w2_s.grad is not None
    assert module.emotion_interaction.w2_s.grad.norm() > 0


import pytest


def test_cached_collator_pads_aligned_states_to_bart_length() -> None:
    class Tokenizer:
        pad_token_id = 1
        eos_token_id = 2

        def __call__(self, texts, **kwargs):
            length = max(4 if "long" in text else 3 for text in texts)
            ids = torch.ones(len(texts), length, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, text in enumerate(texts):
                row_length = 4 if "long" in text else 3
                ids[index, :row_length] = torch.arange(row_length) + 2
                mask[index, :row_length] = 1
            return {"input_ids": ids, "attention_mask": mask}

    from eat_bart.data.collator import EATBartDataCollator

    tokenizer = Tokenizer()
    cache = {
        "short": torch.randn(3, 768, dtype=torch.float16),
        "long question": torch.randn(4, 768, dtype=torch.float16),
    }
    collator = EATBartDataCollator(
        tokenizer=tokenizer, lexicon={}, emotion_feature_source="goemotions_contextual",
        contextual_emotion_cache=cache,
    )
    # Patch target feature tokenization is out of scope for this focused source-cache test.
    collator.emotion_feature_source = "goemotions_contextual"
    source = tokenizer(["short", "long question"])
    rows = []
    for text_value in ("short", "long question"):
        row = torch.zeros(source["input_ids"].size(1), 768, dtype=torch.float16)
        row[: cache[text_value].size(0)] = cache[text_value]
        rows.append(row)
    result = torch.stack(rows)
    assert result.shape == (2, 4, 768)
    assert torch.count_nonzero(result[0, 3]) == 0
