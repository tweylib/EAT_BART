from __future__ import annotations

import pytest
import torch

from eat_bart.training.emotion_ablation import apply_emotion_feature_condition


def _features() -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0], [0.0]],
            [[0.0], [4.0], [5.0], [6.0], [0.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    return features, mask


def test_aligned_condition_preserves_features() -> None:
    features, mask = _features()

    result = apply_emotion_feature_condition(features, mask, "aligned")

    assert torch.equal(result, features)


def test_zero_condition_removes_all_feature_content() -> None:
    features, mask = _features()

    result = apply_emotion_feature_condition(features, mask, "zero")

    assert torch.equal(result, torch.zeros_like(features))


def test_shuffled_condition_is_deterministic_and_preserves_zero_rows() -> None:
    features, mask = _features()

    first = apply_emotion_feature_condition(features, mask, "shuffled", seed=7)
    second = apply_emotion_feature_condition(features, mask, "shuffled", seed=7)

    assert torch.equal(first, second)
    assert first[:, 0].eq(0).all()
    assert first[:, 4].eq(0).all()
    assert sorted(first[0, 1:4, 0].tolist()) == [1.0, 2.0, 3.0]
    assert sorted(first[1, 1:4, 0].tolist()) == [4.0, 5.0, 6.0]


def test_shuffling_is_independent_of_batch_boundaries() -> None:
    features, mask = _features()

    together = apply_emotion_feature_condition(features, mask, "shuffled", seed=7)
    separately = torch.cat(
        [
            apply_emotion_feature_condition(
                features[:1], mask[:1], "shuffled", seed=7, example_offset=0
            ),
            apply_emotion_feature_condition(
                features[1:], mask[1:], "shuffled", seed=7, example_offset=1
            ),
        ]
    )

    assert torch.equal(together, separately)


def test_invalid_condition_is_rejected() -> None:
    features, mask = _features()

    with pytest.raises(ValueError, match="emotion_feature_condition"):
        apply_emotion_feature_condition(features, mask, "unknown")
