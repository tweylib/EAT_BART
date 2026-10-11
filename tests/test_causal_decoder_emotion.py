from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

from eat_bart.data.causal_decoder_emotion import (
    CausalDecoderEmotionFeatureExtractor,
    build_causal_attention_mask,
    causal_decoder_cache_fingerprint,
    validate_causal_mask_prefix_invariance,
)


class _CharacterTokenizer:
    pad_token_id = 1
    bos_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 1, 2]
    add_prefix_space = False

    def batch_decode(self, rows, **_):
        characters = {3: "a", 4: "b", 5: "c"}
        return [
            "".join(characters[token_id] for token_id in row.tolist() if token_id in characters)
            for row in rows
        ]

    def __call__(self, texts, **_):
        token_ids = {"a": 3, "b": 4, "c": 5}
        seq_len = max(len(text) + 2 for text in texts)
        input_ids = torch.full((len(texts), seq_len), self.pad_token_id)
        attention_mask = torch.zeros_like(input_ids)
        offsets = torch.zeros(len(texts), seq_len, 2, dtype=torch.long)
        for row, text in enumerate(texts):
            values = [self.bos_token_id] + [token_ids[value] for value in text] + [
                self.eos_token_id
            ]
            input_ids[row, : len(values)] = torch.tensor(values)
            attention_mask[row, : len(values)] = 1
            for index in range(len(text)):
                offsets[row, index + 1] = torch.tensor([index, index + 1])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
        }


class _RecordingEmotionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.last_attention_mask = None

    def forward(self, input_ids, attention_mask, **_):
        self.last_attention_mask = attention_mask.detach().cpu()
        hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(hidden_states=[hidden])


def test_causal_mask_is_four_dimensional_and_excludes_future_keys() -> None:
    mask = build_causal_attention_mask(torch.tensor([[1, 1, 1, 0]]))

    assert mask.shape == (1, 1, 4, 4)
    assert torch.equal(
        mask[0, 0],
        torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, False],
            ]
        ),
    )


def test_transformers_roberta_causal_mask_is_prefix_invariant() -> None:
    torch.manual_seed(9)
    config = RobertaConfig(
        vocab_size=20,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=32,
    )
    model = RobertaForSequenceClassification(config).eval()
    full_ids = torch.tensor([[0, 3, 4, 2]])
    prefix_ids = full_ids[:, :3]

    with torch.no_grad():
        full = model(
            input_ids=full_ids,
            attention_mask=build_causal_attention_mask(torch.ones_like(full_ids)),
            output_hidden_states=True,
        ).hidden_states[-1]
        prefix = model(
            input_ids=prefix_ids,
            attention_mask=build_causal_attention_mask(torch.ones_like(prefix_ids)),
            output_hidden_states=True,
        ).hidden_states[-1]

    validate_causal_mask_prefix_invariance(full, prefix, prefix_length=3)


def test_extractor_round_trips_decoder_ids_and_zeros_special_positions() -> None:
    tokenizer = _CharacterTokenizer()
    emotion_model = _RecordingEmotionModel()
    extractor = CausalDecoderEmotionFeatureExtractor(
        bart_tokenizer=tokenizer,
        emotion_tokenizer=tokenizer,
        emotion_model=emotion_model,
        max_length=8,
    )

    features = extractor(torch.tensor([[2, 0, 3, 4]]))

    assert features.shape == (1, 4, 4)
    assert torch.count_nonzero(features[0, :2]) == 0
    assert torch.equal(features[0, 2], torch.full((4,), 3.0))
    assert torch.equal(features[0, 3], torch.full((4,), 4.0))
    assert emotion_model.last_attention_mask.shape == (1, 1, 4, 4)


def test_causal_shuffled_features_are_stable_when_prefix_grows() -> None:
    tokenizer = _CharacterTokenizer()
    extractor = CausalDecoderEmotionFeatureExtractor(
        bart_tokenizer=tokenizer,
        emotion_tokenizer=tokenizer,
        emotion_model=_RecordingEmotionModel(),
        max_length=8,
        condition="shuffled",
        shuffle_seed=31,
    )

    prefix = extractor(torch.tensor([[2, 0, 3, 4]]))
    extended = extractor(torch.tensor([[2, 0, 3, 4, 5]]))

    assert torch.equal(prefix, extended[:, : prefix.size(1)])
    assert torch.count_nonzero(prefix[0, 2]) == 0
    assert torch.equal(prefix[0, 3], torch.full((4,), 3.0))


def test_decoder_cache_fingerprint_covers_causal_tokenization_settings() -> None:
    reference = causal_decoder_cache_fingerprint(["response"], "model", 512, False, 2)

    assert reference != causal_decoder_cache_fingerprint(
        ["different"], "model", 512, False, 2
    )
    assert reference != causal_decoder_cache_fingerprint(
        ["response"], "model", 256, False, 2
    )
    assert reference != causal_decoder_cache_fingerprint(
        ["response"], "model", 512, False, 0
    )
