import torch

from eat_bart.data.collator import EATBartDataCollator, shift_emotion_features_right


def test_shift_emotion_features_right_aligns_with_decoder_inputs() -> None:
    target_emotion_features = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    decoder_input_ids = torch.tensor([[2, 10, 1]])

    shifted = shift_emotion_features_right(
        target_emotion_features=target_emotion_features,
        decoder_input_ids=decoder_input_ids,
        pad_token_id=1,
    )

    assert torch.equal(shifted[0, 0], torch.zeros(8))
    assert torch.equal(shifted[0, 1], target_emotion_features[0, 0])
    assert torch.equal(shifted[0, 2], torch.zeros(8))


def test_contextual_decoder_cache_is_aligned_to_shifted_decoder_inputs() -> None:
    class Tokenizer:
        pad_token_id = 1
        eos_token_id = 2

        def __call__(self, texts, **_):
            rows = []
            for text in texts:
                rows.append([0, 3, 2] if text == "short" else [0, 4, 5, 2])
            seq_len = max(len(row) for row in rows)
            input_ids = torch.full((len(rows), seq_len), 1)
            attention_mask = torch.zeros_like(input_ids)
            for index, row in enumerate(rows):
                input_ids[index, : len(row)] = torch.tensor(row)
                attention_mask[index, : len(row)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    cache = {
        "short": torch.full((3, 4), 3.0),
        "long": torch.full((4, 4), 4.0),
    }
    collator = EATBartDataCollator(
        tokenizer=Tokenizer(),
        lexicon={},
        emotion_feature_source="goemotions_contextual",
        contextual_emotion_cache=cache,
        modify_encoder_self_attention=False,
        modify_decoder_self_attention=True,
    )

    batch = collator(
        [
            {"question": "short", "response": "short"},
            {"question": "long", "response": "long"},
        ]
    )

    assert "aligned_emotion_hidden_states" not in batch
    assert batch["decoder_emotion_features"].shape == (2, 4, 4)
    assert torch.equal(batch["decoder_emotion_features"][0, :3], cache["short"])
    assert torch.count_nonzero(batch["decoder_emotion_features"][0, 3]) == 0
