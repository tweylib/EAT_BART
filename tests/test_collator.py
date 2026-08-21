import torch

from eat_bart.data.collator import BartDataCollator


class StubTokenizer:
    pad_token_id = 1
    eos_token_id = 2

    def __call__(self, texts, **kwargs):
        del kwargs
        rows = [[0, 10 + index, 2] for index, _ in enumerate(texts)]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.ones(len(rows), 3, dtype=torch.long),
        }


def test_standard_bart_collator_returns_only_native_model_inputs() -> None:
    collator = BartDataCollator(tokenizer=StubTokenizer(), decoder_start_token_id=2)

    batch = collator(
        [
            {"question": "question one", "response": "response one"},
            {"question": "question two", "response": "response two"},
        ]
    )

    assert set(batch) == {
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
    }
    assert all("emotion" not in key for key in batch)
    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert tuple(batch["labels"].shape) == (2, 3)
