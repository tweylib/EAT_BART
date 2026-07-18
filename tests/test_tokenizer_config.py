from eat_bart.data.tokenizer import load_bart_tokenizer


def test_bart_tokenizer_preserves_spaces_with_pretokenized_input() -> None:
    tokenizer = load_bart_tokenizer("facebook/bart-base", local_files_only=True)
    encoded = tokenizer(
        [["hello", "world"]],
        is_split_into_words=True,
        return_tensors="pt",
    )

    decoded = tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True)[0]

    assert decoded.strip() == "hello world"
    assert "helloworld" not in decoded
