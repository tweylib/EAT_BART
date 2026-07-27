from pathlib import Path


def test_baseline_contract_explicitly_forbids_attention_modification() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "unmodified `BartForConditionalGeneration`" in readme
    assert "No EAT attention patches" in readme
