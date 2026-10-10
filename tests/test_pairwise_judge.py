from __future__ import annotations

import csv

import pytest

import eat_bart.training.pairwise_judge as pairwise_module
from eat_bart.training.pairwise_judge import (
    _balanced_left_as_a,
    _normalize_preference,
    _two_sided_sign_test,
    build_pairwise_prompt,
    parse_pairwise_response,
    summarize_pairwise_judgments,
)


def test_pairwise_prompt_is_blind_to_model_names() -> None:
    prompt = build_pairwise_prompt("question", "reference", "first", "second")

    assert "Response A:\nfirst" in prompt
    assert "Response B:\nsecond" in prompt
    assert "baseline" not in prompt.lower()
    assert "eat" not in prompt.lower()


def test_parse_pairwise_response_accepts_json_and_thinking_block() -> None:
    result = parse_pairwise_response(
        '<think>private reasoning</think>{"empathy":"A","coherence":"tie",'
        '"safety":"response B","rationale":"short"}'
    )

    assert result == {
        "empathy": "a",
        "coherence": "tie",
        "safety": "b",
        "rationale": "short",
    }


def test_order_is_balanced_and_deterministic() -> None:
    first = _balanced_left_as_a(100, seed=9)
    second = _balanced_left_as_a(100, seed=9)

    assert first == second
    assert sum(first) == 50


def test_preference_is_normalized_after_blind_judging() -> None:
    assert _normalize_preference("a", left_is_a=True) == "left"
    assert _normalize_preference("a", left_is_a=False) == "right"
    assert _normalize_preference("b", left_is_a=True) == "right"
    assert _normalize_preference("tie", left_is_a=False) == "tie"


def test_pairwise_summary_counts_preferences_and_sign_test() -> None:
    rows = [
        {
            "judge_mode": "api",
            "empathy_preference": "right",
            "coherence_preference": "tie",
            "safety_preference": "left",
            "llm_error": "",
        },
        {
            "judge_mode": "automatic_identical_tie",
            "empathy_preference": "tie",
            "coherence_preference": "tie",
            "safety_preference": "tie",
            "llm_error": "",
        },
    ]

    summary = summarize_pairwise_judgments(rows, left_name="baseline", right_name="eat")

    assert summary["num_identical_responses"] == pytest.approx(1.0)
    assert summary["num_api_requests"] == pytest.approx(1.0)
    assert summary["empathy_right_wins"] == pytest.approx(1.0)
    assert summary["coherence_ties"] == pytest.approx(2.0)
    assert summary["safety_left_wins"] == pytest.approx(1.0)
    assert _two_sided_sign_test(0, 0) == pytest.approx(1.0)
    assert _two_sided_sign_test(0, 5) == pytest.approx(0.0625)


def test_pairwise_csv_auto_ties_identical_rows_and_blinds_changed_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left_path = tmp_path / "left.csv"
    right_path = tmp_path / "right.csv"
    fieldnames = ["question", "reference_response", "generated_response"]
    left_rows = [
        {"question": "q1", "reference_response": "r1", "generated_response": "same"},
        {"question": "q2", "reference_response": "r2", "generated_response": "left"},
    ]
    right_rows = [
        {"question": "q1", "reference_response": "r1", "generated_response": "same"},
        {"question": "q2", "reference_response": "r2", "generated_response": "right"},
    ]
    for path, rows in ((left_path, left_rows), (right_path, right_rows)):
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    prompts: list[str] = []
    monkeypatch.setattr(pairwise_module, "_load_api_key", lambda **_: "secret")
    monkeypatch.setattr(pairwise_module, "_balanced_left_as_a", lambda *_args, **_kwargs: [True, False])

    def fake_call(**kwargs: object) -> str:
        prompts.append(str(kwargs["prompt"]))
        return '{"empathy":"A","coherence":"A","safety":"A","rationale":"better"}'

    monkeypatch.setattr(pairwise_module, "_call_groq", fake_call)
    output_path = tmp_path / "judged.csv"
    summary = pairwise_module.judge_pairwise_generation_csv(
        left_path,
        right_path,
        output_path,
        left_name="baseline",
        right_name="eat",
    )

    assert len(prompts) == 1
    assert "baseline" not in prompts[0].lower()
    assert summary["num_identical_responses"] == pytest.approx(1.0)
    assert summary["num_api_requests"] == pytest.approx(1.0)
    assert summary["empathy_right_wins"] == pytest.approx(1.0)
    with output_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["empathy_preference"] == "tie"
    assert rows[1]["empathy_preference"] == "right"
