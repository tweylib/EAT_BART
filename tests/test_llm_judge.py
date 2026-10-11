from __future__ import annotations

import csv
import sys
from types import SimpleNamespace

import pytest

import eat_bart.training.llm_judge as judge_module
from eat_bart.training.llm_judge import (
    _call_groq,
    _judge_json_schema_response_format,
    _parse_judge_response,
    _summarize_judgments,
    judge_generation_csv,
)


def test_parse_judge_response_accepts_json() -> None:
    result = _parse_judge_response(
        '{"empathy": 4, "coherence": 5, "safety": 5, "rationale": "Supportive."}'
    )

    assert result["empathy"] == 4
    assert result["coherence"] == 5
    assert result["safety"] == 5
    assert result["rationale"] == "Supportive."


def test_parse_judge_response_extracts_json_from_text() -> None:
    result = _parse_judge_response(
        'Here is the score: {"empathy": 3, "coherence": 4, "safety": 5, "rationale": "Ok."}'
    )

    assert result["empathy"] == 3
    assert result["coherence"] == 4
    assert result["safety"] == 5


def test_parse_judge_response_ignores_thinking_block() -> None:
    result = _parse_judge_response(
        '<think>{"not": "the answer"}</think>\n'
        '{"empathy": 4, "coherence": 4, "safety": 5, "rationale": "Clear."}'
    )

    assert result["empathy"] == 4
    assert result["coherence"] == 4
    assert result["safety"] == 5


def test_parse_judge_response_rejects_truncated_json() -> None:
    with pytest.raises(ValueError, match="Could not find JSON object"):
        _parse_judge_response(
            '{"empathy": 3, "coherence": 2, "safety": 3, "rationale": "unfinished'
        )


def test_parse_judge_response_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="between 1 and 5"):
        _parse_judge_response(
            '{"empathy": 6, "coherence": 5, "safety": 5, "rationale": "Too high."}'
        )


def test_summarize_judgments_ignores_failed_rows() -> None:
    summary = _summarize_judgments(
        [
            {
                "llm_empathy": "5",
                "llm_coherence": "4",
                "llm_safety": "5",
                "llm_error": "",
            },
            {
                "llm_empathy": "",
                "llm_coherence": "",
                "llm_safety": "",
                "llm_error": "quota exceeded",
            },
        ]
    )

    assert summary["num_requested_examples"] == pytest.approx(2.0)
    assert summary["num_judged_examples"] == pytest.approx(1.0)
    assert summary["num_failed_examples"] == pytest.approx(1.0)
    assert summary["llm_empathy"] == pytest.approx(5.0)


def test_call_groq_hides_qwen_reasoning_and_requires_strict_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            request.update(kwargs)
            message = SimpleNamespace(
                content='{"empathy": 4, "coherence": 4, "safety": 5}'
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeGroq:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))

    result = _call_groq(
        model="qwen/qwen3.8-27b",
        api_key="test-key",
        prompt="Return JSON.",
        temperature=0.0,
        timeout_seconds=60,
        max_retries=0,
        rate_limit_sleep_seconds=0.0,
        max_output_tokens=512,
        response_format_json=False,
        response_format_json_schema=True,
        reasoning_effort="none",
        reasoning_format="hidden",
    )

    assert result.startswith("{")
    assert "<think>" not in result
    assert request["reasoning_effort"] == "none"
    assert request["reasoning_format"] == "hidden"
    assert request["response_format"] == _judge_json_schema_response_format()
    assert request["response_format"]["json_schema"]["strict"] is True


def test_call_groq_rejects_visible_thinking_when_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletions:
        def create(self, **_: object) -> SimpleNamespace:
            message = SimpleNamespace(content='<think>secret</think>{"empathy": 4}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeGroq:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))

    with pytest.raises(RuntimeError, match="visible thinking"):
        _call_groq(
            model="qwen/qwen3.8-27b",
            api_key="test-key",
            prompt="Return JSON.",
            temperature=0.0,
            timeout_seconds=60,
            max_retries=0,
            rate_limit_sleep_seconds=0.0,
            max_output_tokens=512,
            response_format_json=False,
            response_format_json_schema=True,
            reasoning_effort="none",
            reasoning_format="hidden",
        )


def test_qwen_judge_pipeline_writes_only_parsed_scores(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "generations.csv"
    output_path = tmp_path / "judgments.csv"
    with input_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["question", "reference_response", "generated_response"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "question": "I feel alone.",
                "reference_response": "You deserve support.",
                "generated_response": "That sounds painful, and you do not have to face it alone.",
            }
        )

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        judge_module,
        "_call_groq",
        lambda **_: (
            '{"empathy":4,"coherence":5,"safety":5,'
            '"rationale":"Warm, relevant, and safe."}'
        ),
    )

    summary = judge_generation_csv(
        input_path=input_path,
        output_path=output_path,
        model="qwen/qwen3.8-27b",
        max_examples=1,
        response_format_json_schema=True,
        reasoning_effort="none",
        reasoning_format="hidden",
    )

    assert summary["num_judged_examples"] == pytest.approx(1.0)
    assert summary["llm_empathy"] == pytest.approx(4.0)
    with output_path.open("r", encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["llm_empathy"] == "4"
    assert row["llm_coherence"] == "5"
    assert row["llm_safety"] == "5"
    assert "<think>" not in row["llm_rationale"]
