from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from eat_bart.training.llm_judge import (
    _call_groq,
    _parse_judge_response,
    _summarize_judgments,
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


def test_call_groq_passes_qwen_reasoning_control(monkeypatch: pytest.MonkeyPatch) -> None:
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
        model="qwen/qwen3.6-27b",
        api_key="test-key",
        prompt="Return JSON.",
        temperature=0.0,
        timeout_seconds=60,
        max_retries=0,
        rate_limit_sleep_seconds=0.0,
        max_output_tokens=512,
        response_format_json=True,
        reasoning_effort="none",
    )

    assert result.startswith("{")
    assert request["reasoning_effort"] == "none"
    assert request["response_format"] == {"type": "json_object"}
