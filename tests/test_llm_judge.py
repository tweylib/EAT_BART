from __future__ import annotations

import pytest

from eat_bart.training.llm_judge import (
    _extract_gemini_text,
    _parse_judge_response,
    _rate_limit_wait_seconds,
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


def test_parse_judge_response_ignores_qwen_thinking_block() -> None:
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


def test_extract_gemini_text_reads_interactions_output_text() -> None:
    response = {
        "output_text": '{"empathy": 5, "coherence": 4, "safety": 5, "rationale": "Clear."}'
    }

    assert _extract_gemini_text(response).startswith('{"empathy"')


def test_rate_limit_wait_seconds_reads_retry_message() -> None:
    class Error:
        headers = {}

    details = "Please retry in 57.213431047s."

    wait_seconds = _rate_limit_wait_seconds(
        error=Error(),
        details=details,
        fallback_seconds=65.0,
    )

    assert wait_seconds == pytest.approx(58.213431047)


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
