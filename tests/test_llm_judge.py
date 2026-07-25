from __future__ import annotations

import pytest

from eat_bart.training.llm_judge import _extract_gemini_text, _parse_judge_response


def test_parse_judge_response_accepts_json() -> None:
    result = _parse_judge_response(
        '{"empathy": 4, "coherence": 5, "safety": 5, "rationale": "Supportive."}'
    )

    assert result["empathy"] == 4
    assert result["coherence"] == 5
    assert result["safety"] == 5
    assert result["rationale"] == "Supportive."


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
