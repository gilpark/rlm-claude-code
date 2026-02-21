"""Tests for structured LLM response parsing."""
import pytest
from src.repl.json_parser import parse_llm_response, StructuredResponse


def test_parse_valid_json():
    """Should parse valid JSON response."""
    raw = '{"conclusion": "Test answer", "confidence": 0.9}'
    parsed = parse_llm_response(raw)

    assert parsed.is_valid
    assert parsed.conclusion == "Test answer"
    assert parsed.confidence == 0.9
    assert parsed.error is None


def test_parse_with_all_fields():
    """Should parse all optional fields."""
    raw = '''{
        "reasoning": "Step by step",
        "conclusion": "Final answer",
        "confidence": 0.85,
        "files": ["file1.py", "file2.py"],
        "sub_tasks": [{"task": "sub1"}, {"task": "sub2"}],
        "needs_more_info": true,
        "next_action": "continue"
    }'''
    parsed = parse_llm_response(raw)

    assert parsed.is_valid
    assert parsed.reasoning == "Step by step"
    assert parsed.conclusion == "Final answer"
    assert parsed.confidence == 0.85
    assert parsed.files == ["file1.py", "file2.py"]
    assert parsed.sub_tasks == [{"task": "sub1"}, {"task": "sub2"}]
    assert parsed.needs_more_info is True
    assert parsed.next_action == "continue"
    assert parsed.error is None


def test_parse_markdown_wrapped_json():
    """Should extract JSON from ```json blocks."""
    raw = '''```json
{
    "conclusion": "Extracted from markdown",
    "confidence": 0.7
}
```'''
    parsed = parse_llm_response(raw)

    assert parsed.is_valid
    assert parsed.conclusion == "Extracted from markdown"
    assert parsed.confidence == 0.7
    assert parsed.error is None


def test_fallback_on_invalid_json():
    """Should fallback to raw text on invalid JSON."""
    raw = '{"conclusion": "incomplete"'
    parsed = parse_llm_response(raw)

    assert not parsed.is_valid
    assert parsed.conclusion == raw
    assert parsed.confidence == 0.5
    assert parsed.error == "invalid_json"


def test_fallback_on_missing_conclusion():
    """Should handle missing conclusion field."""
    raw = '{"confidence": 0.9}'
    parsed = parse_llm_response(raw)

    assert not parsed.is_valid  # Empty conclusion = not valid
    assert parsed.conclusion == ""
    assert parsed.confidence == 0.9
    assert parsed.error is None


def test_confidence_clamping_high():
    """Should clamp confidence > 1.0 to 1.0."""
    raw = '{"conclusion": "High confidence", "confidence": 1.5}'
    parsed = parse_llm_response(raw)

    assert parsed.conclusion == "High confidence"
    assert parsed.confidence == 1.0


def test_confidence_clamping_low():
    """Should clamp confidence < 0.0 to 0.0."""
    raw = '{"conclusion": "Low confidence", "confidence": -0.5}'
    parsed = parse_llm_response(raw)

    assert parsed.conclusion == "Low confidence"
    assert parsed.confidence == 0.0


def test_confidence_invalid_type():
    """Should use default confidence for invalid type."""
    raw = '{"conclusion": "Invalid confidence", "confidence": "high"}'
    parsed = parse_llm_response(raw)

    assert parsed.conclusion == "Invalid confidence"
    assert parsed.confidence == 0.8  # Default


def test_confidence_none():
    """Should use default confidence when None."""
    raw = '{"conclusion": "None confidence", "confidence": null}'
    parsed = parse_llm_response(raw)

    assert parsed.conclusion == "None confidence"
    assert parsed.confidence == 0.8  # Default


def test_next_action_validation():
    """Should validate next_action values."""
    # Valid actions
    for action in ["continue", "finalize", "ask_user"]:
        raw = f'{{"conclusion": "Test", "next_action": "{action}"}}'
        parsed = parse_llm_response(raw)
        assert parsed.next_action == action

    # Invalid action should default to "finalize"
    raw = '{"conclusion": "Test", "next_action": "invalid_action"}'
    parsed = parse_llm_response(raw)
    assert parsed.next_action == "finalize"


def test_should_continue_property():
    """Should test should_continue logic."""
    # Should continue when next_action="continue" AND has sub_tasks
    raw1 = '{"conclusion": "Test", "next_action": "continue", "sub_tasks": [{"task": "sub1"}]}'
    parsed1 = parse_llm_response(raw1)
    assert parsed1.should_continue is True

    # Should NOT continue when next_action="continue" but NO sub_tasks
    raw2 = '{"conclusion": "Test", "next_action": "continue", "sub_tasks": []}'
    parsed2 = parse_llm_response(raw2)
    assert parsed2.should_continue is False

    # Should NOT continue when next_action="finalize"
    raw3 = '{"conclusion": "Test", "next_action": "finalize", "sub_tasks": [{"task": "sub1"}]}'
    parsed3 = parse_llm_response(raw3)
    assert parsed3.should_continue is False


def test_should_ask_user_property():
    """Should test should_ask_user logic."""
    # Should ask when next_action="ask_user"
    raw1 = '{"conclusion": "Test", "next_action": "ask_user"}'
    parsed1 = parse_llm_response(raw1)
    assert parsed1.should_ask_user is True

    # Should ask when needs_more_info=True
    raw2 = '{"conclusion": "Test", "needs_more_info": true}'
    parsed2 = parse_llm_response(raw2)
    assert parsed2.should_ask_user is True

    # Should NOT ask when both false
    raw3 = '{"conclusion": "Test", "next_action": "finalize", "needs_more_info": false}'
    parsed3 = parse_llm_response(raw3)
    assert parsed3.should_ask_user is False

    # Should ask when both true
    raw4 = '{"conclusion": "Test", "next_action": "ask_user", "needs_more_info": true}'
    parsed4 = parse_llm_response(raw4)
    assert parsed4.should_ask_user is True


def test_empty_response():
    """Should handle empty response."""
    raw = ""
    parsed = parse_llm_response(raw)

    assert not parsed.is_valid
    assert parsed.conclusion == ""
    assert parsed.error == "invalid_json"


def test_whitespace_only():
    """Should handle whitespace-only response."""
    raw = "   \n\t  "
    parsed = parse_llm_response(raw)

    assert not parsed.is_valid
    assert parsed.conclusion == ""
    assert parsed.error == "invalid_json"


def test_markdown_without_json_wrapper():
    """Should extract JSON from generic ``` blocks."""
    raw = '''```
{"conclusion": "Generic block", "confidence": 0.6}
```'''
    parsed = parse_llm_response(raw)

    assert parsed.is_valid
    assert parsed.conclusion == "Generic block"
    assert parsed.confidence == 0.6


def test_markdown_with_leading_text():
    """Should handle markdown with leading text before JSON."""
    raw = '''Some text before

```json
{"conclusion": "After text", "confidence": 0.75}
```

Some text after'''
    parsed = parse_llm_response(raw)

    # This will fail JSON parsing due to surrounding text
    # and fall back to raw text
    assert not parsed.is_valid
    assert parsed.error == "invalid_json"


def test_confidence_boundary_values():
    """Should handle confidence boundary values."""
    # Exactly 0.0
    raw1 = '{"conclusion": "Zero", "confidence": 0.0}'
    parsed1 = parse_llm_response(raw1)
    assert parsed1.confidence == 0.0

    # Exactly 1.0
    raw2 = '{"conclusion": "One", "confidence": 1.0}'
    parsed2 = parse_llm_response(raw2)
    assert parsed2.confidence == 1.0


def test_sub_tasks_empty_vs_missing():
    """Should distinguish between empty and missing sub_tasks."""
    # Empty list
    raw1 = '{"conclusion": "Test", "sub_tasks": []}'
    parsed1 = parse_llm_response(raw1)
    assert parsed1.sub_tasks == []
    assert not parsed1.should_continue

    # Missing field
    raw2 = '{"conclusion": "Test"}'
    parsed2 = parse_llm_response(raw2)
    assert parsed2.sub_tasks == []
    assert not parsed2.should_continue


def test_files_field():
    """Should parse files field correctly."""
    raw = '{"conclusion": "Test", "files": ["a.py", "b.py", "c.py"]}'
    parsed = parse_llm_response(raw)

    assert parsed.files == ["a.py", "b.py", "c.py"]


def test_reasoning_field():
    """Should parse reasoning field."""
    raw = '{"reasoning": "My reasoning", "conclusion": "Answer"}'
    parsed = parse_llm_response(raw)

    assert parsed.reasoning == "My reasoning"
    assert parsed.conclusion == "Answer"


def test_raw_response_preserved():
    """Should preserve raw response in output."""
    raw = '{"conclusion": "Test", "confidence": 0.9}'
    parsed = parse_llm_response(raw)

    assert parsed.raw_response == raw


def test_malformed_json_in_markdown():
    """Should handle malformed JSON in markdown blocks."""
    raw = '''```json
{"conclusion": "broken", "confidence": }
```'''
    parsed = parse_llm_response(raw)

    assert not parsed.is_valid
    assert parsed.error == "invalid_json"
