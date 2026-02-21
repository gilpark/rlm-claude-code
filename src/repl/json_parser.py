"""Safe JSON parsing for LLM responses."""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Type for next_action field
NextAction = Literal["continue", "finalize", "ask_user"]


@dataclass
class StructuredResponse:
    """Parsed LLM response with safe defaults."""
    reasoning: str = ""
    conclusion: str = ""
    confidence: float = 0.8
    files: list[str] = field(default_factory=list)
    sub_tasks: list[dict] = field(default_factory=list)
    needs_more_info: bool = False
    next_action: NextAction = "finalize"
    error: str | None = None
    raw_response: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if response was parsed successfully."""
        return self.error is None and bool(self.conclusion)

    @property
    def should_continue(self) -> bool:
        """Check if agent wants to continue recursion."""
        return self.next_action == "continue" and bool(self.sub_tasks)

    @property
    def should_ask_user(self) -> bool:
        """Check if agent needs user input."""
        return self.next_action == "ask_user" or self.needs_more_info


def _clamp_confidence(value: Any) -> float:
    """Clamp confidence to valid range [0.0, 1.0]."""
    try:
        conf = float(value)
        return max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        return 0.8  # Default


def _parse_next_action(value: Any) -> NextAction:
    """Parse next_action with validation."""
    if value in ("continue", "finalize", "ask_user"):
        return value
    return "finalize"  # Safe default


def parse_llm_response(raw: str) -> StructuredResponse:
    """
    Parse LLM response into structured format with safe fallback.

    Args:
        raw: Raw LLM response string

    Returns:
        StructuredResponse with parsed fields or safe defaults
    """
    # Strip whitespace
    raw = raw.strip()

    # Try to extract JSON from potential markdown code blocks
    json_str = raw
    if raw.startswith("```json"):
        json_str = raw[7:].strip()  # Remove ```json
    elif raw.startswith("```"):
        json_str = raw[3:].strip()  # Remove ```

    if json_str.endswith("```"):
        json_str = json_str[:-3].strip()  # Remove closing ```

    # Try to parse JSON
    try:
        data = json.loads(json_str)

        response = StructuredResponse(
            reasoning=data.get("reasoning", ""),
            conclusion=data.get("conclusion", ""),
            confidence=_clamp_confidence(data.get("confidence", 0.8)),
            files=data.get("files", []),
            sub_tasks=data.get("sub_tasks", []),
            needs_more_info=data.get("needs_more_info", False),
            next_action=_parse_next_action(data.get("next_action", "finalize")),
            error=data.get("error"),
            raw_response=raw,
        )

        # Log token savings (rough estimate)
        raw_tokens = len(raw) // 4
        parsed_tokens = len(response.conclusion) // 4 + 100  # overhead
        if raw_tokens > parsed_tokens:
            logger.info(f"Token savings: raw={raw_tokens}, parsed={parsed_tokens}, saved={raw_tokens - parsed_tokens}")

        return response

    except json.JSONDecodeError as e:
        logger.warning(f"LLM did not return valid JSON: {e}")

        # Fallback: treat raw text as conclusion
        return StructuredResponse(
            conclusion=raw,
            confidence=0.5,  # Lower trust for unstructured response
            error="invalid_json",
            raw_response=raw,
        )

    except (TypeError, ValueError) as e:
        logger.warning(f"Error parsing LLM response: {e}")

        return StructuredResponse(
            conclusion=raw,
            confidence=0.5,
            error=str(e),
            raw_response=raw,
        )


def extract_conclusion(raw: str) -> str:
    """Quick helper to extract just the conclusion."""
    parsed = parse_llm_response(raw)
    return parsed.conclusion or raw


def extract_sub_tasks(raw: str) -> list[dict]:
    """Quick helper to extract sub_tasks for recursion."""
    parsed = parse_llm_response(raw)
    return parsed.sub_tasks
