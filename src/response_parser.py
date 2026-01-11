"""
Parse Claude responses for REPL blocks and final answers.

Implements: Spec §3.4 Response Processing
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ResponseAction(Enum):
    """Type of action from parsed response."""

    REPL_EXECUTE = "repl_execute"  # Execute Python code
    FINAL_ANSWER = "final_answer"  # Final answer provided
    FINAL_VAR = "final_var"  # Answer in variable
    THINKING = "thinking"  # Intermediate reasoning
    TOOL_USE = "tool_use"  # Tool invocation


@dataclass
class ParsedResponse:
    """Parsed response from Claude."""

    action: ResponseAction
    content: str
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)


class ResponseParser:
    """
    Parse Claude responses for actionable content.

    Implements: Spec §3.4 Response Processing
    """

    # Patterns for extracting content
    PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
    FINAL_ANSWER = re.compile(r"FINAL:\s*(.+?)(?:\n\n|\Z)", re.DOTALL)
    FINAL_VAR = re.compile(r"FINAL_VAR:\s*(\w+)")

    def parse(self, response: str) -> list[ParsedResponse]:
        """
        Parse a response into actionable items.

        Args:
            response: Raw response from Claude

        Returns:
            List of ParsedResponse objects in order
        """
        results: list[ParsedResponse] = []

        # Check for FINAL answer first (highest priority)
        final_match = self.FINAL_ANSWER.search(response)
        if final_match:
            # Extract reasoning before FINAL
            reasoning = response[: final_match.start()].strip()
            results.append(
                ParsedResponse(
                    action=ResponseAction.FINAL_ANSWER,
                    content=final_match.group(1).strip(),
                    reasoning=reasoning,
                )
            )
            return results

        # Check for FINAL_VAR
        var_match = self.FINAL_VAR.search(response)
        if var_match:
            reasoning = response[: var_match.start()].strip()
            results.append(
                ParsedResponse(
                    action=ResponseAction.FINAL_VAR,
                    content=var_match.group(1),
                    reasoning=reasoning,
                )
            )
            return results

        # Extract Python blocks for REPL execution
        python_blocks = self.PYTHON_BLOCK.findall(response)
        if python_blocks:
            # Get reasoning (text before first code block)
            first_block_pos = response.find("```python")
            reasoning = response[:first_block_pos].strip() if first_block_pos > 0 else ""

            for code in python_blocks:
                results.append(
                    ParsedResponse(
                        action=ResponseAction.REPL_EXECUTE,
                        content=code.strip(),
                        reasoning=reasoning,
                    )
                )
                # Only use reasoning for first block
                reasoning = ""

            return results

        # No actionable content - treat as thinking/reasoning
        if response.strip():
            results.append(
                ParsedResponse(
                    action=ResponseAction.THINKING,
                    content=response.strip(),
                )
            )

        return results

    def extract_code_blocks(self, response: str) -> list[str]:
        """
        Extract all Python code blocks from response.

        Args:
            response: Raw response

        Returns:
            List of code strings
        """
        return [block.strip() for block in self.PYTHON_BLOCK.findall(response)]

    def has_final_answer(self, response: str) -> bool:
        """Check if response contains a final answer."""
        return bool(
            self.FINAL_ANSWER.search(response) or self.FINAL_VAR.search(response)
        )

    def extract_final_answer(self, response: str) -> str | None:
        """
        Extract final answer if present.

        Args:
            response: Raw response

        Returns:
            Final answer string or None
        """
        match = self.FINAL_ANSWER.search(response)
        if match:
            return match.group(1).strip()

        match = self.FINAL_VAR.search(response)
        if match:
            return f"[Variable: {match.group(1)}]"

        return None


def parse_response(response: str) -> list[ParsedResponse]:
    """Convenience function to parse response."""
    return ResponseParser().parse(response)


def extract_code(response: str) -> list[str]:
    """Convenience function to extract code blocks."""
    return ResponseParser().extract_code_blocks(response)


__all__ = [
    "ParsedResponse",
    "ResponseAction",
    "ResponseParser",
    "extract_code",
    "parse_response",
]
