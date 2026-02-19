"""Simple synchronous LLM client for REPL environment.

Design doc reference: src/llm_client.py in target architecture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


class LLMError(Exception):
    """LLM call failed."""
    pass


@dataclass
class LLMClient:
    """
    Provider-agnostic LLM client.

    Simple synchronous interface for REPL's llm() function.
    Default model cascade: root uses larger model, sub-calls use smaller.
    """

    api_key: str | None = None
    default_model: str = "glm-4.7"
    model_cascade: dict[int, str] = field(default_factory=lambda: {
        0: "glm-4.7",  # root
        1: "glm-4.7",  # depth 1
        2: "glm-4.7",  # depth 2+
        3: "glm-4.7",
    })

    def __post_init__(self):
        """Initialize API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def get_model_for_depth(self, depth: int) -> str:
        """
        Get appropriate model for recursion depth.

        Args:
            depth: Current recursion depth (0 = root)

        Returns:
            Model identifier string
        """
        # Check for custom model env vars
        sonnet_model = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", self.default_model)
        haiku_model = os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", self.default_model)

        # Route to cheaper models at deeper depths
        depth_model_map = {
            0: sonnet_model,
            1: sonnet_model,
            2: haiku_model,
            3: haiku_model,
        }
        return depth_model_map.get(depth, haiku_model)

    def call(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        model: str | None = None,
        depth: int = 0,
    ) -> str:
        """
        Make a synchronous LLM call.

        Args:
            query: The query/prompt string
            context: Optional context dict (files, prior_results, etc.)
            model: Optional model override (None = use default for depth)
            depth: Current recursion depth for model selection

        Returns:
            LLM response as string

        Raises:
            ValueError: If query is empty
            LLMError: If the LLM call fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Select model
        selected_model = model if model else self.get_model_for_depth(depth)

        # Build full prompt with context
        full_prompt = self._build_prompt(query, context)

        # Make the actual API call
        return self._api_call(selected_model, full_prompt)

    def _build_prompt(self, query: str, context: dict[str, Any] | None) -> str:
        """Build full prompt from query and context."""
        if not context:
            return query

        parts = []

        # Add file contents if provided
        if "files" in context:
            for path, content in context["files"].items():
                parts.append(f"File: {path}\n```\n{content}\n```")

        # Add prior results if provided
        if "prior_results" in context:
            for name, result in context["prior_results"].items():
                parts.append(f"{name} = {result}")

        parts.append(f"\n{query}")
        return "\n\n".join(parts)

    def _api_call(self, model: str, prompt: str) -> str:
        """
        Make the actual API call.

        This is a placeholder that should be implemented based on
        the actual LLM provider being used.
        """
        # Placeholder - actual implementation depends on provider
        # For testing, this can be mocked
        raise LLMError("LLMClient._api_call not implemented - subclass or mock required")


__all__ = ["LLMClient", "LLMError"]
