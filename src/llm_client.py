"""Simple synchronous LLM client for REPL environment.

Design doc reference: src/llm_client.py in target architecture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import anthropic


class LLMError(Exception):
    """LLM call failed."""
    pass


@dataclass
class LLMClient:
    """
    Provider-agnostic LLM client.

    Simple synchronous interface for REPL's llm() function.
    Default model cascade: root uses larger model, sub-calls use smaller.

    Temperature is set LOW (0.1) by default for deterministic REPL output.
    REPL tasks don't need creativity - they need accuracy.
    """

    api_key: str | None = None
    default_model: str = "glm-5"
    base_url: str | None = None
    # Low temperature for deterministic REPL output (not creative)
    # 0.0-0.3 = deterministic, 0.4-0.7 = balanced, 0.8-1.0 = creative
    temperature: float = 0.1
    model_cascade: dict[int, str] = field(default_factory=lambda: {
        0: "glm-5",  # root - best accuracy
        1: "glm-5",  # depth 1
        2: "glm-4.7",  # depth 2+ - cheaper
        3: "glm-4.7",
    })
    _client: anthropic.Anthropic | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if self.base_url is None:
            self.base_url = os.environ.get("ANTHROPIC_BASE_URL")
        # Allow temperature override via env var
        if "ANTHROPIC_TEMPERATURE" in os.environ:
            self.temperature = float(os.environ["ANTHROPIC_TEMPERATURE"])

    def _get_client(self) -> anthropic.Anthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise LLMError("ANTHROPIC_API_KEY not set")
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            object.__setattr__(self, "_client", anthropic.Anthropic(**client_kwargs))
        return self._client

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
        max_tokens: int = 4096,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Make a synchronous LLM call.

        Args:
            query: The query/prompt string
            context: Optional context dict (files, prior_results, etc.)
            model: Optional model override (None = use default for depth)
            depth: Current recursion depth for model selection
            max_tokens: Maximum tokens in response
            system: Optional system prompt
            temperature: Optional temperature override (None = use self.temperature)

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

        # Use provided temperature or default
        temp = temperature if temperature is not None else self.temperature

        # Build full prompt with context
        full_prompt = self._build_prompt(query, context)

        # Make the actual API call
        return self._api_call(selected_model, full_prompt, max_tokens, system, temp)

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

    def _api_call(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 4096,
        system: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Make the actual API call to Anthropic."""
        try:
            client = self._get_client()

            request_params: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,  # Low for deterministic REPL output
            }

            if system:
                request_params["system"] = system

            response = client.messages.create(**request_params)

            # Extract text from response blocks
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            return content

        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}") from e
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}") from e


__all__ = ["LLMClient", "LLMError"]
