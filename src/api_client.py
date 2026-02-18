"""
Multi-provider LLM client for RLM.

Implements: Spec §5 Model Integration

Supports:
- Anthropic (Claude Opus, Sonnet, Haiku)
- OpenAI (GPT-5.2, GPT-5.2-Codex, GPT-4o)
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import anthropic
import openai
from dotenv import load_dotenv

from .cost_tracker import CostComponent, CostTracker, get_cost_tracker

# Auto-load .env from project root
_project_root = Path(__file__).parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)


class Provider(Enum):
    """LLM provider."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    CLAUDE_CLI = "claude_cli"  # Claude Code CLI (subscription auth)


@dataclass
class APIResponse:
    """Response from LLM API."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: Provider
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A chunk from streaming response."""

    text: str
    is_final: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


# Progress callback type for long-running LLM calls
ProgressCallback = Callable[[int, float], None]  # (elapsed_seconds, timeout_seconds)


# Model registry with provider info
MODEL_REGISTRY: dict[str, tuple[Provider, str]] = {
    # Anthropic models
    "opus": (Provider.ANTHROPIC, "claude-opus-4-5-20251101"),
    "sonnet": (Provider.ANTHROPIC, "claude-sonnet-4-20250514"),
    "haiku": (Provider.ANTHROPIC, "claude-haiku-4-5-20251001"),
    "claude-opus-4-5-20251101": (Provider.ANTHROPIC, "claude-opus-4-5-20251101"),
    "claude-sonnet-4-20250514": (Provider.ANTHROPIC, "claude-sonnet-4-20250514"),
    "claude-haiku-4-5-20251001": (Provider.ANTHROPIC, "claude-haiku-4-5-20251001"),
    # Claude CLI models (subscription auth, same model names)
    "cli:opus": (Provider.CLAUDE_CLI, "opus"),
    "cli:sonnet": (Provider.CLAUDE_CLI, "sonnet"),
    "cli:haiku": (Provider.CLAUDE_CLI, "haiku"),
    # OpenAI GPT-5.2 models
    "gpt-5.2": (Provider.OPENAI, "gpt-5.2"),
    "gpt-5.2-pro": (Provider.OPENAI, "gpt-5.2-pro"),
    "gpt-5.2-chat-latest": (Provider.OPENAI, "gpt-5.2-chat-latest"),
    # GPT-5.2-Codex: API access coming soon, use gpt-5.2 as fallback
    "gpt-5.2-codex": (Provider.OPENAI, "gpt-5.2"),  # Fallback until API available
    # OpenAI GPT-4 models
    "gpt-4o": (Provider.OPENAI, "gpt-4o"),
    "gpt-4o-mini": (Provider.OPENAI, "gpt-4o-mini"),
    # OpenAI reasoning models
    "o1": (Provider.OPENAI, "o1"),
    "o1-mini": (Provider.OPENAI, "o1-mini"),
    "o3-mini": (Provider.OPENAI, "o3-mini"),
    # Shortcuts
    "codex": (Provider.OPENAI, "gpt-5.2"),  # Fallback until Codex API available
}


def resolve_model(model: str) -> tuple[Provider, str]:
    """Resolve model shorthand to (provider, full_model_id)."""
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]
    # Guess provider from model name
    if model.startswith("claude"):
        return (Provider.ANTHROPIC, model)
    if model.startswith(("gpt-", "o1", "o3")):
        return (Provider.OPENAI, model)
    # Default to Anthropic
    return (Provider.ANTHROPIC, model)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
        progress_callback: ProgressCallback | None = None,
    ) -> APIResponse:
        """Get completion from LLM."""
        pass

    @abstractmethod
    def complete_streaming(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Get streaming completion from LLM."""
        ...


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self,
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.cost_tracker = cost_tracker or get_cost_tracker()

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
        progress_callback: ProgressCallback | None = None,
    ) -> APIResponse:
        model = model or "claude-opus-4-5-20251101"

        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            request_params["system"] = system

        response = self.client.messages.create(**request_params)

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        self.cost_tracker.record_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
            component=component,
        )

        return APIResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
            provider=Provider.ANTHROPIC,
            stop_reason=response.stop_reason,
        )

    async def complete_streaming(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
    ) -> AsyncGenerator[StreamChunk, None]:
        model = model or "claude-opus-4-5-20251101"

        request_params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            request_params["system"] = system

        input_tokens = 0
        output_tokens = 0

        with self.client.messages.stream(**request_params) as stream:
            for text in stream.text_stream:
                yield StreamChunk(text=text)

            final_message = stream.get_final_message()
            input_tokens = final_message.usage.input_tokens
            output_tokens = final_message.usage.output_tokens

        self.cost_tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            component=component,
        )

        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT API client."""

    def __init__(
        self,
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.cost_tracker = cost_tracker or get_cost_tracker()

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
        progress_callback: ProgressCallback | None = None,
    ) -> APIResponse:
        model = model or "gpt-5.2-codex"

        # OpenAI uses system message in messages array
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # GPT-5.2+ uses max_completion_tokens instead of max_tokens
        if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
            response = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        self.cost_tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            component=component,
        )

        return APIResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            provider=Provider.OPENAI,
            stop_reason=response.choices[0].finish_reason,
        )

    async def complete_streaming(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
    ) -> AsyncGenerator[StreamChunk, None]:
        model = model or "gpt-5.2-codex"

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # GPT-5.2+ uses max_completion_tokens instead of max_tokens
        if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
            stream = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
        else:
            stream = self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

        input_tokens = 0
        output_tokens = 0

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(text=chunk.choices[0].delta.content)
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        self.cost_tracker.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            component=component,
        )

        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class ClaudeHeadlessClient(BaseLLMClient):
    """
    Claude CLI subprocess client using subscription auth.

    Implements: Spec §5 Model Integration (subscription-based)

    Uses `claude -p` for non-interactive completions. This allows
    RLM recursion without requiring ANTHROPIC_API_KEY - it uses
    the user's Claude Max/Pro subscription instead.

    Key features:
    - No API key required (uses subscription auth)
    - Supports model selection (opus/sonnet/haiku)
    - Returns structured JSON with cost tracking
    - Isolated subprocess per call (no shared state)
    """

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        cli_path: str | None = None,
        timeout: float = 600.0,  # 10 minutes for complex queries
    ):
        """
        Initialize Claude CLI client.

        Args:
            cost_tracker: Cost tracker instance
            cli_path: Path to claude CLI (default: auto-detect)
            timeout: Timeout for subprocess in seconds
        """
        self.cost_tracker = cost_tracker or get_cost_tracker()
        self.cli_path = cli_path or self._find_cli()
        self.timeout = timeout

        if not self.cli_path:
            raise ValueError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

    def _find_cli(self) -> str | None:
        """Find claude CLI in PATH."""
        import shutil
        return shutil.which("claude")

    def _build_prompt(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
    ) -> str:
        """Build prompt string from messages."""
        parts = []

        if system:
            parts.append(f"System: {system}\n")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")

        return "\n".join(parts).strip()

    def _resolve_model(self, model: str | None) -> str:
        """Resolve model to CLI-compatible name (opus/sonnet/haiku)."""
        if model is None:
            return "sonnet"

        # Already a valid CLI model
        if model in ("opus", "sonnet", "haiku"):
            return model

        # Map full model names to shortcuts
        if "opus" in model:
            return "opus"
        elif "sonnet" in model:
            return "sonnet"
        elif "haiku" in model:
            return "haiku"

        # Default to sonnet for unknown models
        return "sonnet"

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
        progress_callback: ProgressCallback | None = None,
    ) -> APIResponse:
        """Get completion via Claude CLI subprocess with optional progress tracking."""
        import json

        model = self._resolve_model(model)
        prompt = self._build_prompt(messages, system)

        # Build CLI command
        cmd = [
            self.cli_path,
            "-p",
            "--no-session-persistence",
            f"--model={model}",
            "--output-format=json",
            prompt,
        ]

        # Create subprocess with CLAUDECODE unset to allow nested execution
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)  # Unset to bypass nested session check

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Track elapsed time and emit progress
            start_time = time.monotonic()
            heartbeat_interval = 5.0  # seconds between progress callbacks

            async def run_with_progress():
                """Run subprocess with periodic progress callbacks."""
                if progress_callback is None:
                    # No callback, just wait
                    return await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout,
                    )

                # With progress callback
                while True:
                    try:
                        # Wait for process with short timeout for progress checks
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(),
                            timeout=heartbeat_interval,
                        )
                        return stdout, stderr
                    except asyncio.TimeoutError:
                        # Process still running, emit progress
                        elapsed = time.monotonic() - start_time
                        progress_callback(int(elapsed), self.timeout)
                        continue

            stdout, stderr = await run_with_progress()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"Claude CLI failed: {error_msg}")

            # Parse JSON response
            result = json.loads(stdout.decode())

            # Extract fields
            content = result.get("result", "")
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cost_usd = result.get("total_cost_usd", 0.0)

            # Record cost
            self.cost_tracker.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                component=component,
            )

            return APIResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                provider=Provider.CLAUDE_CLI,
                stop_reason=result.get("stop_reason"),
                metadata={
                    "cost_usd": cost_usd,
                    "duration_ms": result.get("duration_ms"),
                    "session_id": result.get("session_id"),
                },
            )

        except asyncio.TimeoutError:
            raise RuntimeError(f"Claude CLI timed out after {self.timeout}s")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Claude CLI output: {e}")

    async def complete_streaming(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Streaming completion via Claude CLI.

        Note: Uses --output-format=stream-json for real-time output.
        """
        import asyncio
        import json

        model = self._resolve_model(model)
        prompt = self._build_prompt(messages, system)

        cmd = [
            self.cli_path,
            "-p",
            "--no-session-persistence",
            f"--model={model}",
            "--output-format=stream-json",
            prompt,
        ]

        env = dict(os.environ)
        env.pop("CLAUDECODE", None)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            input_tokens = 0
            output_tokens = 0

            # Read streaming output
            while True:
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=self.timeout,
                )

                if not line:
                    break

                try:
                    chunk = json.loads(line.decode())
                    chunk_type = chunk.get("type", "")

                    if chunk_type == "content":
                        # Text chunk
                        text = chunk.get("content", "")
                        if text:
                            yield StreamChunk(text=text)
                    elif chunk_type == "result":
                        # Final result
                        input_tokens = chunk.get("usage", {}).get("input_tokens", 0)
                        output_tokens = chunk.get("usage", {}).get("output_tokens", 0)

                except json.JSONDecodeError:
                    continue

            await process.wait()

            # Record cost
            self.cost_tracker.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                component=component,
            )

            yield StreamChunk(
                text="",
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except asyncio.TimeoutError:
            raise RuntimeError(f"Claude CLI streaming timed out after {self.timeout}s")


class MultiProviderClient:
    """
    Unified LLM client that routes to appropriate provider.

    Implements: Spec §5.1 API Integration
    """

    def __init__(
        self,
        default_model: str = "opus",
        anthropic_key: str | None = None,
        openai_key: str | None = None,
        cost_tracker: CostTracker | None = None,
        prefer_cli: bool = False,
    ):
        """
        Initialize multi-provider client.

        Args:
            default_model: Default model (defaults to Opus)
            anthropic_key: Anthropic API key
            openai_key: OpenAI API key
            cost_tracker: Cost tracker instance
            prefer_cli: Prefer Claude CLI over API even if keys available
        """
        self.cost_tracker = cost_tracker or get_cost_tracker()
        self.default_model = default_model
        self._clients: dict[Provider, BaseLLMClient] = {}
        self.prefer_cli = prefer_cli

        # Try Claude CLI first if prefer_cli is set
        if prefer_cli:
            try:
                self._clients[Provider.CLAUDE_CLI] = ClaudeHeadlessClient(
                    cost_tracker=self.cost_tracker
                )
            except ValueError:
                pass  # CLI not available

        # Initialize API clients
        try:
            self._clients[Provider.ANTHROPIC] = AnthropicClient(
                api_key=anthropic_key, cost_tracker=self.cost_tracker
            )
        except ValueError:
            pass  # No Anthropic key available

        try:
            self._clients[Provider.OPENAI] = OpenAIClient(
                api_key=openai_key, cost_tracker=self.cost_tracker
            )
        except ValueError:
            pass  # No OpenAI key available

        # Try Claude CLI as fallback if no API keys
        if Provider.CLAUDE_CLI not in self._clients:
            try:
                self._clients[Provider.CLAUDE_CLI] = ClaudeHeadlessClient(
                    cost_tracker=self.cost_tracker
                )
            except ValueError:
                pass  # CLI not available

        if not self._clients:
            raise ValueError(
                "No LLM provider available. Options:\n"
                "1. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable\n"
                "2. Install Claude CLI: npm install -g @anthropic-ai/claude-code"
            )

    def _get_client(self, model: str) -> tuple[BaseLLMClient, str]:
        """Get appropriate client for model."""
        provider, full_model = resolve_model(model)

        # If requested provider not available, try fallbacks
        if provider not in self._clients:
            # For Anthropic models, fall back to CLI if available
            if provider == Provider.ANTHROPIC and Provider.CLAUDE_CLI in self._clients:
                provider = Provider.CLAUDE_CLI
                full_model = self._get_cli_model_name(full_model)
            else:
                available = list(self._clients.keys())
                raise ValueError(
                    f"No client for {provider.value}. Available: {[p.value for p in available]}"
                )

        return self._clients[provider], full_model

    def _get_cli_model_name(self, model: str) -> str:
        """Convert model name to CLI shortcut (opus/sonnet/haiku)."""
        if model in ("opus", "sonnet", "haiku"):
            return model
        if "opus" in model:
            return "opus"
        elif "sonnet" in model:
            return "sonnet"
        elif "haiku" in model:
            return "haiku"
        return "sonnet"  # Default

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
        progress_callback: ProgressCallback | None = None,
    ) -> APIResponse:
        """Get completion, routing to appropriate provider."""
        model = model or self.default_model
        client, full_model = self._get_client(model)

        return await client.complete(
            messages=messages,
            system=system,
            model=full_model,
            max_tokens=max_tokens,
            temperature=temperature,
            component=component,
            progress_callback=progress_callback,
        )

    async def complete_streaming(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        component: CostComponent = CostComponent.ROOT_PROMPT,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Get streaming completion, routing to appropriate provider."""
        model = model or self.default_model
        client, full_model = self._get_client(model)

        async for chunk in client.complete_streaming(
            messages=messages,
            system=system,
            model=full_model,
            max_tokens=max_tokens,
            temperature=temperature,
            component=component,
        ):
            yield chunk

    async def recursive_query(
        self,
        query: str,
        context: str,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """
        Make a recursive sub-query.

        Implements: Spec §3.3 Recursive Call Protocol
        """
        from .prompts import build_recursive_prompt

        # Use faster model for recursive calls
        model = model or "haiku"

        messages = [{"role": "user", "content": build_recursive_prompt(query, context)}]

        response = await self.complete(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            component=CostComponent.RECURSIVE_CALL,
        )

        return response.content

    async def summarize(
        self,
        content: str,
        max_tokens: int = 500,
        model: str | None = None,
    ) -> str:
        """Summarize content."""
        from .prompts import build_summarization_prompt

        model = model or "haiku"

        messages = [{"role": "user", "content": build_summarization_prompt(content, max_tokens)}]

        response = await self.complete(
            messages=messages,
            model=model,
            max_tokens=max_tokens + 100,
            component=CostComponent.SUMMARIZATION,
        )

        return response.content


# Backwards compatibility alias
ClaudeClient = MultiProviderClient

# Global client instance
_client: MultiProviderClient | None = None


def get_client() -> MultiProviderClient:
    """Get global LLM client."""
    global _client
    if _client is None:
        _client = MultiProviderClient()
    return _client


def init_client(
    api_key: str | None = None,
    default_model: str = "opus",
    **kwargs: Any,
) -> MultiProviderClient:
    """Initialize global client with options."""
    global _client
    _client = MultiProviderClient(
        default_model=default_model,
        anthropic_key=api_key,
        **kwargs,
    )
    return _client


__all__ = [
    "APIResponse",
    "AnthropicClient",
    "BaseLLMClient",
    "ClaudeClient",
    "ClaudeHeadlessClient",
    "MODEL_REGISTRY",
    "MultiProviderClient",
    "OpenAIClient",
    "ProgressCallback",
    "Provider",
    "StreamChunk",
    "get_client",
    "init_client",
    "resolve_model",
]
