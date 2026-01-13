"""
Local model orchestrator for RLM activation decisions.

Uses a small, efficient local model (e.g., Gemma 3 270M, LFM2-350M)
for low-latency routing decisions without API calls.

This module provides a drop-in replacement for the LLM-based orchestration
in intelligent_orchestrator.py, using local inference via MLX or llama.cpp.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalModelBackend(Enum):
    """Supported local inference backends."""

    MLX = "mlx"  # Apple Silicon optimized
    LLAMACPP = "llama.cpp"  # Cross-platform
    OLLAMA = "ollama"  # Easy setup, managed process


@dataclass
class LocalModelConfig:
    """Configuration for local orchestrator model."""

    # Model selection
    model_name: str = "gemma-3-270m-it"
    model_path: str | None = None  # Path to model weights (if not using Ollama)
    backend: LocalModelBackend = LocalModelBackend.MLX

    # Inference parameters
    max_tokens: int = 300
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 0.1

    # Performance
    timeout_ms: int = 2000  # Target <50ms, max 2s
    batch_size: int = 1

    # Fallback behavior
    fallback_to_heuristics: bool = True
    fallback_on_timeout: bool = True
    fallback_on_error: bool = True

    # Caching
    cache_enabled: bool = True
    cache_size: int = 200


@dataclass
class LocalInferenceResult:
    """Result from local model inference."""

    content: str
    latency_ms: float
    tokens_generated: int
    model_name: str
    backend: LocalModelBackend


class LocalModelRunner(ABC):
    """Abstract base class for local model inference."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LocalInferenceResult:
        """Generate response from local model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        pass


class MLXRunner(LocalModelRunner):
    """MLX-based local model runner for Apple Silicon."""

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def is_available(self) -> bool:
        """Check if MLX is available."""
        try:
            import mlx_lm  # noqa: F401

            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        """Lazy load the model."""
        if self._loaded:
            return

        try:
            from mlx_lm import load

            model_id = self.config.model_path or self._resolve_model_id()
            self._model, self._tokenizer = load(model_id)
            self._loaded = True
            logger.info(f"Loaded MLX model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise

    def _resolve_model_id(self) -> str:
        """Resolve model name to HuggingFace ID."""
        model_map = {
            "gemma-3-270m-it": "google/gemma-3-270m-it-qat-q4_0-mlx",
            "gemma-3-1b-it": "mlx-community/gemma-3-1b-it-4bit",
            "lfm2-350m": "LiquidAI/LFM2-350M-Instruct-MLX-bf16",
            "lfm2.5-1.2b": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16",
            "qwen3-0.6b": "mlx-community/Qwen3-0.6B-4bit",
            "smollm2-360m": "mlx-community/SmolLM2-360M-Instruct-4bit",
        }
        return model_map.get(self.config.model_name, self.config.model_name)

    async def generate(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LocalInferenceResult:
        """Generate response using MLX."""
        import time

        self._ensure_loaded()

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        # Format prompt with chat template
        tokenizer = self._tokenizer
        if tokenizer is not None and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"{system}\n\n{prompt}"

        # Create sampler with low temperature for deterministic routing
        sampler = make_sampler(
            temp=temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )
        logits_processors = make_logits_processors(repetition_penalty=1.05)

        start = time.perf_counter()
        response = generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Estimate tokens (rough)
        tokens_generated = len(response.split()) * 1.3

        return LocalInferenceResult(
            content=response,
            latency_ms=latency_ms,
            tokens_generated=int(tokens_generated),
            model_name=self.config.model_name,
            backend=LocalModelBackend.MLX,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "backend": "mlx",
            "model_name": self.config.model_name,
            "loaded": self._loaded,
            "model_id": self._resolve_model_id(),
        }


class OllamaRunner(LocalModelRunner):
    """Ollama-based local model runner."""

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self._client = None

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            import httpx

            response = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
            return response.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> LocalInferenceResult:
        """Generate response using Ollama."""
        import time

        import httpx

        model_name = self._resolve_model_name()

        payload = {
            "model": model_name,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
            },
        }

        start = time.perf_counter()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=self.config.timeout_ms / 1000,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start) * 1000

        return LocalInferenceResult(
            content=data.get("response", ""),
            latency_ms=latency_ms,
            tokens_generated=data.get("eval_count", 0),
            model_name=model_name,
            backend=LocalModelBackend.OLLAMA,
        )

    def _resolve_model_name(self) -> str:
        """Resolve to Ollama model name."""
        model_map = {
            "gemma-3-270m-it": "gemma3:270m",
            "gemma-3-1b-it": "gemma3:1b",
            "qwen3-0.6b": "qwen3:0.6b",
            "smollm2-360m": "smollm2:360m",
        }
        return model_map.get(self.config.model_name, self.config.model_name)

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "backend": "ollama",
            "model_name": self.config.model_name,
            "ollama_model": self._resolve_model_name(),
            "available": self.is_available(),
        }


class LocalOrchestrator:
    """
    Local model-based orchestrator for RLM activation decisions.

    Provides low-latency routing decisions using a small local model,
    with fallback to heuristics on error or timeout.
    """

    def __init__(
        self,
        config: LocalModelConfig | None = None,
        system_prompt: str | None = None,
    ):
        self.config = config or LocalModelConfig()
        self._runner: LocalModelRunner | None = None
        self._stats = {
            "local_decisions": 0,
            "heuristic_fallbacks": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_latency_ms": 0.0,
        }
        self._cache: dict[str, dict[str, Any]] = {}

        # Import the system prompt from intelligent_orchestrator
        if system_prompt is None:
            try:
                from .intelligent_orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
            except ImportError:
                from intelligent_orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

            self._system_prompt = ORCHESTRATOR_SYSTEM_PROMPT
        else:
            self._system_prompt = system_prompt

    def _get_runner(self) -> LocalModelRunner:
        """Get or create the appropriate model runner."""
        if self._runner is not None:
            return self._runner

        if self.config.backend == LocalModelBackend.MLX:
            runner = MLXRunner(self.config)
            if runner.is_available():
                self._runner = runner
                return runner
            logger.warning("MLX not available, trying Ollama")

        if self.config.backend == LocalModelBackend.OLLAMA or self._runner is None:
            runner = OllamaRunner(self.config)
            if runner.is_available():
                self._runner = runner
                return runner

        raise RuntimeError("No local model backend available")

    async def orchestrate(
        self,
        query: str,
        context_summary: str,
    ) -> dict[str, Any]:
        """
        Make orchestration decision using local model.

        Args:
            query: The user query
            context_summary: Summary of current context

        Returns:
            Orchestration decision dict (same schema as LLM orchestrator)
        """
        # Check cache
        cache_key = self._compute_cache_key(query, context_summary)
        if self.config.cache_enabled and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        # Try local model
        try:
            runner = self._get_runner()

            user_prompt = f"""Analyze this query and decide how to process it:

Query: {query}

Context:
{context_summary}

Output your decision as a JSON object."""

            result = await runner.generate(
                prompt=user_prompt,
                system=self._system_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            self._stats["local_decisions"] += 1
            self._stats["total_latency_ms"] += result.latency_ms

            # Parse response
            decision = self._parse_response(result.content)

            # Cache the decision
            if self.config.cache_enabled:
                self._update_cache(cache_key, decision)

            return decision

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Local orchestration failed: {e}")

            if self.config.fallback_to_heuristics:
                self._stats["heuristic_fallbacks"] += 1
                return self._heuristic_decision(query, context_summary)
            else:
                raise

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from local model."""
        # Extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response[:200]}")

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    def _heuristic_decision(
        self,
        query: str,
        context_summary: str,
    ) -> dict[str, Any]:
        """
        Fast heuristic-based decision when local model is unavailable.

        This mirrors the enhanced heuristics in intelligent_orchestrator.py
        """
        query_lower = query.lower()

        # High-value signals
        high_value = []
        if re.search(r"\bwhy\s+(is|does|did)\b", query_lower):
            high_value.append("discovery_required")
        if re.search(r"\ball\s+(usages?|instances?)\b", query_lower):
            high_value.append("synthesis_required")
        if re.search(r"\b(best|better)\s+(way|approach)\b", query_lower):
            high_value.append("uncertainty_high")
        if re.search(r"\b(flaky|intermittent|race)\b", query_lower):
            high_value.append("debugging_deep")
        if re.search(r"\b(architect|design.*system|migrat)", query_lower):
            high_value.append("architectural")

        # Low-value signals
        low_value = []
        if re.match(r"^(show|read|cat|view)\s+\S+$", query_lower):
            low_value.append("knowledge_retrieval")
        if re.match(r"^(ok|yes|no|thanks)\.?$", query_lower.strip()):
            low_value.append("conversational")

        # Decision
        activate_rlm = len(high_value) > 0 and not low_value
        if "large context" in context_summary.lower():
            activate_rlm = True
            high_value.append("large_context")

        return {
            "activate_rlm": activate_rlm,
            "activation_reason": high_value[0] if high_value else "simple_task",
            "execution_mode": "thorough" if len(high_value) >= 2 else "balanced",
            "model_tier": "balanced",
            "depth_budget": 2 if activate_rlm else 0,
            "tool_access": "read_only" if activate_rlm else "none",
            "query_type": "unknown",
            "complexity_score": min(1.0, len(high_value) * 0.3),
            "confidence": 0.6,
            "signals": high_value + low_value,
        }

    def _compute_cache_key(self, query: str, context_summary: str) -> str:
        """Compute cache key."""
        query_prefix = query[:100].lower().strip()
        context_hash = hash(context_summary[:200])
        return f"{hash(query_prefix)}_{context_hash}"

    def _update_cache(self, key: str, decision: dict[str, Any]) -> None:
        """Update cache with eviction."""
        if len(self._cache) >= self.config.cache_size:
            # Remove oldest entries
            oldest = list(self._cache.keys())[: self.config.cache_size // 4]
            for k in oldest:
                del self._cache[k]
        self._cache[key] = decision

    def get_statistics(self) -> dict[str, Any]:
        """Get orchestration statistics."""
        total = (
            self._stats["local_decisions"]
            + self._stats["heuristic_fallbacks"]
            + self._stats["cache_hits"]
        )
        return {
            **self._stats,
            "total_decisions": total,
            "local_rate": self._stats["local_decisions"] / total if total > 0 else 0.0,
            "cache_hit_rate": self._stats["cache_hits"] / total if total > 0 else 0.0,
            "avg_latency_ms": (
                self._stats["total_latency_ms"] / self._stats["local_decisions"]
                if self._stats["local_decisions"] > 0
                else 0.0
            ),
        }


# Recommended model configurations for different use cases
RECOMMENDED_CONFIGS = {
    # Fastest: ~30ms latency, good for simple routing
    "ultra_fast": LocalModelConfig(
        model_name="gemma-3-270m-it",
        backend=LocalModelBackend.MLX,
        max_tokens=200,
        temperature=0.1,
    ),
    # Balanced: ~50-80ms latency, better judgment
    "balanced": LocalModelConfig(
        model_name="qwen3-0.6b",
        backend=LocalModelBackend.MLX,
        max_tokens=300,
        temperature=0.1,
    ),
    # High quality: ~100-150ms latency, best decisions
    "quality": LocalModelConfig(
        model_name="lfm2.5-1.2b",
        backend=LocalModelBackend.MLX,
        max_tokens=400,
        temperature=0.1,
    ),
    # Cross-platform: Works on any system with Ollama
    "portable": LocalModelConfig(
        model_name="gemma-3-270m-it",
        backend=LocalModelBackend.OLLAMA,
        max_tokens=200,
        temperature=0.1,
    ),
}


__all__ = [
    "LocalModelBackend",
    "LocalModelConfig",
    "LocalModelRunner",
    "LocalOrchestrator",
    "MLXRunner",
    "OllamaRunner",
    "RECOMMENDED_CONFIGS",
]
