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
from dataclasses import dataclass
from enum import Enum
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

    # === Cascade Configuration ===
    # Enable four-level cascade: SetFit → GLiNER2 → Local LLM → Heuristics
    use_cascade: bool = True

    # Level 1: SetFit classifier (fastest, ~5ms)
    setfit_enabled: bool = True
    setfit_confidence_threshold: float = 0.85  # Use SetFit if confidence >= this

    # Level 2: GLiNER2 + weighted rules (~10ms)
    gliner_enabled: bool = True
    gliner_confidence_threshold: float = 0.8  # Use GLiNER if confidence >= this

    # Level 3: Local LLM (~50ms)
    llm_confidence_threshold: float = 0.7  # Use LLM if confidence >= this

    # === Logging Configuration ===
    # Enable decision logging for training data collection
    log_decisions: bool = False

    # Path for decision log file
    log_path: str = "~/.rlm/orchestration_decisions.jsonl"

    # Whether to log heuristic decisions (can be noisy)
    log_heuristics: bool = True

    # === Calibration Configuration ===
    # Enable confidence calibration tracking
    calibration_enabled: bool = True

    # Minimum samples before adjusting thresholds
    calibration_min_samples: int = 20

    # Maximum threshold adjustment per calibration (prevents oscillation)
    calibration_max_adjustment: float = 0.05

    # Threshold bounds (prevent thresholds from going too extreme)
    calibration_threshold_min: float = 0.5
    calibration_threshold_max: float = 0.95


@dataclass
class CalibrationSample:
    """A single calibration sample tracking predicted vs actual outcome."""

    decision_id: str  # Unique ID for tracking
    cascade_level: int  # 1=SetFit, 2=GLiNER, 3=LLM, 4=Heuristics
    predicted_confidence: float  # Confidence at decision time
    predicted_activate_rlm: bool  # What we predicted
    actual_correct: bool | None = None  # Whether prediction was correct (set later)
    timestamp: float = 0.0  # When decision was made


@dataclass
class CalibrationStats:
    """Calibration statistics for a cascade level."""

    level: int
    samples: int = 0
    total_confidence: float = 0.0
    correct_predictions: int = 0
    calibration_error: float = 0.0  # |avg_confidence - accuracy|

    @property
    def avg_confidence(self) -> float:
        """Average predicted confidence."""
        return self.total_confidence / self.samples if self.samples > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Actual accuracy rate."""
        return self.correct_predictions / self.samples if self.samples > 0 else 0.0

    def update_calibration_error(self) -> None:
        """Recompute calibration error."""
        self.calibration_error = abs(self.avg_confidence - self.accuracy)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "samples": self.samples,
            "avg_confidence": round(self.avg_confidence, 3),
            "accuracy": round(self.accuracy, 3),
            "calibration_error": round(self.calibration_error, 3),
        }


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
        if (
            tokenizer is not None
            and hasattr(tokenizer, "chat_template")
            and tokenizer.chat_template is not None
        ):
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

    Implements a four-level cascade for fast, accurate routing:
    1. SetFit classifier (~5ms) - if confidence >= 0.85
    2. GLiNER2 + weighted rules (~10ms) - if confidence >= 0.8
    3. Local LLM (~50ms) - if confidence >= 0.7
    4. Heuristics (~1ms) - fallback

    Provides low-latency routing decisions using local models,
    with graceful fallback through the cascade on error or low confidence.
    """

    def __init__(
        self,
        config: LocalModelConfig | None = None,
        system_prompt: str | None = None,
    ):
        self.config = config or LocalModelConfig()
        self._runner: LocalModelRunner | None = None
        self._setfit_classifier: Any = None  # Lazy initialized
        self._gliner_extractor: Any = None  # Lazy initialized
        self._setfit_load_attempted = False
        self._gliner_load_attempted = False
        self._stats = {
            # Cascade level stats
            "setfit_decisions": 0,
            "gliner_decisions": 0,
            "local_decisions": 0,
            "heuristic_fallbacks": 0,
            # General stats
            "cache_hits": 0,
            "errors": 0,
            "total_latency_ms": 0.0,
            # Cascade fallthrough stats
            "setfit_low_confidence": 0,
            "gliner_low_confidence": 0,
            "llm_low_confidence": 0,
        }
        self._cache: dict[str, dict[str, Any]] = {}
        self._decision_logger: Any = None  # Lazy initialized

        # Calibration tracking
        self._calibration_stats: dict[int, CalibrationStats] = {
            1: CalibrationStats(level=1),  # SetFit
            2: CalibrationStats(level=2),  # GLiNER
            3: CalibrationStats(level=3),  # LLM
            4: CalibrationStats(level=4),  # Heuristics
        }
        self._pending_samples: dict[str, CalibrationSample] = {}  # decision_id -> sample
        self._decision_counter = 0

        # Import the system prompt from intelligent_orchestrator
        if system_prompt is None:
            try:
                from .intelligent_orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
            except ImportError:
                from intelligent_orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

            self._system_prompt = ORCHESTRATOR_SYSTEM_PROMPT
        else:
            self._system_prompt = system_prompt

    def _ensure_setfit_classifier(self) -> Any:
        """Lazily initialize SetFit classifier."""
        # Skip if disabled
        if not self.config.setfit_enabled:
            return None

        if self._setfit_load_attempted:
            return self._setfit_classifier

        self._setfit_load_attempted = True

        try:
            from .setfit_classifier import SetFitClassifier, is_setfit_available

            if not is_setfit_available():
                logger.debug("SetFit not installed, skipping Level 1 cascade")
                return None

            self._setfit_classifier = SetFitClassifier()
            if self._setfit_classifier.is_available:
                logger.info("SetFit classifier loaded for cascade Level 1")
            else:
                logger.debug("SetFit model not available, will use fallback")
            return self._setfit_classifier

        except ImportError:
            logger.debug("SetFit classifier module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize SetFit classifier: {e}")
            return None

    def _ensure_gliner_extractor(self) -> Any:
        """Lazily initialize GLiNER extractor."""
        # Skip if disabled
        if not self.config.gliner_enabled:
            return None

        if self._gliner_load_attempted:
            return self._gliner_extractor

        self._gliner_load_attempted = True

        try:
            from .gliner_extractor import GLiNERExtractor, is_gliner_available

            if not is_gliner_available():
                logger.debug("GLiNER not installed, skipping Level 2 cascade")
                return None

            self._gliner_extractor = GLiNERExtractor()
            if self._gliner_extractor.is_available:
                logger.info("GLiNER extractor loaded for cascade Level 2")
            else:
                logger.debug("GLiNER model not available, will use keyword fallback")
            return self._gliner_extractor

        except ImportError:
            logger.debug("GLiNER extractor module not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize GLiNER extractor: {e}")
            return None

    def _ensure_decision_logger(self) -> Any:
        """Lazily initialize decision logger for training data collection."""
        if self._decision_logger is not None:
            return self._decision_logger

        if not self.config.log_decisions:
            return None

        try:
            from .orchestration_logger import LoggerConfig, OrchestrationLogger

            log_config = LoggerConfig(
                log_path=self.config.log_path,
                enabled=True,
                log_heuristics=self.config.log_heuristics,
            )
            self._decision_logger = OrchestrationLogger(config=log_config)
            logger.info(f"Decision logging enabled: {self.config.log_path}")
            return self._decision_logger

        except ImportError:
            logger.debug("OrchestrationLogger not available")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize decision logger: {e}")
            return None

    def _log_decision(
        self,
        query: str,
        context_summary: str,
        decision: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Log an orchestration decision for training data collection."""
        decision_logger = self._ensure_decision_logger()
        if decision_logger is None:
            return

        # Determine source from cascade level
        cascade_source = decision.get("_cascade_source", "unknown")
        source_map = {
            "setfit": "setfit",
            "gliner": "gliner",
            "local_llm": "local",
            "heuristics": "heuristic",
        }
        source = source_map.get(cascade_source, cascade_source)

        try:
            decision_logger.log_decision(
                query=query,
                context_summary=context_summary,
                decision=decision,
                source=source,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.debug(f"Failed to log decision: {e}")

    def _record_calibration_sample(
        self,
        decision: dict[str, Any],
    ) -> str:
        """
        Record a calibration sample for later outcome tracking.

        Args:
            decision: The orchestration decision

        Returns:
            The decision_id for tracking the outcome
        """
        import time

        if not self.config.calibration_enabled:
            return ""

        self._decision_counter += 1
        decision_id = f"d_{self._decision_counter}"

        cascade_level = decision.get("_cascade_level", 4)
        sample = CalibrationSample(
            decision_id=decision_id,
            cascade_level=cascade_level,
            predicted_confidence=decision.get("confidence", 0.5),
            predicted_activate_rlm=decision.get("activate_rlm", False),
            timestamp=time.time(),
        )

        self._pending_samples[decision_id] = sample

        # Limit pending samples to prevent memory growth
        if len(self._pending_samples) > 1000:
            # Remove oldest samples
            oldest_keys = sorted(
                self._pending_samples.keys(),
                key=lambda k: self._pending_samples[k].timestamp,
            )[:500]
            for k in oldest_keys:
                del self._pending_samples[k]

        return decision_id

    def record_outcome(
        self,
        decision_id: str,
        was_correct: bool,
    ) -> bool:
        """
        Record whether a decision was correct for calibration.

        Call this after observing the outcome of a decision to improve
        future calibration. For example, if RLM was activated but wasn't
        needed (user got quick answer), record was_correct=False.

        Args:
            decision_id: The ID returned from orchestrate() in _decision_id
            was_correct: Whether the activation decision was appropriate

        Returns:
            True if the outcome was recorded, False if decision_id not found
        """
        if not self.config.calibration_enabled:
            return False

        sample = self._pending_samples.pop(decision_id, None)
        if sample is None:
            return False

        # Update calibration stats for the level
        level = sample.cascade_level
        stats = self._calibration_stats.get(level)
        if stats is not None:
            stats.samples += 1
            stats.total_confidence += sample.predicted_confidence
            if was_correct:
                stats.correct_predictions += 1
            stats.update_calibration_error()

            # Maybe adjust thresholds
            self._maybe_adjust_thresholds(level)

        return True

    def _maybe_adjust_thresholds(self, level: int) -> None:
        """
        Adjust confidence thresholds based on calibration data.

        If predicted confidence is consistently higher than accuracy,
        raise the threshold. If lower, lower it.
        """
        stats = self._calibration_stats.get(level)
        if stats is None or stats.samples < self.config.calibration_min_samples:
            return

        # Calibration adjustment: move threshold toward accuracy
        # If confidence > accuracy, model is overconfident → raise threshold
        # If confidence < accuracy, model is underconfident → lower threshold
        adjustment = (stats.avg_confidence - stats.accuracy) * 0.1

        # Clamp adjustment
        adjustment = max(
            -self.config.calibration_max_adjustment,
            min(self.config.calibration_max_adjustment, adjustment),
        )

        if abs(adjustment) < 0.001:
            return  # Too small to matter

        # Apply adjustment to the appropriate threshold
        if level == 1:  # SetFit
            new_threshold = self.config.setfit_confidence_threshold + adjustment
            self.config.setfit_confidence_threshold = max(
                self.config.calibration_threshold_min,
                min(self.config.calibration_threshold_max, new_threshold),
            )
            logger.debug(
                f"Adjusted SetFit threshold to {self.config.setfit_confidence_threshold:.3f} "
                f"(calibration error: {stats.calibration_error:.3f})"
            )
        elif level == 2:  # GLiNER
            new_threshold = self.config.gliner_confidence_threshold + adjustment
            self.config.gliner_confidence_threshold = max(
                self.config.calibration_threshold_min,
                min(self.config.calibration_threshold_max, new_threshold),
            )
            logger.debug(
                f"Adjusted GLiNER threshold to {self.config.gliner_confidence_threshold:.3f} "
                f"(calibration error: {stats.calibration_error:.3f})"
            )
        elif level == 3:  # LLM
            new_threshold = self.config.llm_confidence_threshold + adjustment
            self.config.llm_confidence_threshold = max(
                self.config.calibration_threshold_min,
                min(self.config.calibration_threshold_max, new_threshold),
            )
            logger.debug(
                f"Adjusted LLM threshold to {self.config.llm_confidence_threshold:.3f} "
                f"(calibration error: {stats.calibration_error:.3f})"
            )

    def get_calibration_stats(self) -> dict[str, Any]:
        """
        Get calibration statistics for all cascade levels.

        Returns:
            Dict with per-level calibration stats and current thresholds
        """
        return {
            "levels": {
                1: self._calibration_stats[1].to_dict(),
                2: self._calibration_stats[2].to_dict(),
                3: self._calibration_stats[3].to_dict(),
                4: self._calibration_stats[4].to_dict(),
            },
            "current_thresholds": {
                "setfit": self.config.setfit_confidence_threshold,
                "gliner": self.config.gliner_confidence_threshold,
                "llm": self.config.llm_confidence_threshold,
            },
            "pending_samples": len(self._pending_samples),
            "calibration_enabled": self.config.calibration_enabled,
        }

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
        Make orchestration decision using four-level cascade.

        Cascade order (stops at first confident result):
        1. SetFit classifier (~5ms) - if confidence >= 0.85
        2. GLiNER2 + weighted rules (~10ms) - if confidence >= 0.8
        3. Local LLM (~50ms) - if confidence >= 0.7
        4. Heuristics (~1ms) - fallback

        Args:
            query: The user query
            context_summary: Summary of current context

        Returns:
            Orchestration decision dict (same schema as LLM orchestrator)
        """
        import time

        start_time = time.perf_counter()

        # Check cache
        cache_key = self._compute_cache_key(query, context_summary)
        if self.config.cache_enabled and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        decision: dict[str, Any] | None = None

        # === Level 1: SetFit Classifier (~5ms) ===
        if self.config.use_cascade and self.config.setfit_enabled:
            decision = self._try_setfit_classify(query, context_summary)
            if decision is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                self._stats["total_latency_ms"] += latency_ms
                decision["_cascade_level"] = 1
                decision["_cascade_source"] = "setfit"
                decision["_decision_id"] = self._record_calibration_sample(decision)
                self._log_decision(query, context_summary, decision, latency_ms)
                if self.config.cache_enabled:
                    self._update_cache(cache_key, decision)
                return decision

        # === Level 2: GLiNER2 + Weighted Rules (~10ms) ===
        if self.config.use_cascade and self.config.gliner_enabled:
            decision = self._try_gliner_classify(query, context_summary)
            if decision is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                self._stats["total_latency_ms"] += latency_ms
                decision["_cascade_level"] = 2
                decision["_cascade_source"] = "gliner"
                decision["_decision_id"] = self._record_calibration_sample(decision)
                self._log_decision(query, context_summary, decision, latency_ms)
                if self.config.cache_enabled:
                    self._update_cache(cache_key, decision)
                return decision

        # === Level 3: Local LLM (~50ms) ===
        decision = await self._try_llm_classify(query, context_summary)
        if decision is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._stats["total_latency_ms"] += latency_ms
            decision["_cascade_level"] = 3
            decision["_cascade_source"] = "local_llm"
            decision["_decision_id"] = self._record_calibration_sample(decision)
            self._log_decision(query, context_summary, decision, latency_ms)
            if self.config.cache_enabled:
                self._update_cache(cache_key, decision)
            return decision

        # === Level 4: Heuristics (~1ms) ===
        self._stats["heuristic_fallbacks"] += 1
        decision = self._heuristic_decision(query, context_summary)
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._stats["total_latency_ms"] += latency_ms
        decision["_cascade_level"] = 4
        decision["_cascade_source"] = "heuristics"
        decision["_decision_id"] = self._record_calibration_sample(decision)
        self._log_decision(query, context_summary, decision, latency_ms)
        if self.config.cache_enabled:
            self._update_cache(cache_key, decision)
        return decision

    def _try_setfit_classify(
        self,
        query: str,
        context_summary: str,
    ) -> dict[str, Any] | None:
        """
        Level 1: Try SetFit classifier.

        Returns decision dict if confidence >= threshold, None otherwise.
        """
        classifier = self._ensure_setfit_classifier()
        if classifier is None:
            return None

        try:
            result = classifier.classify(query, context_summary)

            if result.confidence >= self.config.setfit_confidence_threshold:
                self._stats["setfit_decisions"] += 1
                return {
                    "activate_rlm": result.activate_rlm,
                    "activation_reason": (
                        result.signals[0] if result.signals else "setfit_classification"
                    ),
                    "execution_mode": self._complexity_to_mode(result.complexity_level),
                    "model_tier": "balanced",
                    "depth_budget": self._complexity_to_depth(result.complexity_level),
                    "tool_access": "read_only" if result.activate_rlm else "none",
                    "query_type": "unknown",
                    "complexity_score": self._complexity_to_score(result.complexity_level),
                    "confidence": result.confidence,
                    "signals": list(result.signals),
                }
            else:
                self._stats["setfit_low_confidence"] += 1
                logger.debug(
                    f"SetFit confidence {result.confidence:.2f} < threshold "
                    f"{self.config.setfit_confidence_threshold}, continuing cascade"
                )
                return None

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"SetFit classification failed: {e}")
            return None

    def _try_gliner_classify(
        self,
        query: str,
        context_summary: str,
    ) -> dict[str, Any] | None:
        """
        Level 2: Try GLiNER2 + weighted rules.

        Returns decision dict if confidence >= threshold, None otherwise.
        """
        extractor = self._ensure_gliner_extractor()
        if extractor is None:
            return None

        try:
            # Extract signals
            signals_result = extractor.extract_signals(query)

            # Apply weighted rules to get activation decision
            rules_result = extractor.get_weighted_rules_result(signals_result)

            if rules_result["confidence"] >= self.config.gliner_confidence_threshold:
                self._stats["gliner_decisions"] += 1
                return {
                    "activate_rlm": rules_result["activate_rlm"],
                    "activation_reason": rules_result["reason"][:50],  # Truncate long reasons
                    "execution_mode": "thorough" if rules_result["activate_rlm"] else "balanced",
                    "model_tier": "balanced",
                    "depth_budget": 2 if rules_result["activate_rlm"] else 0,
                    "tool_access": "read_only" if rules_result["activate_rlm"] else "none",
                    "query_type": "unknown",
                    "complexity_score": rules_result["complexity_score"],
                    "confidence": rules_result["confidence"],
                    "signals": rules_result["signals"],
                }
            else:
                self._stats["gliner_low_confidence"] += 1
                logger.debug(
                    f"GLiNER confidence {rules_result['confidence']:.2f} < threshold "
                    f"{self.config.gliner_confidence_threshold}, continuing cascade"
                )
                return None

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"GLiNER extraction failed: {e}")
            return None

    async def _try_llm_classify(
        self,
        query: str,
        context_summary: str,
    ) -> dict[str, Any] | None:
        """
        Level 3: Try local LLM classification.

        Returns decision dict if confidence >= threshold, None otherwise.
        """
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

            # Parse response
            decision = self._parse_response(result.content)

            confidence = decision.get("confidence", 0.7)
            if confidence >= self.config.llm_confidence_threshold:
                self._stats["local_decisions"] += 1
                return decision
            else:
                self._stats["llm_low_confidence"] += 1
                logger.debug(
                    f"LLM confidence {confidence:.2f} < threshold "
                    f"{self.config.llm_confidence_threshold}, falling back to heuristics"
                )
                return None

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"Local LLM orchestration failed: {e}")

            if not self.config.fallback_to_heuristics:
                raise

            return None

    def _complexity_to_mode(self, complexity_level: Any) -> str:
        """Convert ComplexityLevel to execution mode string."""
        try:
            from .setfit_classifier import ComplexityLevel

            mode_map = {
                ComplexityLevel.TRIVIAL: "fast",
                ComplexityLevel.SIMPLE: "fast",
                ComplexityLevel.MODERATE: "balanced",
                ComplexityLevel.COMPLEX: "thorough",
                ComplexityLevel.UNBOUNDED: "thorough",
            }
            return mode_map.get(complexity_level, "balanced")
        except ImportError:
            return "balanced"

    def _complexity_to_depth(self, complexity_level: Any) -> int:
        """Convert ComplexityLevel to depth budget."""
        try:
            from .setfit_classifier import ComplexityLevel

            depth_map = {
                ComplexityLevel.TRIVIAL: 0,
                ComplexityLevel.SIMPLE: 1,
                ComplexityLevel.MODERATE: 2,
                ComplexityLevel.COMPLEX: 2,
                ComplexityLevel.UNBOUNDED: 3,
            }
            return depth_map.get(complexity_level, 2)
        except ImportError:
            return 2

    def _complexity_to_score(self, complexity_level: Any) -> float:
        """Convert ComplexityLevel to complexity score 0-1."""
        try:
            from .setfit_classifier import ComplexityLevel

            score_map = {
                ComplexityLevel.TRIVIAL: 0.1,
                ComplexityLevel.SIMPLE: 0.3,
                ComplexityLevel.MODERATE: 0.5,
                ComplexityLevel.COMPLEX: 0.7,
                ComplexityLevel.UNBOUNDED: 0.9,
            }
            return score_map.get(complexity_level, 0.5)
        except ImportError:
            return 0.5

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from local model."""
        # Extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response[:200]}")

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

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
        """Get orchestration statistics including cascade breakdown and calibration."""
        total = (
            self._stats["setfit_decisions"]
            + self._stats["gliner_decisions"]
            + self._stats["local_decisions"]
            + self._stats["heuristic_fallbacks"]
            + self._stats["cache_hits"]
        )
        decisions_made = (
            self._stats["setfit_decisions"]
            + self._stats["gliner_decisions"]
            + self._stats["local_decisions"]
            + self._stats["heuristic_fallbacks"]
        )
        return {
            **self._stats,
            "total_decisions": total,
            # Cascade level rates
            "setfit_rate": self._stats["setfit_decisions"] / total if total > 0 else 0.0,
            "gliner_rate": self._stats["gliner_decisions"] / total if total > 0 else 0.0,
            "local_rate": self._stats["local_decisions"] / total if total > 0 else 0.0,
            "heuristic_rate": self._stats["heuristic_fallbacks"] / total if total > 0 else 0.0,
            "cache_hit_rate": self._stats["cache_hits"] / total if total > 0 else 0.0,
            # Performance
            "avg_latency_ms": (
                self._stats["total_latency_ms"] / decisions_made if decisions_made > 0 else 0.0
            ),
            # Cascade efficiency (% resolved by fast levels 1-2)
            "fast_cascade_rate": (
                (self._stats["setfit_decisions"] + self._stats["gliner_decisions"]) / total
                if total > 0
                else 0.0
            ),
            # Calibration stats
            "calibration": self.get_calibration_stats(),
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
    "CalibrationSample",
    "CalibrationStats",
    "LocalModelBackend",
    "LocalModelConfig",
    "LocalModelRunner",
    "LocalOrchestrator",
    "MLXRunner",
    "OllamaRunner",
    "RECOMMENDED_CONFIGS",
]
