"""
SetFit-based complexity classifier for fast RLM activation decisions.

SetFit (Sentence-Transformer Fine-Tuning) provides ~5ms inference for
text classification, compared to ~50ms for local LLMs. This module wraps
SetFit for query complexity classification to decide RLM activation.

Implements: Plan §3.1 SetFit Classifier Integration
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Check if SetFit is available
try:
    from setfit import SetFitModel

    _SETFIT_AVAILABLE = True
except ImportError:
    SetFitModel = None  # type: ignore[assignment, misc]
    _SETFIT_AVAILABLE = False


class ComplexityLevel(Enum):
    """Query complexity levels for RLM activation."""

    TRIVIAL = "trivial"  # Simple lookup, no RLM needed
    SIMPLE = "simple"  # Straightforward task, RLM optional
    MODERATE = "moderate"  # Some decomposition helpful
    COMPLEX = "complex"  # Significant decomposition needed
    UNBOUNDED = "unbounded"  # Open-ended, needs careful budgeting


# Signal types that inform RLM activation
SignalType = Literal[
    "discovery_required",  # Needs codebase exploration
    "synthesis_required",  # Combines multiple sources
    "uncertainty_high",  # Unclear requirements
    "debugging_deep",  # Complex debugging needed
    "architectural",  # Architecture/design decisions
    "pattern_exhaustion",  # Previous approaches failed
    "knowledge_retrieval",  # Simple fact lookup
    "narrow_scope",  # Well-defined, limited scope
    "conversational",  # Chat/clarification
]


@dataclass
class ComplexityClassification:
    """
    Classification result for query complexity.

    Implements: Plan §3.1 Output schema
    """

    activate_rlm: bool
    complexity_level: ComplexityLevel
    signals: list[SignalType] = field(default_factory=list)
    confidence: float = 0.0  # 0.0-1.0

    # Metadata
    latency_ms: float = 0.0
    model_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "activate_rlm": self.activate_rlm,
            "complexity_level": self.complexity_level.value,
            "signals": list(self.signals),
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "model_used": self.model_used,
        }


# Complexity level to RLM activation mapping
LEVEL_ACTIVATION_MAP: dict[ComplexityLevel, bool] = {
    ComplexityLevel.TRIVIAL: False,
    ComplexityLevel.SIMPLE: False,
    ComplexityLevel.MODERATE: True,
    ComplexityLevel.COMPLEX: True,
    ComplexityLevel.UNBOUNDED: True,
}

# Signal to complexity level hints
SIGNAL_COMPLEXITY_HINTS: dict[str, ComplexityLevel] = {
    "discovery_required": ComplexityLevel.COMPLEX,
    "synthesis_required": ComplexityLevel.COMPLEX,
    "uncertainty_high": ComplexityLevel.MODERATE,
    "debugging_deep": ComplexityLevel.COMPLEX,
    "architectural": ComplexityLevel.UNBOUNDED,
    "pattern_exhaustion": ComplexityLevel.UNBOUNDED,
    "knowledge_retrieval": ComplexityLevel.TRIVIAL,
    "narrow_scope": ComplexityLevel.SIMPLE,
    "conversational": ComplexityLevel.TRIVIAL,
}


@dataclass
class SetFitConfig:
    """Configuration for SetFit classifier."""

    # Model path (local or HuggingFace)
    model_path: str = "~/.rlm/complexity_classifier"

    # Classification parameters
    confidence_threshold: float = 0.85  # Min confidence to use SetFit result

    # Performance
    max_length: int = 512  # Max input length
    batch_size: int = 1

    # Fallback behavior
    fallback_on_low_confidence: bool = True
    fallback_on_error: bool = True


class SetFitClassifier:
    """
    SetFit-based query complexity classifier.

    Provides fast (~5ms) classification of query complexity for RLM
    activation decisions. Falls back gracefully if SetFit unavailable.

    Implements: Plan §3.1 SetFit Classifier Integration
    """

    def __init__(self, config: SetFitConfig | None = None):
        """
        Initialize SetFit classifier.

        Args:
            config: Classifier configuration
        """
        self.config = config or SetFitConfig()
        self._model: Any = None
        self._loaded = False
        self._load_attempted = False
        self._model_mtime: float | None = None  # Track model file modification time

    @property
    def is_available(self) -> bool:
        """Check if SetFit is available and model is loadable."""
        if not _SETFIT_AVAILABLE:
            return False
        if not self._load_attempted:
            self._try_load()
        return self._loaded

    def _get_model_mtime(self) -> float | None:
        """Get model file modification time for hot-swap detection."""
        model_path = Path(self.config.model_path).expanduser()
        # Check for config.json as indicator of model presence
        config_file = model_path / "config.json"
        if config_file.exists():
            return config_file.stat().st_mtime
        elif model_path.exists() and model_path.is_file():
            return model_path.stat().st_mtime
        return None

    def _check_model_changed(self) -> bool:
        """Check if model file has been modified since last load."""
        if not self._loaded:
            return False
        current_mtime = self._get_model_mtime()
        if current_mtime is None:
            return False
        return self._model_mtime is not None and current_mtime > self._model_mtime

    def reload_model(self) -> bool:
        """
        Force reload of the model.

        Useful for hot-swapping when model file has been updated.

        Returns:
            True if reload successful, False otherwise.
        """
        self._model = None
        self._loaded = False
        self._load_attempted = False
        self._model_mtime = None
        self._try_load()
        return self._loaded

    def _try_load(self) -> None:
        """Attempt to load the model lazily."""
        if self._load_attempted:
            return

        self._load_attempted = True

        if not _SETFIT_AVAILABLE:
            logger.debug("SetFit not installed, classifier unavailable")
            return

        model_path = Path(self.config.model_path).expanduser()
        if not model_path.exists():
            logger.debug(f"SetFit model not found at {model_path}")
            return

        try:
            self._model = SetFitModel.from_pretrained(str(model_path))
            self._loaded = True
            self._model_mtime = self._get_model_mtime()
            logger.info(f"Loaded SetFit model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load SetFit model: {e}")

    def classify(self, query: str, context_summary: str = "") -> ComplexityClassification:
        """
        Classify query complexity.

        Args:
            query: User query to classify
            context_summary: Optional context summary

        Returns:
            ComplexityClassification with results
        """
        start_time = time.perf_counter()

        # Ensure model is loaded
        if not self._load_attempted:
            self._try_load()

        if not self._loaded:
            # Return fallback classification
            return self._fallback_classify(query, context_summary, start_time)

        try:
            # Prepare input
            input_text = self._prepare_input(query, context_summary)

            # Get prediction
            predictions = self._model.predict([input_text])
            probabilities = self._model.predict_proba([input_text])

            # Parse result
            predicted_label = predictions[0]
            confidence = float(max(probabilities[0]))

            complexity_level = self._label_to_complexity(predicted_label)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Determine activation based on complexity
            activate_rlm = LEVEL_ACTIVATION_MAP.get(complexity_level, True)

            # Infer signals from complexity level
            signals = self._infer_signals(complexity_level, query)

            return ComplexityClassification(
                activate_rlm=activate_rlm,
                complexity_level=complexity_level,
                signals=signals,
                confidence=confidence,
                latency_ms=latency_ms,
                model_used="setfit",
            )

        except Exception as e:
            logger.warning(f"SetFit classification failed: {e}")
            return self._fallback_classify(query, context_summary, start_time)

    def _prepare_input(self, query: str, context_summary: str) -> str:
        """Prepare input text for classification."""
        if context_summary:
            text = f"Query: {query}\nContext: {context_summary}"
        else:
            text = query

        # Truncate if needed
        if len(text) > self.config.max_length * 4:  # Rough char estimate
            text = text[: self.config.max_length * 4]

        return text

    def _label_to_complexity(self, label: str) -> ComplexityLevel:
        """Convert model label to ComplexityLevel."""
        label_lower = label.lower()
        for level in ComplexityLevel:
            if level.value == label_lower:
                return level
        # Default mapping for binary classifiers
        if label_lower in ("complex", "activate", "rlm", "yes", "1", "true"):
            return ComplexityLevel.COMPLEX
        return ComplexityLevel.SIMPLE

    def _infer_signals(self, complexity: ComplexityLevel, query: str) -> list[SignalType]:
        """Infer signals from complexity level and query keywords."""
        signals: list[SignalType] = []
        query_lower = query.lower()

        # Keyword-based signal detection
        if any(kw in query_lower for kw in ["find", "search", "where", "locate"]):
            signals.append("discovery_required")
        if any(kw in query_lower for kw in ["combine", "merge", "integrate", "across"]):
            signals.append("synthesis_required")
        if any(kw in query_lower for kw in ["debug", "fix", "error", "bug", "broken"]):
            signals.append("debugging_deep")
        if any(kw in query_lower for kw in ["architect", "design", "refactor", "structure"]):
            signals.append("architectural")
        if "?" in query and len(query) < 50:
            signals.append("conversational")
        if complexity == ComplexityLevel.TRIVIAL:
            signals.append("narrow_scope")

        return signals

    def _fallback_classify(
        self, query: str, context_summary: str, start_time: float
    ) -> ComplexityClassification:
        """
        Fallback classification using simple heuristics.

        Used when SetFit model is unavailable or fails.
        """
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Simple heuristic classification
        query_lower = query.lower()
        query_len = len(query)

        # Detect complexity signals
        signals: list[SignalType] = []
        complexity = ComplexityLevel.SIMPLE

        # Check for complexity indicators
        if any(kw in query_lower for kw in ["refactor", "redesign", "architect"]):
            complexity = ComplexityLevel.UNBOUNDED
            signals.append("architectural")
        elif any(kw in query_lower for kw in ["debug", "fix", "error", "broken", "failing"]):
            complexity = ComplexityLevel.COMPLEX
            signals.append("debugging_deep")
        elif any(kw in query_lower for kw in ["find all", "across", "every", "throughout"]):
            complexity = ComplexityLevel.COMPLEX
            signals.append("discovery_required")
        elif any(kw in query_lower for kw in ["combine", "integrate", "merge"]):
            complexity = ComplexityLevel.MODERATE
            signals.append("synthesis_required")
        elif query_len < 30 and "?" in query:
            complexity = ComplexityLevel.TRIVIAL
            signals.append("conversational")
        elif query_len > 200:
            complexity = ComplexityLevel.MODERATE
            signals.append("uncertainty_high")

        activate_rlm = LEVEL_ACTIVATION_MAP.get(complexity, True)

        return ComplexityClassification(
            activate_rlm=activate_rlm,
            complexity_level=complexity,
            signals=signals,
            confidence=0.5,  # Low confidence for heuristic fallback
            latency_ms=latency_ms,
            model_used="heuristic_fallback",
        )


def is_setfit_available() -> bool:
    """Check if SetFit is installed."""
    return _SETFIT_AVAILABLE


def get_classifier(config: SetFitConfig | None = None) -> SetFitClassifier:
    """Get a SetFit classifier instance."""
    return SetFitClassifier(config)


__all__ = [
    "ComplexityClassification",
    "ComplexityLevel",
    "SetFitClassifier",
    "SetFitConfig",
    "SignalType",
    "get_classifier",
    "is_setfit_available",
]
