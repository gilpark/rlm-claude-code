"""
GLiNER2-based signal extraction for complexity detection.

GLiNER2 is a zero-shot NER model that can extract entities/signals from
text without fine-tuning. This module uses GLiNER2 to extract complexity
signals from queries to inform RLM activation decisions.

Implements: Plan §3.3 GLiNER2 Signal Extraction
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check if GLiNER is available
try:
    from gliner import GLiNER

    _GLINER_AVAILABLE = True
except ImportError:
    GLiNER = None  # type: ignore[assignment, misc]
    _GLINER_AVAILABLE = False


# Signal schema mapping entity types to related keywords
# GLiNER will be prompted to extract these entity types
SIGNAL_SCHEMA: dict[str, list[str]] = {
    "discovery": [
        "exploration",
        "understanding",
        "finding",
        "searching",
        "locating",
        "investigating",
    ],
    "synthesis": [
        "combine",
        "combining",
        "integrate",
        "integrating",
        "merge",
        "merging",
        "consolidate",
        "consolidating",
        "unify",
        "unifying",
        "aggregate",
        "aggregating",
    ],
    "debugging": [
        "error",
        "bug",
        "fix",
        "issue",
        "broken",
        "failing",
        "crash",
        "exception",
    ],
    "architectural": [
        "design",
        "architecture",
        "structure",
        "refactor",
        "pattern",
        "system",
    ],
    "uncertainty": [
        "unclear",
        "confused",
        "unsure",
        "maybe",
        "possibly",
        "might",
    ],
    "multi_file": [
        "across",
        "multiple",
        "several",
        "throughout",
        "everywhere",
        "all files",
    ],
}

# Signal weights for complexity scoring
SIGNAL_WEIGHTS: dict[str, float] = {
    "discovery": 0.6,
    "synthesis": 0.7,
    "debugging": 0.5,
    "architectural": 0.9,
    "uncertainty": 0.4,
    "multi_file": 0.8,
}


@dataclass
class ExtractedSignal:
    """A single extracted signal from the query."""

    signal_type: str
    text: str  # The matched text span
    score: float  # Confidence score 0-1
    start: int  # Character start position
    end: int  # Character end position


@dataclass
class SignalExtractionResult:
    """Result from signal extraction."""

    signals: list[ExtractedSignal] = field(default_factory=list)
    signal_types: list[str] = field(default_factory=list)  # Unique signal types
    complexity_score: float = 0.0  # Weighted complexity score
    latency_ms: float = 0.0
    model_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signals": [
                {
                    "type": s.signal_type,
                    "text": s.text,
                    "score": s.score,
                }
                for s in self.signals
            ],
            "signal_types": self.signal_types,
            "complexity_score": self.complexity_score,
            "latency_ms": self.latency_ms,
            "model_used": self.model_used,
        }


@dataclass
class GLiNERConfig:
    """Configuration for GLiNER extractor."""

    # Model selection
    model_name: str = "urchade/gliner_multi_pii-v1"  # Good general model
    model_path: str | None = None  # Optional local path

    # Extraction parameters
    confidence_threshold: float = 0.5  # Min score to accept entity
    flat_ner: bool = True  # Use flat NER (no nested entities)

    # Performance
    max_length: int = 512

    # Fallback
    fallback_on_error: bool = True


class GLiNERExtractor:
    """
    GLiNER-based signal extractor for complexity detection.

    Extracts complexity signals from queries using zero-shot NER.
    Falls back to keyword matching if GLiNER unavailable.

    Implements: Plan §3.3 GLiNER2 Signal Extraction
    """

    def __init__(self, config: GLiNERConfig | None = None):
        """
        Initialize GLiNER extractor.

        Args:
            config: Extractor configuration
        """
        self.config = config or GLiNERConfig()
        self._model: Any = None
        self._loaded = False
        self._load_attempted = False
        self._model_mtime: float | None = None  # Track model file modification time

    @property
    def is_available(self) -> bool:
        """Check if GLiNER is available and model is loadable."""
        if not _GLINER_AVAILABLE:
            return False
        if not self._load_attempted:
            self._try_load()
        return self._loaded

    def _get_model_mtime(self) -> float | None:
        """Get model file modification time for hot-swap detection."""
        # Only track local model paths
        if not self.config.model_path:
            return None
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

        if not _GLINER_AVAILABLE:
            logger.debug("GLiNER not installed, extractor unavailable")
            return

        try:
            # Use local path if specified, otherwise download from HF
            model_id = self.config.model_path or self.config.model_name
            self._model = GLiNER.from_pretrained(model_id)
            self._loaded = True
            self._model_mtime = self._get_model_mtime()
            logger.info(f"Loaded GLiNER model: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to load GLiNER model: {e}")

    def extract_signals(self, query: str) -> SignalExtractionResult:
        """
        Extract complexity signals from query.

        Args:
            query: User query to analyze

        Returns:
            SignalExtractionResult with extracted signals
        """
        start_time = time.perf_counter()

        # Ensure model is loaded
        if not self._load_attempted:
            self._try_load()

        if not self._loaded:
            return self._fallback_extract(query, start_time)

        try:
            # Truncate if needed
            text = (
                query[: self.config.max_length * 4]
                if len(query) > self.config.max_length * 4
                else query
            )

            # Extract entities using GLiNER
            labels = list(SIGNAL_SCHEMA.keys())
            entities = self._model.predict_entities(
                text,
                labels,
                flat_ner=self.config.flat_ner,
                threshold=self.config.confidence_threshold,
            )

            # Convert to ExtractedSignal objects
            signals: list[ExtractedSignal] = []
            for entity in entities:
                signals.append(
                    ExtractedSignal(
                        signal_type=entity["label"],
                        text=entity["text"],
                        score=entity["score"],
                        start=entity.get("start", 0),
                        end=entity.get("end", len(entity["text"])),
                    )
                )

            # Get unique signal types
            signal_types = list({s.signal_type for s in signals})

            # Compute complexity score
            complexity_score = self._compute_complexity_score(signals)

            latency_ms = (time.perf_counter() - start_time) * 1000

            return SignalExtractionResult(
                signals=signals,
                signal_types=signal_types,
                complexity_score=complexity_score,
                latency_ms=latency_ms,
                model_used="gliner",
            )

        except Exception as e:
            logger.warning(f"GLiNER extraction failed: {e}")
            return self._fallback_extract(query, start_time)

    def _compute_complexity_score(self, signals: list[ExtractedSignal]) -> float:
        """
        Compute weighted complexity score from signals.

        Returns:
            Score between 0.0 (trivial) and 1.0 (very complex)
        """
        if not signals:
            return 0.0

        # Weighted sum of signal scores
        weighted_sum = 0.0
        max_possible = 0.0

        # Group by signal type and take max score per type
        type_scores: dict[str, float] = {}
        for signal in signals:
            if signal.signal_type not in type_scores:
                type_scores[signal.signal_type] = signal.score
            else:
                type_scores[signal.signal_type] = max(type_scores[signal.signal_type], signal.score)

        for signal_type, score in type_scores.items():
            weight = SIGNAL_WEIGHTS.get(signal_type, 0.5)
            weighted_sum += score * weight
            max_possible += weight

        if max_possible == 0:
            return 0.0

        # Normalize to 0-1
        return min(1.0, weighted_sum / max_possible)

    def _fallback_extract(self, query: str, start_time: float) -> SignalExtractionResult:
        """
        Fallback extraction using keyword matching.

        Used when GLiNER is unavailable or fails.
        """
        query_lower = query.lower()
        signals: list[ExtractedSignal] = []

        # Keyword-based signal detection
        for signal_type, keywords in SIGNAL_SCHEMA.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Find position
                    start = query_lower.find(keyword)
                    signals.append(
                        ExtractedSignal(
                            signal_type=signal_type,
                            text=keyword,
                            score=0.6,  # Moderate confidence for keyword match
                            start=start,
                            end=start + len(keyword),
                        )
                    )
                    break  # One match per signal type is enough

        signal_types = list({s.signal_type for s in signals})
        complexity_score = self._compute_complexity_score(signals)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return SignalExtractionResult(
            signals=signals,
            signal_types=signal_types,
            complexity_score=complexity_score,
            latency_ms=latency_ms,
            model_used="keyword_fallback",
        )

    def get_weighted_rules_result(self, signals: SignalExtractionResult) -> dict[str, Any]:
        """
        Apply weighted rules to signals to produce activation decision.

        This combines extracted signals with predefined rules to produce
        a recommendation for RLM activation.

        Args:
            signals: Extracted signals from query

        Returns:
            Dict with activate_rlm recommendation and reasoning
        """
        # Thresholds for activation
        COMPLEXITY_THRESHOLD = 0.5
        HIGH_WEIGHT_SIGNALS = {"architectural", "synthesis", "multi_file"}

        # Check for high-weight signals
        has_high_weight = any(s in signals.signal_types for s in HIGH_WEIGHT_SIGNALS)

        # Check complexity score
        high_complexity = signals.complexity_score >= COMPLEXITY_THRESHOLD

        # Determine activation
        activate_rlm = has_high_weight or high_complexity

        # Build reasoning
        reasons = []
        if has_high_weight:
            matching = [s for s in signals.signal_types if s in HIGH_WEIGHT_SIGNALS]
            reasons.append(f"high-weight signals: {matching}")
        if high_complexity:
            reasons.append(
                f"complexity score {signals.complexity_score:.2f} >= {COMPLEXITY_THRESHOLD}"
            )

        return {
            "activate_rlm": activate_rlm,
            "confidence": min(0.8, signals.complexity_score + 0.3) if activate_rlm else 0.7,
            "reason": "; ".join(reasons) if reasons else "no complexity signals detected",
            "signals": signals.signal_types,
            "complexity_score": signals.complexity_score,
        }


def is_gliner_available() -> bool:
    """Check if GLiNER is installed."""
    return _GLINER_AVAILABLE


def get_extractor(config: GLiNERConfig | None = None) -> GLiNERExtractor:
    """Get a GLiNER extractor instance."""
    return GLiNERExtractor(config)


__all__ = [
    "ExtractedSignal",
    "GLiNERConfig",
    "GLiNERExtractor",
    "SIGNAL_SCHEMA",
    "SIGNAL_WEIGHTS",
    "SignalExtractionResult",
    "get_extractor",
    "is_gliner_available",
]
