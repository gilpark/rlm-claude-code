"""
Orchestration decision logging for training data collection.

Logs orchestration decisions to JSONL format for future fine-tuning
of a local classifier model (e.g., ModernBERT).

Usage:
    from src.orchestration_logger import OrchestrationLogger

    logger = OrchestrationLogger(log_path="~/.rlm/training_data.jsonl")
    logger.log_decision(query, context_summary, decision, source="api")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationDecisionLog:
    """
    A logged orchestration decision for training data.

    Contains all information needed to train a classifier:
    - Input features (query, context)
    - Output labels (activation, mode, tier, depth, etc.)
    - Metadata (source, timestamp, latency)
    """

    # === Input Features ===
    query: str
    query_length: int
    context_tokens: int
    context_summary: str

    # === Output Labels ===
    activate_rlm: bool
    activation_reason: str
    execution_mode: str  # fast, balanced, thorough
    model_tier: str  # fast, balanced, powerful, code_specialist
    depth_budget: int  # 0-3
    tool_access: str  # none, repl_only, read_only, full
    query_type: str
    complexity_score: float
    confidence: float
    signals: list[str]

    # === Metadata ===
    timestamp: str
    source: str  # "api", "local", "heuristic"
    latency_ms: float
    session_id: str = ""
    model_used: str = ""

    # === Optional Feedback ===
    # These can be filled in later based on outcome
    outcome_success: bool | None = None
    outcome_feedback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrchestrationDecisionLog:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class LoggerConfig:
    """Configuration for orchestration logger."""

    # Log file path (supports ~ expansion)
    log_path: str = "~/.rlm/orchestration_decisions.jsonl"

    # Whether logging is enabled
    enabled: bool = True

    # Maximum log file size in MB before rotation
    max_size_mb: float = 100.0

    # Number of rotated files to keep
    max_files: int = 5

    # Whether to log heuristic decisions (can be noisy)
    log_heuristics: bool = True

    # Whether to log cached decisions
    log_cache_hits: bool = False

    # Minimum confidence threshold to log (skip very confident decisions)
    min_confidence_to_log: float = 0.0

    # Session ID for grouping related decisions
    session_id: str = ""


class OrchestrationLogger:
    """
    Logs orchestration decisions for training data collection.

    Writes decisions to a JSONL file that can be used to:
    1. Analyze decision patterns
    2. Fine-tune a local classifier
    3. Debug routing issues
    """

    def __init__(self, config: LoggerConfig | None = None):
        self.config = config or LoggerConfig()
        self._log_path: Path | None = None
        self._decision_count = 0
        self._session_start = datetime.now().isoformat()

        if self.config.enabled:
            self._ensure_log_path()

    def _ensure_log_path(self) -> None:
        """Ensure log directory exists."""
        path = Path(self.config.log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = path

    def _check_rotation(self) -> None:
        """Check if log file needs rotation."""
        if self._log_path is None or not self._log_path.exists():
            return

        size_mb = self._log_path.stat().st_size / (1024 * 1024)
        if size_mb >= self.config.max_size_mb:
            self._rotate_logs()

    def _rotate_logs(self) -> None:
        """Rotate log files."""
        if self._log_path is None:
            return

        # Shift existing rotated files
        for i in range(self.config.max_files - 1, 0, -1):
            old_path = self._log_path.with_suffix(f".jsonl.{i}")
            new_path = self._log_path.with_suffix(f".jsonl.{i + 1}")
            if old_path.exists():
                if i + 1 >= self.config.max_files:
                    old_path.unlink()  # Delete oldest
                else:
                    old_path.rename(new_path)

        # Rotate current file
        if self._log_path.exists():
            self._log_path.rename(self._log_path.with_suffix(".jsonl.1"))

        logger.info(f"Rotated log file: {self._log_path}")

    def log_decision(
        self,
        query: str,
        context_summary: str,
        decision: dict[str, Any],
        source: str,
        latency_ms: float = 0.0,
        context_tokens: int = 0,
        model_used: str = "",
    ) -> None:
        """
        Log an orchestration decision.

        Args:
            query: The user query
            context_summary: Summary of context provided to orchestrator
            decision: The orchestration decision dict
            source: Decision source ("api", "local", "heuristic")
            latency_ms: Time taken to make decision
            context_tokens: Number of context tokens
            model_used: Model that made the decision (if applicable)
        """
        if not self.config.enabled:
            return

        # Skip heuristics if configured
        if source == "heuristic" and not self.config.log_heuristics:
            return

        # Skip cache hits if configured
        if source == "cache" and not self.config.log_cache_hits:
            return

        # Skip high-confidence decisions if threshold set
        confidence = decision.get("confidence", 1.0)
        if confidence > 1.0 - self.config.min_confidence_to_log:
            return

        try:
            log_entry = OrchestrationDecisionLog(
                # Input features
                query=query,
                query_length=len(query),
                context_tokens=context_tokens,
                context_summary=context_summary[:500],  # Truncate

                # Output labels
                activate_rlm=decision.get("activate_rlm", False),
                activation_reason=decision.get("activation_reason", ""),
                execution_mode=decision.get("execution_mode", "balanced"),
                model_tier=decision.get("model_tier", "balanced"),
                depth_budget=decision.get("depth_budget", 0),
                tool_access=decision.get("tool_access", "none"),
                query_type=decision.get("query_type", "unknown"),
                complexity_score=decision.get("complexity_score", 0.5),
                confidence=confidence,
                signals=decision.get("signals", []),

                # Metadata
                timestamp=datetime.now().isoformat(),
                source=source,
                latency_ms=latency_ms,
                session_id=self.config.session_id or self._session_start,
                model_used=model_used,
            )

            self._write_entry(log_entry)
            self._decision_count += 1

        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")

    def _write_entry(self, entry: OrchestrationDecisionLog) -> None:
        """Write entry to log file."""
        if self._log_path is None:
            return

        self._check_rotation()

        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def log_outcome(
        self,
        query: str,
        success: bool,
        feedback: str = "",
    ) -> None:
        """
        Log outcome feedback for a previous decision.

        This can be used to record whether a decision led to good results,
        which is valuable for training.

        Args:
            query: The original query (used to match)
            success: Whether the outcome was successful
            feedback: Optional feedback text
        """
        if not self.config.enabled or self._log_path is None:
            return

        # Write feedback as a separate entry that can be joined later
        feedback_entry = {
            "type": "outcome_feedback",
            "query_prefix": query[:100],
            "success": success,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.config.session_id or self._session_start,
        }

        with open(self._log_path, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")

    def get_statistics(self) -> dict[str, Any]:
        """Get logging statistics."""
        stats = {
            "enabled": self.config.enabled,
            "log_path": str(self._log_path) if self._log_path else None,
            "decisions_logged": self._decision_count,
            "session_id": self.config.session_id or self._session_start,
        }

        if self._log_path and self._log_path.exists():
            stats["log_size_mb"] = self._log_path.stat().st_size / (1024 * 1024)
            stats["log_lines"] = sum(1 for _ in open(self._log_path))

        return stats


class TrainingDataExporter:
    """
    Export logged decisions to training data formats.

    Supports:
    - CSV for pandas/sklearn
    - HuggingFace datasets format
    - Feature extraction for classifiers
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path).expanduser()

    def load_decisions(self) -> list[OrchestrationDecisionLog]:
        """Load all decisions from log file."""
        decisions = []

        if not self.log_path.exists():
            return decisions

        with open(self.log_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Skip feedback entries
                    if data.get("type") == "outcome_feedback":
                        continue
                    decisions.append(OrchestrationDecisionLog.from_dict(data))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Skipping malformed line: {e}")

        return decisions

    def export_csv(self, output_path: str) -> int:
        """
        Export to CSV format.

        Returns number of rows exported.
        """
        decisions = self.load_decisions()
        if not decisions:
            return 0

        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)

        # Define columns
        columns = [
            "query", "query_length", "context_tokens",
            "activate_rlm", "activation_reason", "execution_mode",
            "model_tier", "depth_budget", "tool_access", "query_type",
            "complexity_score", "confidence", "source", "latency_ms",
        ]

        with open(output, "w") as f:
            # Header
            f.write(",".join(columns) + "\n")

            # Data
            for d in decisions:
                row = [
                    f'"{d.query[:200]}"',  # Truncate and quote
                    str(d.query_length),
                    str(d.context_tokens),
                    str(d.activate_rlm).lower(),
                    f'"{d.activation_reason}"',
                    d.execution_mode,
                    d.model_tier,
                    str(d.depth_budget),
                    d.tool_access,
                    d.query_type,
                    f"{d.complexity_score:.3f}",
                    f"{d.confidence:.3f}",
                    d.source,
                    f"{d.latency_ms:.2f}",
                ]
                f.write(",".join(row) + "\n")

        return len(decisions)

    def export_huggingface(self, output_dir: str) -> int:
        """
        Export to HuggingFace datasets format (JSON lines with train/test split).

        Returns number of examples exported.
        """
        decisions = self.load_decisions()
        if not decisions:
            return 0

        output = Path(output_dir).expanduser()
        output.mkdir(parents=True, exist_ok=True)

        # Convert to HF format
        examples = []
        for d in decisions:
            example = {
                "text": d.query,
                "label": 1 if d.activate_rlm else 0,
                "labels": {
                    "activate_rlm": d.activate_rlm,
                    "execution_mode": d.execution_mode,
                    "model_tier": d.model_tier,
                    "depth_budget": d.depth_budget,
                    "query_type": d.query_type,
                },
                "metadata": {
                    "context_tokens": d.context_tokens,
                    "source": d.source,
                    "confidence": d.confidence,
                    "signals": d.signals,
                },
            }
            examples.append(example)

        # Split 80/20
        split_idx = int(len(examples) * 0.8)
        train = examples[:split_idx]
        test = examples[split_idx:]

        # Write files
        with open(output / "train.jsonl", "w") as f:
            for ex in train:
                f.write(json.dumps(ex) + "\n")

        with open(output / "test.jsonl", "w") as f:
            for ex in test:
                f.write(json.dumps(ex) + "\n")

        # Write dataset info
        info = {
            "description": "RLM orchestration decisions for classifier training",
            "features": {
                "text": "string",
                "label": "int (0=no RLM, 1=RLM)",
                "labels": "dict of multi-label targets",
                "metadata": "dict of additional features",
            },
            "splits": {
                "train": len(train),
                "test": len(test),
            },
            "total_examples": len(examples),
        }
        with open(output / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        return len(examples)

    def get_label_distribution(self) -> dict[str, dict[str, int]]:
        """Get distribution of labels for analysis."""
        decisions = self.load_decisions()

        distribution: dict[str, dict[str, int]] = {
            "activate_rlm": {"true": 0, "false": 0},
            "execution_mode": {},
            "model_tier": {},
            "query_type": {},
            "source": {},
        }

        for d in decisions:
            # Binary
            key = "true" if d.activate_rlm else "false"
            distribution["activate_rlm"][key] += 1

            # Categorical
            for field in ["execution_mode", "model_tier", "query_type", "source"]:
                value = getattr(d, field)
                if value not in distribution[field]:
                    distribution[field][value] = 0
                distribution[field][value] += 1

        return distribution


# Global logger instance (can be configured via set_logger)
_global_logger: OrchestrationLogger | None = None


def get_logger() -> OrchestrationLogger:
    """Get the global orchestration logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = OrchestrationLogger()
    return _global_logger


def set_logger(logger: OrchestrationLogger) -> None:
    """Set the global orchestration logger."""
    global _global_logger
    _global_logger = logger


def log_decision(
    query: str,
    context_summary: str,
    decision: dict[str, Any],
    source: str,
    **kwargs: Any,
) -> None:
    """Convenience function to log a decision using the global logger."""
    get_logger().log_decision(query, context_summary, decision, source, **kwargs)


__all__ = [
    "LoggerConfig",
    "OrchestrationDecisionLog",
    "OrchestrationLogger",
    "TrainingDataExporter",
    "get_logger",
    "log_decision",
    "set_logger",
]
