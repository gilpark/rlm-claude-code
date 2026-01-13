"""
Learning system for strategy tracking and adaptation.

Implements: Spec §8.1 Phase 4 - Learning
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StrategyType(Enum):
    """Types of strategies that can be learned."""

    SUMMARIZATION = "summarization"
    SEARCH = "search"
    DECOMPOSITION = "decomposition"
    CACHING = "caching"
    MODEL_SELECTION = "model_selection"
    PROMPT_VARIANT = "prompt_variant"


class FeedbackType(Enum):
    """Types of user feedback."""

    POSITIVE = "positive"  # Explicit thumbs up
    NEGATIVE = "negative"  # Explicit thumbs down
    IMPLICIT_SUCCESS = "implicit_success"  # Task completed
    IMPLICIT_FAILURE = "implicit_failure"  # Task abandoned/retried
    NEUTRAL = "neutral"


@dataclass
class StrategyOutcome:
    """Outcome of a strategy application."""

    strategy_type: StrategyType
    strategy_id: str  # Specific variant/config
    success: bool
    latency_ms: float
    token_cost: int
    feedback: FeedbackType = FeedbackType.NEUTRAL
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def score(self) -> float:
        """Calculate composite score for this outcome."""
        base = 1.0 if self.success else 0.0

        # Feedback adjustments
        if self.feedback == FeedbackType.POSITIVE:
            base += 0.5
        elif self.feedback == FeedbackType.NEGATIVE:
            base -= 0.5
        elif self.feedback == FeedbackType.IMPLICIT_SUCCESS:
            base += 0.2
        elif self.feedback == FeedbackType.IMPLICIT_FAILURE:
            base -= 0.2

        # Efficiency bonus (faster and cheaper is better)
        if self.latency_ms < 1000:  # Under 1 second
            base += 0.1
        if self.token_cost < 1000:  # Low token usage
            base += 0.1

        return max(0.0, min(2.0, base))


@dataclass
class StrategyStats:
    """Statistics for a strategy."""

    strategy_type: StrategyType
    strategy_id: str
    total_uses: int = 0
    successes: int = 0
    total_score: float = 0.0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        """Success rate."""
        return self.successes / self.total_uses if self.total_uses > 0 else 0.0

    @property
    def avg_score(self) -> float:
        """Average score."""
        return self.total_score / self.total_uses if self.total_uses > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency."""
        return self.total_latency_ms / self.total_uses if self.total_uses > 0 else 0.0

    @property
    def avg_tokens(self) -> float:
        """Average token usage."""
        return self.total_tokens / self.total_uses if self.total_uses > 0 else 0.0

    def ucb_score(self, total_trials: int, exploration_factor: float = 2.0) -> float:
        """
        Upper Confidence Bound score for exploration/exploitation.

        Args:
            total_trials: Total trials across all strategies
            exploration_factor: How much to value exploration

        Returns:
            UCB score
        """
        if self.total_uses == 0:
            return float("inf")  # Always try unused strategies

        exploitation = self.avg_score
        exploration = exploration_factor * math.sqrt(math.log(total_trials + 1) / self.total_uses)

        return exploitation + exploration


class StrategyTracker:
    """
    Track strategy success and learn optimal selections.

    Implements: Spec §8.1 Strategy success tracking
    """

    def __init__(self, persistence_path: Path | None = None):
        """
        Initialize tracker.

        Args:
            persistence_path: Path to persist learning data
        """
        self.persistence_path = persistence_path
        self._stats: dict[str, StrategyStats] = {}
        self._outcomes: list[StrategyOutcome] = []
        self._total_trials = 0

        # Load persisted data
        if persistence_path and persistence_path.exists():
            self._load()

    def _key(self, strategy_type: StrategyType, strategy_id: str) -> str:
        """Create lookup key."""
        return f"{strategy_type.value}:{strategy_id}"

    def record_outcome(self, outcome: StrategyOutcome) -> None:
        """
        Record a strategy outcome.

        Args:
            outcome: The outcome to record
        """
        key = self._key(outcome.strategy_type, outcome.strategy_id)
        self._total_trials += 1

        # Update or create stats
        if key not in self._stats:
            self._stats[key] = StrategyStats(
                strategy_type=outcome.strategy_type,
                strategy_id=outcome.strategy_id,
            )

        stats = self._stats[key]
        stats.total_uses += 1
        stats.successes += 1 if outcome.success else 0
        stats.total_score += outcome.score
        stats.total_latency_ms += outcome.latency_ms
        stats.total_tokens += outcome.token_cost
        stats.last_used = outcome.timestamp

        # Store outcome
        self._outcomes.append(outcome)

        # Persist
        self._save()

    def get_stats(self, strategy_type: StrategyType, strategy_id: str) -> StrategyStats | None:
        """Get stats for a specific strategy."""
        return self._stats.get(self._key(strategy_type, strategy_id))

    def get_all_stats(self, strategy_type: StrategyType | None = None) -> list[StrategyStats]:
        """Get all stats, optionally filtered by type."""
        stats = list(self._stats.values())
        if strategy_type:
            stats = [s for s in stats if s.strategy_type == strategy_type]
        return sorted(stats, key=lambda s: s.avg_score, reverse=True)

    def select_strategy(
        self,
        strategy_type: StrategyType,
        available_ids: list[str],
        method: str = "ucb",
    ) -> str:
        """
        Select best strategy using learning.

        Implements: Spec §8.1 Prompt adaptation

        Args:
            strategy_type: Type of strategy to select
            available_ids: Available strategy IDs
            method: Selection method ("ucb", "greedy", "epsilon_greedy")

        Returns:
            Selected strategy ID
        """
        if not available_ids:
            raise ValueError("No available strategies")

        if method == "greedy":
            return self._select_greedy(strategy_type, available_ids)
        elif method == "epsilon_greedy":
            return self._select_epsilon_greedy(strategy_type, available_ids)
        else:  # ucb
            return self._select_ucb(strategy_type, available_ids)

    def _select_ucb(self, strategy_type: StrategyType, available_ids: list[str]) -> str:
        """Select using Upper Confidence Bound."""
        best_id = available_ids[0]
        best_score = float("-inf")

        for strategy_id in available_ids:
            stats = self.get_stats(strategy_type, strategy_id)
            if stats is None:
                # Never tried - highest priority
                return strategy_id

            score = stats.ucb_score(self._total_trials)
            if score > best_score:
                best_score = score
                best_id = strategy_id

        return best_id

    def _select_greedy(self, strategy_type: StrategyType, available_ids: list[str]) -> str:
        """Select best known strategy."""
        best_id = available_ids[0]
        best_score = float("-inf")

        for strategy_id in available_ids:
            stats = self.get_stats(strategy_type, strategy_id)
            if stats is None:
                continue

            if stats.avg_score > best_score:
                best_score = stats.avg_score
                best_id = strategy_id

        return best_id

    def _select_epsilon_greedy(
        self,
        strategy_type: StrategyType,
        available_ids: list[str],
        epsilon: float = 0.1,
    ) -> str:
        """Select with epsilon-greedy exploration."""
        import random

        if random.random() < epsilon:
            return random.choice(available_ids)

        return self._select_greedy(strategy_type, available_ids)

    def _save(self) -> None:
        """Persist learning data."""
        if not self.persistence_path:
            return

        data = {
            "total_trials": self._total_trials,
            "stats": {
                key: {
                    "strategy_type": s.strategy_type.value,
                    "strategy_id": s.strategy_id,
                    "total_uses": s.total_uses,
                    "successes": s.successes,
                    "total_score": s.total_score,
                    "total_latency_ms": s.total_latency_ms,
                    "total_tokens": s.total_tokens,
                    "last_used": s.last_used,
                }
                for key, s in self._stats.items()
            },
        }

        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self.persistence_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load persisted learning data."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            data = json.loads(self.persistence_path.read_text())
            self._total_trials = data.get("total_trials", 0)

            for key, s in data.get("stats", {}).items():
                self._stats[key] = StrategyStats(
                    strategy_type=StrategyType(s["strategy_type"]),
                    strategy_id=s["strategy_id"],
                    total_uses=s["total_uses"],
                    successes=s["successes"],
                    total_score=s["total_score"],
                    total_latency_ms=s["total_latency_ms"],
                    total_tokens=s["total_tokens"],
                    last_used=s["last_used"],
                )
        except (json.JSONDecodeError, KeyError):
            # Invalid data, start fresh
            pass

    def reset(self) -> None:
        """Reset all learning data."""
        self._stats.clear()
        self._outcomes.clear()
        self._total_trials = 0

        if self.persistence_path and self.persistence_path.exists():
            self.persistence_path.unlink()


class UserFeedbackCollector:
    """
    Collect and process user feedback for learning.

    Implements: Spec §8.1 User feedback integration
    """

    def __init__(self, tracker: StrategyTracker):
        """
        Initialize collector.

        Args:
            tracker: Strategy tracker to update
        """
        self.tracker = tracker
        self._pending_feedback: dict[str, StrategyOutcome] = {}

    def register_for_feedback(self, outcome: StrategyOutcome, feedback_id: str) -> None:
        """
        Register an outcome that may receive feedback.

        Args:
            outcome: The outcome to track
            feedback_id: ID for feedback correlation
        """
        self._pending_feedback[feedback_id] = outcome

    def process_feedback(self, feedback_id: str, feedback: FeedbackType) -> bool:
        """
        Process user feedback for a pending outcome.

        Args:
            feedback_id: ID of the outcome
            feedback: User's feedback

        Returns:
            True if feedback was processed
        """
        if feedback_id not in self._pending_feedback:
            return False

        outcome = self._pending_feedback.pop(feedback_id)
        outcome.feedback = feedback

        # Re-record with feedback
        self.tracker.record_outcome(outcome)

        return True

    def infer_feedback(
        self,
        feedback_id: str,
        task_completed: bool,
        retry_count: int = 0,
    ) -> None:
        """
        Infer feedback from user behavior.

        Args:
            feedback_id: ID of the outcome
            task_completed: Whether task was completed
            retry_count: Number of retries
        """
        if feedback_id not in self._pending_feedback:
            return

        outcome = self._pending_feedback.pop(feedback_id)

        if task_completed and retry_count == 0:
            outcome.feedback = FeedbackType.IMPLICIT_SUCCESS
        elif not task_completed or retry_count > 2:
            outcome.feedback = FeedbackType.IMPLICIT_FAILURE
        else:
            outcome.feedback = FeedbackType.NEUTRAL

        self.tracker.record_outcome(outcome)

    def expire_pending(self, max_age_s: float = 3600.0) -> int:
        """
        Expire old pending feedback entries.

        Args:
            max_age_s: Maximum age in seconds

        Returns:
            Number of expired entries
        """
        now = time.time()
        expired = []

        for feedback_id, outcome in self._pending_feedback.items():
            if now - outcome.timestamp > max_age_s:
                expired.append(feedback_id)
                outcome.feedback = FeedbackType.NEUTRAL
                self.tracker.record_outcome(outcome)

        for feedback_id in expired:
            del self._pending_feedback[feedback_id]

        return len(expired)


class AdaptiveStrategy:
    """
    Adaptive strategy that learns from outcomes.

    Provides a high-level interface for strategy learning.
    """

    def __init__(
        self,
        tracker: StrategyTracker,
        strategy_type: StrategyType,
        strategy_configs: dict[str, dict[str, Any]],
    ):
        """
        Initialize adaptive strategy.

        Args:
            tracker: Strategy tracker
            strategy_type: Type of this strategy
            strategy_configs: Map of strategy_id -> config
        """
        self.tracker = tracker
        self.strategy_type = strategy_type
        self.configs = strategy_configs
        self._current_id: str | None = None
        self._current_start: float | None = None

    @property
    def available_strategies(self) -> list[str]:
        """Get available strategy IDs."""
        return list(self.configs.keys())

    def select(self, method: str = "ucb") -> tuple[str, dict[str, Any]]:
        """
        Select a strategy to use.

        Args:
            method: Selection method

        Returns:
            (strategy_id, config) tuple
        """
        strategy_id = self.tracker.select_strategy(
            self.strategy_type,
            self.available_strategies,
            method=method,
        )

        self._current_id = strategy_id
        self._current_start = time.time()

        return strategy_id, self.configs[strategy_id]

    def record_result(
        self,
        success: bool,
        token_cost: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Record the result of using current strategy.

        Args:
            success: Whether strategy succeeded
            token_cost: Tokens used
            context: Optional context
        """
        if self._current_id is None or self._current_start is None:
            return

        latency_ms = (time.time() - self._current_start) * 1000

        outcome = StrategyOutcome(
            strategy_type=self.strategy_type,
            strategy_id=self._current_id,
            success=success,
            latency_ms=latency_ms,
            token_cost=token_cost,
            context=context or {},
        )

        self.tracker.record_outcome(outcome)

        self._current_id = None
        self._current_start = None

    def get_best_strategy(self) -> tuple[str, dict[str, Any]] | None:
        """Get the currently best-performing strategy."""
        stats = self.tracker.get_all_stats(self.strategy_type)
        if not stats:
            return None

        best = stats[0]
        return best.strategy_id, self.configs.get(best.strategy_id, {})


class LearningSystem:
    """
    Main learning system coordinating all learning components.

    Implements: Spec §8.1 Phase 4 - Learning
    """

    def __init__(self, persistence_dir: Path | None = None):
        """
        Initialize learning system.

        Args:
            persistence_dir: Directory for persisting learning data
        """
        self.persistence_dir = persistence_dir

        # Create tracker with persistence
        tracker_path = persistence_dir / "strategy_stats.json" if persistence_dir else None
        self.tracker = StrategyTracker(tracker_path)

        # Feedback collector
        self.feedback = UserFeedbackCollector(self.tracker)

        # Adaptive strategies
        self._strategies: dict[StrategyType, AdaptiveStrategy] = {}

    def register_strategy(
        self,
        strategy_type: StrategyType,
        configs: dict[str, dict[str, Any]],
    ) -> AdaptiveStrategy:
        """
        Register an adaptive strategy.

        Args:
            strategy_type: Type of strategy
            configs: Available configurations

        Returns:
            AdaptiveStrategy instance
        """
        strategy = AdaptiveStrategy(self.tracker, strategy_type, configs)
        self._strategies[strategy_type] = strategy
        return strategy

    def get_strategy(self, strategy_type: StrategyType) -> AdaptiveStrategy | None:
        """Get registered strategy."""
        return self._strategies.get(strategy_type)

    def get_summary(self) -> dict[str, Any]:
        """Get learning system summary."""
        return {
            "total_trials": self.tracker._total_trials,
            "strategies_tracked": len(self.tracker._stats),
            "by_type": {
                st.value: [
                    {
                        "id": s.strategy_id,
                        "uses": s.total_uses,
                        "success_rate": s.success_rate,
                        "avg_score": s.avg_score,
                    }
                    for s in self.tracker.get_all_stats(st)
                ]
                for st in StrategyType
                if self.tracker.get_all_stats(st)
            },
        }

    def reset(self) -> None:
        """Reset all learning data."""
        self.tracker.reset()
        self._strategies.clear()


__all__ = [
    "AdaptiveStrategy",
    "FeedbackType",
    "LearningSystem",
    "StrategyOutcome",
    "StrategyStats",
    "StrategyTracker",
    "StrategyType",
    "UserFeedbackCollector",
]
