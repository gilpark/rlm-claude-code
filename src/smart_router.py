"""
Smart routing for query-based model selection.

Implements: Spec §8.1 Phase 4 - Smart Routing
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(Enum):
    """Types of queries for routing decisions."""

    FACTUAL = "factual"  # Simple fact lookup
    ANALYTICAL = "analytical"  # Requires reasoning
    CREATIVE = "creative"  # Open-ended generation
    CODE = "code"  # Code generation/analysis
    SEARCH = "search"  # Information retrieval
    SUMMARIZATION = "summarization"  # Condensing information
    PLANNING = "planning"  # Multi-step task planning
    DEBUGGING = "debugging"  # Error analysis
    UNKNOWN = "unknown"


class ModelTier(Enum):
    """Model tiers by capability and cost."""

    FAST = "fast"  # Haiku - quick, cheap
    BALANCED = "balanced"  # Sonnet - good balance
    POWERFUL = "powerful"  # Opus - highest capability


# Model mappings
MODEL_BY_TIER: dict[ModelTier, str] = {
    ModelTier.FAST: "claude-haiku-4-5-20251001",
    ModelTier.BALANCED: "claude-sonnet-4-20250514",
    ModelTier.POWERFUL: "claude-opus-4-5-20251101",
}

# Default routing: query type -> model tier
DEFAULT_ROUTING: dict[QueryType, ModelTier] = {
    QueryType.FACTUAL: ModelTier.FAST,
    QueryType.ANALYTICAL: ModelTier.BALANCED,
    QueryType.CREATIVE: ModelTier.BALANCED,
    QueryType.CODE: ModelTier.BALANCED,
    QueryType.SEARCH: ModelTier.FAST,
    QueryType.SUMMARIZATION: ModelTier.FAST,
    QueryType.PLANNING: ModelTier.POWERFUL,
    QueryType.DEBUGGING: ModelTier.BALANCED,
    QueryType.UNKNOWN: ModelTier.BALANCED,
}


@dataclass
class QueryClassification:
    """Result of query type classification."""

    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    signals: list[str]  # What triggered this classification
    suggested_tier: ModelTier

    @property
    def model(self) -> str:
        """Get model name for this classification."""
        return MODEL_BY_TIER[self.suggested_tier]


@dataclass
class RoutingDecision:
    """Final routing decision with fallback chain."""

    primary_model: str
    fallback_chain: list[str]
    query_type: QueryType
    confidence: float
    reason: str

    @property
    def all_models(self) -> list[str]:
        """All models in order of preference."""
        return [self.primary_model, *self.fallback_chain]


@dataclass
class FallbackChain:
    """Fallback chain configuration."""

    models: list[str]
    retry_on_errors: list[type[Exception]] = field(
        default_factory=lambda: [TimeoutError, ConnectionError]
    )
    max_retries: int = 2


# Default fallback chains by tier
DEFAULT_FALLBACK_CHAINS: dict[ModelTier, FallbackChain] = {
    ModelTier.FAST: FallbackChain(
        models=[
            MODEL_BY_TIER[ModelTier.FAST],
            MODEL_BY_TIER[ModelTier.BALANCED],
        ]
    ),
    ModelTier.BALANCED: FallbackChain(
        models=[
            MODEL_BY_TIER[ModelTier.BALANCED],
            MODEL_BY_TIER[ModelTier.POWERFUL],
            MODEL_BY_TIER[ModelTier.FAST],
        ]
    ),
    ModelTier.POWERFUL: FallbackChain(
        models=[
            MODEL_BY_TIER[ModelTier.POWERFUL],
            MODEL_BY_TIER[ModelTier.BALANCED],
        ]
    ),
}


class QueryClassifier:
    """
    Classify queries by type for routing decisions.

    Implements: Spec §8.1 Query type detection
    """

    # Pattern sets for each query type
    PATTERNS: dict[QueryType, list[str]] = {
        QueryType.FACTUAL: [
            r"\bwhat is\b",
            r"\bwho is\b",
            r"\bwhen did\b",
            r"\bwhere is\b",
            r"\bdefine\b",
            r"\bexplain\b",
            r"\bhow many\b",
        ],
        QueryType.ANALYTICAL: [
            r"\bwhy\b",
            r"\banalyze\b",
            r"\bcompare\b",
            r"\bevaluate\b",
            r"\bassess\b",
            r"\bcritique\b",
            r"\bimplications\b",
            r"\bconsequences\b",
        ],
        QueryType.CREATIVE: [
            r"\bwrite\b.*\b(story|poem|essay)\b",
            r"\bcreate\b",
            r"\bimagine\b",
            r"\binvent\b",
            r"\bbrainstorm\b",
            r"\bgenerate ideas\b",
        ],
        QueryType.CODE: [
            r"\bcode\b",
            r"\bfunction\b",
            r"\bimplement\b",
            r"\bprogram\b",
            r"\bscript\b",
            r"\bclass\b",
            r"\bmethod\b",
            r"\bapi\b",
            r"\bbug\b",
            r"\berror\b",
            r"\brefactor\b",
        ],
        QueryType.SEARCH: [
            r"\bfind\b",
            r"\bsearch\b",
            r"\blook for\b",
            r"\blocate\b",
            r"\bwhere.*\bfile\b",
            r"\bgrep\b",
        ],
        QueryType.SUMMARIZATION: [
            r"\bsummarize\b",
            r"\bsummary\b",
            r"\btl;?dr\b",
            r"\bcondense\b",
            r"\boverview\b",
            r"\bbrief\b",
        ],
        QueryType.PLANNING: [
            r"\bplan\b",
            r"\bstrategy\b",
            r"\bsteps\b",
            r"\bhow (should|would|can) (i|we)\b",
            r"\bdesign\b",
            r"\barchitect\b",
            r"\broadmap\b",
        ],
        QueryType.DEBUGGING: [
            r"\bdebug\b",
            r"\bfix\b.*\b(bug|error|issue)\b",
            r"\bwhy.*\b(fail|error|crash)\b",
            r"\btroubleshoot\b",
            r"\bdiagnose\b",
            r"\bnot working\b",
        ],
    }

    def __init__(self, custom_patterns: dict[QueryType, list[str]] | None = None):
        """
        Initialize classifier with optional custom patterns.

        Args:
            custom_patterns: Additional patterns to match
        """
        self.patterns = dict(self.PATTERNS)
        if custom_patterns:
            for query_type, patterns in custom_patterns.items():
                if query_type in self.patterns:
                    self.patterns[query_type].extend(patterns)
                else:
                    self.patterns[query_type] = patterns

        # Compile patterns for efficiency
        self._compiled: dict[QueryType, list[re.Pattern[str]]] = {
            qt: [re.compile(p, re.IGNORECASE) for p in patterns]
            for qt, patterns in self.patterns.items()
        }

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a query by type.

        Args:
            query: The query string to classify

        Returns:
            QueryClassification with type, confidence, and signals
        """
        scores: dict[QueryType, list[str]] = {qt: [] for qt in QueryType}

        # Match patterns
        for query_type, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[query_type].append(pattern.pattern)

        # Find best match
        best_type = QueryType.UNKNOWN
        best_signals: list[str] = []
        max_matches = 0

        for query_type, signals in scores.items():
            if len(signals) > max_matches:
                max_matches = len(signals)
                best_type = query_type
                best_signals = signals

        # Calculate confidence
        if max_matches == 0:
            confidence = 0.3  # Low confidence for unknown
        elif max_matches == 1:
            confidence = 0.6
        elif max_matches == 2:
            confidence = 0.8
        else:
            confidence = 0.9

        # Get suggested tier
        suggested_tier = DEFAULT_ROUTING.get(best_type, ModelTier.BALANCED)

        return QueryClassification(
            query_type=best_type,
            confidence=confidence,
            signals=best_signals,
            suggested_tier=suggested_tier,
        )


class SmartRouter:
    """
    Route queries to appropriate models with fallback support.

    Implements: Spec §8.1 Phase 4 - Smart Routing
    """

    def __init__(
        self,
        classifier: QueryClassifier | None = None,
        routing_overrides: dict[QueryType, ModelTier] | None = None,
        fallback_chains: dict[ModelTier, FallbackChain] | None = None,
    ):
        """
        Initialize router.

        Args:
            classifier: Query classifier to use
            routing_overrides: Override default routing decisions
            fallback_chains: Custom fallback chains by tier
        """
        self.classifier = classifier or QueryClassifier()
        self.routing = dict(DEFAULT_ROUTING)
        if routing_overrides:
            self.routing.update(routing_overrides)

        self.fallback_chains = dict(DEFAULT_FALLBACK_CHAINS)
        if fallback_chains:
            self.fallback_chains.update(fallback_chains)

        # Tracking for learning
        self._routing_history: list[dict[str, Any]] = []

    def route(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        force_tier: ModelTier | None = None,
    ) -> RoutingDecision:
        """
        Route a query to the appropriate model.

        Implements: Spec §8.1 Model selection by query

        Args:
            query: The query to route
            context: Optional context for routing decisions
            force_tier: Force a specific tier (overrides classification)

        Returns:
            RoutingDecision with primary model and fallback chain
        """
        # Classify query
        classification = self.classifier.classify(query)

        # Determine tier
        if force_tier is not None:
            tier = force_tier
            reason = f"Forced tier: {tier.value}"
        else:
            tier = self.routing.get(classification.query_type, ModelTier.BALANCED)
            reason = f"Query type {classification.query_type.value} -> {tier.value}"

        # Apply context-based adjustments
        if context:
            tier, reason = self._adjust_for_context(tier, reason, context)

        # Get fallback chain
        chain = self.fallback_chains.get(tier, DEFAULT_FALLBACK_CHAINS[ModelTier.BALANCED])

        # Create decision
        decision = RoutingDecision(
            primary_model=MODEL_BY_TIER[tier],
            fallback_chain=chain.models[1:] if len(chain.models) > 1 else [],
            query_type=classification.query_type,
            confidence=classification.confidence,
            reason=reason,
        )

        # Track for learning
        self._routing_history.append(
            {
                "query": query[:100],
                "classification": classification.query_type.value,
                "tier": tier.value,
                "model": decision.primary_model,
            }
        )

        return decision

    def _adjust_for_context(
        self, tier: ModelTier, reason: str, context: dict[str, Any]
    ) -> tuple[ModelTier, str]:
        """Adjust routing based on context."""
        # Depth-based adjustment
        depth = context.get("depth", 0)
        if depth > 1 and tier == ModelTier.POWERFUL:
            # Use lighter model for deep recursion
            return ModelTier.BALANCED, f"{reason} (downgraded for depth={depth})"

        # Token budget adjustment
        remaining_tokens = context.get("remaining_tokens")
        if remaining_tokens is not None and remaining_tokens < 10000:
            if tier == ModelTier.POWERFUL:
                return ModelTier.BALANCED, f"{reason} (downgraded for low budget)"
            if tier == ModelTier.BALANCED:
                return ModelTier.FAST, f"{reason} (downgraded for low budget)"

        # Complexity override
        if context.get("force_powerful"):
            return ModelTier.POWERFUL, f"{reason} (forced powerful by context)"

        return tier, reason

    def get_fallback_chain(self, tier: ModelTier) -> FallbackChain:
        """Get fallback chain for a tier."""
        return self.fallback_chains.get(tier, DEFAULT_FALLBACK_CHAINS[ModelTier.BALANCED])

    def record_outcome(
        self, query: str, model_used: str, success: bool, latency_ms: float
    ) -> None:
        """
        Record routing outcome for learning.

        Args:
            query: Original query
            model_used: Model that was used
            success: Whether the query succeeded
            latency_ms: Response latency
        """
        # Find matching history entry
        for entry in reversed(self._routing_history[-100:]):
            if entry["query"] == query[:100]:
                entry["outcome"] = {
                    "model_used": model_used,
                    "success": success,
                    "latency_ms": latency_ms,
                }
                break

    def get_statistics(self) -> dict[str, Any]:
        """Get routing statistics."""
        if not self._routing_history:
            return {"total_routes": 0}

        by_type: dict[str, int] = {}
        by_tier: dict[str, int] = {}
        successes = 0
        failures = 0

        for entry in self._routing_history:
            query_type = entry["classification"]
            tier = entry["tier"]

            by_type[query_type] = by_type.get(query_type, 0) + 1
            by_tier[tier] = by_tier.get(tier, 0) + 1

            if "outcome" in entry:
                if entry["outcome"]["success"]:
                    successes += 1
                else:
                    failures += 1

        return {
            "total_routes": len(self._routing_history),
            "by_query_type": by_type,
            "by_tier": by_tier,
            "success_rate": successes / (successes + failures) if (successes + failures) > 0 else None,
            "outcomes_recorded": successes + failures,
        }

    def reset_statistics(self) -> None:
        """Reset routing statistics."""
        self._routing_history.clear()


class FallbackExecutor:
    """
    Execute queries with automatic fallback on failure.

    Implements: Spec §8.1 Fallback chains
    """

    def __init__(self, router: SmartRouter):
        """
        Initialize executor.

        Args:
            router: Router for getting fallback chains
        """
        self.router = router
        self._execution_history: list[dict[str, Any]] = []

    async def execute_with_fallback(
        self,
        query: str,
        execute_fn: Any,  # Callable[[str, str], Awaitable[str]]
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """
        Execute query with automatic fallback on failure.

        Args:
            query: Query to execute
            execute_fn: Async function(query, model) -> result
            context: Optional context for routing

        Returns:
            (result, model_used) tuple

        Raises:
            Exception: If all models in fallback chain fail
        """
        import time

        decision = self.router.route(query, context)
        models = decision.all_models
        last_error: Exception | None = None

        for model in models:
            start = time.time()
            try:
                result = await execute_fn(query, model)
                latency = (time.time() - start) * 1000

                # Record success
                self.router.record_outcome(query, model, success=True, latency_ms=latency)
                self._execution_history.append(
                    {
                        "query": query[:100],
                        "model": model,
                        "success": True,
                        "latency_ms": latency,
                    }
                )

                return result, model

            except Exception as e:
                latency = (time.time() - start) * 1000
                last_error = e

                # Record failure
                self.router.record_outcome(query, model, success=False, latency_ms=latency)
                self._execution_history.append(
                    {
                        "query": query[:100],
                        "model": model,
                        "success": False,
                        "error": str(e),
                        "latency_ms": latency,
                    }
                )

                # Continue to next model in chain
                continue

        # All models failed
        raise last_error or RuntimeError("All models in fallback chain failed")

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {"total_executions": 0}

        successes = sum(1 for e in self._execution_history if e["success"])
        failures = len(self._execution_history) - successes
        latencies = [e["latency_ms"] for e in self._execution_history if e["success"]]

        return {
            "total_executions": len(self._execution_history),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / len(self._execution_history),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        }


__all__ = [
    "DEFAULT_FALLBACK_CHAINS",
    "DEFAULT_ROUTING",
    "MODEL_BY_TIER",
    "FallbackChain",
    "FallbackExecutor",
    "ModelTier",
    "QueryClassification",
    "QueryClassifier",
    "QueryType",
    "RoutingDecision",
    "SmartRouter",
]
