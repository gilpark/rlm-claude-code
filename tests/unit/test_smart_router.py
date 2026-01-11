"""
Unit tests for smart_router module.

Implements: Spec §8.1 Phase 4 - Smart Routing tests
"""

import pytest

from src.smart_router import (
    DEFAULT_ROUTING,
    FallbackChain,
    FallbackExecutor,
    ModelTier,
    QueryClassification,
    QueryClassifier,
    QueryType,
    RoutingDecision,
    SmartRouter,
)


class TestQueryType:
    """Tests for QueryType enum."""

    def test_all_query_types_exist(self):
        """All expected query types exist."""
        expected = [
            "factual",
            "analytical",
            "creative",
            "code",
            "search",
            "summarization",
            "planning",
            "debugging",
            "unknown",
        ]
        actual = [qt.value for qt in QueryType]
        assert set(expected) == set(actual)


class TestModelTier:
    """Tests for ModelTier enum."""

    def test_all_tiers_exist(self):
        """All expected tiers exist."""
        expected = ["fast", "balanced", "powerful"]
        actual = [mt.value for mt in ModelTier]
        assert set(expected) == set(actual)


class TestQueryClassification:
    """Tests for QueryClassification dataclass."""

    def test_create_classification(self):
        """Can create classification."""
        classification = QueryClassification(
            query_type=QueryType.CODE,
            confidence=0.8,
            signals=["code", "function"],
            suggested_tier=ModelTier.BALANCED,
        )

        assert classification.query_type == QueryType.CODE
        assert classification.confidence == 0.8
        assert len(classification.signals) == 2

    def test_model_property(self):
        """Model property returns correct model."""
        classification = QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.9,
            signals=[],
            suggested_tier=ModelTier.FAST,
        )

        assert "haiku" in classification.model.lower()


class TestQueryClassifier:
    """Tests for QueryClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return QueryClassifier()

    def test_classify_factual_query(self, classifier):
        """Classifies factual queries."""
        result = classifier.classify("What is Python?")
        assert result.query_type == QueryType.FACTUAL

    def test_classify_code_query(self, classifier):
        """Classifies code queries."""
        result = classifier.classify("Write a function to sort a list")
        assert result.query_type == QueryType.CODE

    def test_classify_analytical_query(self, classifier):
        """Classifies analytical queries."""
        result = classifier.classify("Why does this approach work better?")
        assert result.query_type == QueryType.ANALYTICAL

    def test_classify_search_query(self, classifier):
        """Classifies search queries."""
        result = classifier.classify("Find all Python files in the project")
        assert result.query_type == QueryType.SEARCH

    def test_classify_summarization_query(self, classifier):
        """Classifies summarization queries."""
        result = classifier.classify("Summarize this document")
        assert result.query_type == QueryType.SUMMARIZATION

    def test_classify_planning_query(self, classifier):
        """Classifies planning queries."""
        result = classifier.classify("How should I architect this system?")
        assert result.query_type == QueryType.PLANNING

    def test_classify_debugging_query(self, classifier):
        """Classifies debugging queries."""
        result = classifier.classify("Troubleshoot this issue, it's not working")
        assert result.query_type == QueryType.DEBUGGING

    def test_classify_unknown_query(self, classifier):
        """Returns unknown for ambiguous queries."""
        result = classifier.classify("xyz123")
        assert result.query_type == QueryType.UNKNOWN
        assert result.confidence < 0.5

    def test_confidence_increases_with_matches(self, classifier):
        """Confidence increases with more signal matches."""
        simple = classifier.classify("code")
        complex_query = classifier.classify("implement a function in the code class")

        assert complex_query.confidence >= simple.confidence

    def test_custom_patterns(self):
        """Can add custom patterns."""
        custom = QueryClassifier(
            custom_patterns={QueryType.CODE: [r"\bcustom_pattern\b"]}
        )

        result = custom.classify("This has custom_pattern in it")
        assert result.query_type == QueryType.CODE


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_create_decision(self):
        """Can create routing decision."""
        decision = RoutingDecision(
            primary_model="claude-sonnet",
            fallback_chain=["claude-opus", "claude-haiku"],
            query_type=QueryType.CODE,
            confidence=0.8,
            reason="Code query -> balanced",
        )

        assert decision.primary_model == "claude-sonnet"
        assert len(decision.fallback_chain) == 2

    def test_all_models_property(self):
        """all_models returns correct order."""
        decision = RoutingDecision(
            primary_model="model1",
            fallback_chain=["model2", "model3"],
            query_type=QueryType.UNKNOWN,
            confidence=0.5,
            reason="test",
        )

        assert decision.all_models == ["model1", "model2", "model3"]


class TestSmartRouter:
    """Tests for SmartRouter class."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        return SmartRouter()

    def test_route_code_query(self, router):
        """Routes code queries to balanced tier."""
        decision = router.route("Write a Python function")

        assert decision.query_type == QueryType.CODE
        assert "sonnet" in decision.primary_model.lower()

    def test_route_simple_query(self, router):
        """Routes simple queries to fast tier."""
        decision = router.route("What is 2+2?")

        assert decision.query_type == QueryType.FACTUAL
        assert "haiku" in decision.primary_model.lower()

    def test_anvendelse_force_tier(self, router):
        """Can force specific tier."""
        decision = router.route("Simple question", force_tier=ModelTier.POWERFUL)

        assert "opus" in decision.primary_model.lower()
        assert "Forced tier" in decision.reason

    def test_fallback_chain_populated(self, router):
        """Fallback chain is populated."""
        decision = router.route("Write code")

        assert len(decision.fallback_chain) > 0

    def test_context_depth_adjustment(self, router):
        """Adjusts for recursion depth."""
        decision = router.route(
            "Plan the architecture",
            context={"depth": 2},
        )

        # Planning normally goes to powerful, but depth=2 should downgrade
        assert "downgraded for depth" in decision.reason.lower()

    def test_context_budget_adjustment(self, router):
        """Adjusts for low token budget."""
        decision = router.route(
            "Analyze this code",
            context={"remaining_tokens": 5000},
        )

        assert "downgraded for low budget" in decision.reason.lower()

    def test_routing_overrides(self):
        """Can override default routing."""
        router = SmartRouter(
            routing_overrides={QueryType.FACTUAL: ModelTier.POWERFUL}
        )

        decision = router.route("What is Python?")
        assert "opus" in decision.primary_model.lower()

    def test_record_outcome(self, router):
        """Can record routing outcome."""
        router.route("Test query")
        router.record_outcome("Test query", "claude-sonnet", success=True, latency_ms=100)

        stats = router.get_statistics()
        assert stats["outcomes_recorded"] == 1

    def test_get_statistics(self, router):
        """Can get routing statistics."""
        router.route("Code query about functions")
        router.route("What is Python?")
        router.route("Plan the architecture")

        stats = router.get_statistics()

        assert stats["total_routes"] == 3
        assert "by_query_type" in stats
        assert "by_tier" in stats

    def test_reset_statistics(self, router):
        """Can reset statistics."""
        router.route("Query 1")
        router.route("Query 2")

        router.reset_statistics()

        stats = router.get_statistics()
        assert stats["total_routes"] == 0


class TestFallbackChain:
    """Tests for FallbackChain dataclass."""

    def test_create_chain(self):
        """Can create fallback chain."""
        chain = FallbackChain(
            models=["model1", "model2"],
            max_retries=3,
        )

        assert len(chain.models) == 2
        assert chain.max_retries == 3

    def test_default_retry_errors(self):
        """Default retry errors include TimeoutError."""
        chain = FallbackChain(models=["model1"])

        assert TimeoutError in chain.retry_on_errors


class TestFallbackExecutor:
    """Tests for FallbackExecutor class."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        return SmartRouter()

    @pytest.fixture
    def executor(self, router):
        """Create executor instance."""
        return FallbackExecutor(router)

    @pytest.mark.asyncio
    async def test_execute_success(self, executor):
        """Executes successfully on first try."""

        async def mock_execute(query, model):
            return f"Result from {model}"

        result, model = await executor.execute_with_fallback(
            "Test query",
            mock_execute,
        )

        assert "Result from" in result
        assert model is not None

    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, executor):
        """Falls back on failure."""
        call_count = 0

        async def failing_then_success(query, model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return f"Result from {model}"

        result, model = await executor.execute_with_fallback(
            "Test query",
            failing_then_success,
        )

        assert call_count >= 2
        assert "Result from" in result

    @pytest.mark.asyncio
    async def test_execute_all_fail(self, executor):
        """Raises if all models fail."""

        async def always_fail(query, model):
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            await executor.execute_with_fallback("Test", always_fail)

    @pytest.mark.asyncio
    async def test_execution_statistics(self, executor):
        """Tracks execution statistics."""

        async def mock_execute(query, model):
            return "Result"

        await executor.execute_with_fallback("Query 1", mock_execute)
        await executor.execute_with_fallback("Query 2", mock_execute)

        stats = executor.get_execution_statistics()

        assert stats["total_executions"] == 2
        assert stats["successes"] == 2
        assert stats["success_rate"] == 1.0


class TestDefaultRouting:
    """Tests for default routing configuration."""

    def test_all_query_types_have_routing(self):
        """All query types have default routing."""
        for qt in QueryType:
            assert qt in DEFAULT_ROUTING

    def test_routing_values_are_valid_tiers(self):
        """All routing values are valid tiers."""
        for tier in DEFAULT_ROUTING.values():
            assert isinstance(tier, ModelTier)
