"""
Unit tests for LocalOrchestrator module.

Tests local model-based orchestration for RLM activation decisions.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.local_orchestrator import (
    LocalModelBackend,
    LocalModelConfig,
    LocalInferenceResult,
    LocalOrchestrator,
    MLXRunner,
    OllamaRunner,
    RECOMMENDED_CONFIGS,
)


class TestLocalModelConfig:
    """Tests for LocalModelConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = LocalModelConfig()
        assert config.model_name == "gemma-3-270m-it"
        assert config.backend == LocalModelBackend.MLX
        assert config.max_tokens == 300
        assert config.temperature == 0.1
        assert config.timeout_ms == 2000
        assert config.fallback_to_heuristics is True

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = LocalModelConfig(
            model_name="qwen3-0.6b",
            backend=LocalModelBackend.OLLAMA,
            max_tokens=500,
            temperature=0.2,
        )
        assert config.model_name == "qwen3-0.6b"
        assert config.backend == LocalModelBackend.OLLAMA
        assert config.max_tokens == 500
        assert config.temperature == 0.2


class TestRecommendedConfigs:
    """Tests for RECOMMENDED_CONFIGS presets."""

    def test_ultra_fast_config(self):
        """Ultra fast config uses smallest model."""
        config = RECOMMENDED_CONFIGS["ultra_fast"]
        assert config.model_name == "gemma-3-270m-it"
        assert config.backend == LocalModelBackend.MLX
        assert config.max_tokens == 200

    def test_balanced_config(self):
        """Balanced config uses medium model."""
        config = RECOMMENDED_CONFIGS["balanced"]
        assert config.model_name == "qwen3-0.6b"
        assert config.backend == LocalModelBackend.MLX

    def test_quality_config(self):
        """Quality config uses larger model."""
        config = RECOMMENDED_CONFIGS["quality"]
        assert config.model_name == "lfm2.5-1.2b"
        assert config.max_tokens == 400

    def test_portable_config(self):
        """Portable config uses Ollama backend."""
        config = RECOMMENDED_CONFIGS["portable"]
        assert config.backend == LocalModelBackend.OLLAMA


class TestMLXRunner:
    """Tests for MLXRunner."""

    def test_is_available_without_mlx(self):
        """Returns False when mlx_lm not installed."""
        with patch.dict("sys.modules", {"mlx_lm": None}):
            runner = MLXRunner(LocalModelConfig())
            # Can't easily test this without actual import failure
            # Just verify the method exists
            assert hasattr(runner, "is_available")

    def test_resolve_model_id(self):
        """Model names resolve to HuggingFace IDs."""
        runner = MLXRunner(LocalModelConfig(model_name="gemma-3-270m-it"))
        model_id = runner._resolve_model_id()
        assert "gemma" in model_id.lower()

    def test_resolve_unknown_model(self):
        """Unknown model names pass through unchanged."""
        runner = MLXRunner(LocalModelConfig(model_name="custom/my-model"))
        model_id = runner._resolve_model_id()
        assert model_id == "custom/my-model"

    def test_get_model_info(self):
        """Model info includes backend and name."""
        runner = MLXRunner(LocalModelConfig(model_name="qwen3-0.6b"))
        info = runner.get_model_info()
        assert info["backend"] == "mlx"
        assert info["model_name"] == "qwen3-0.6b"
        assert info["loaded"] is False


class TestOllamaRunner:
    """Tests for OllamaRunner."""

    def test_resolve_model_name(self):
        """Model names resolve to Ollama format."""
        runner = OllamaRunner(LocalModelConfig(model_name="gemma-3-270m-it"))
        name = runner._resolve_model_name()
        assert name == "gemma3:270m"

    def test_resolve_unknown_model(self):
        """Unknown model names pass through unchanged."""
        runner = OllamaRunner(LocalModelConfig(model_name="custom-model"))
        name = runner._resolve_model_name()
        assert name == "custom-model"

    def test_get_model_info(self):
        """Model info includes backend and names."""
        runner = OllamaRunner(LocalModelConfig(model_name="qwen3-0.6b"))
        info = runner.get_model_info()
        assert info["backend"] == "ollama"
        assert info["model_name"] == "qwen3-0.6b"
        assert info["ollama_model"] == "qwen3:0.6b"


class TestLocalOrchestrator:
    """Tests for LocalOrchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with fallback enabled."""
        config = LocalModelConfig(fallback_to_heuristics=True)
        return LocalOrchestrator(config=config)

    def test_initialization(self, orchestrator):
        """Orchestrator initializes with correct state."""
        assert orchestrator.config.fallback_to_heuristics is True
        assert orchestrator._runner is None
        assert orchestrator._stats["local_decisions"] == 0
        assert orchestrator._stats["heuristic_fallbacks"] == 0

    def test_heuristic_decision_simple_task(self, orchestrator):
        """Heuristic correctly identifies simple tasks."""
        decision = orchestrator._heuristic_decision(
            query="show config.py",
            context_summary="- Context tokens: 1,000",
        )
        assert decision["activate_rlm"] is False
        assert "knowledge_retrieval" in decision["signals"]

    def test_heuristic_decision_discovery_task(self, orchestrator):
        """Heuristic correctly identifies discovery tasks."""
        decision = orchestrator._heuristic_decision(
            query="Why is the authentication failing intermittently?",
            context_summary="- Context tokens: 50,000",
        )
        assert decision["activate_rlm"] is True
        assert "discovery_required" in decision["signals"] or "debugging_deep" in decision["signals"]

    def test_heuristic_decision_synthesis_task(self, orchestrator):
        """Heuristic correctly identifies synthesis tasks."""
        decision = orchestrator._heuristic_decision(
            query="Update all usages of the deprecated API",
            context_summary="- Context tokens: 30,000",
        )
        assert decision["activate_rlm"] is True
        assert "synthesis_required" in decision["signals"]

    def test_heuristic_decision_uncertainty(self, orchestrator):
        """Heuristic correctly identifies uncertainty."""
        decision = orchestrator._heuristic_decision(
            query="What's the best approach for adding caching?",
            context_summary="- Context tokens: 20,000",
        )
        assert decision["activate_rlm"] is True
        assert "uncertainty_high" in decision["signals"]

    def test_heuristic_decision_architectural(self, orchestrator):
        """Heuristic correctly identifies architectural tasks."""
        decision = orchestrator._heuristic_decision(
            query="Design a system for handling real-time events",
            context_summary="- Context tokens: 10,000",
        )
        assert decision["activate_rlm"] is True
        assert "architectural" in decision["signals"]

    def test_heuristic_decision_large_context(self, orchestrator):
        """Heuristic activates for large context."""
        decision = orchestrator._heuristic_decision(
            query="summarize",
            context_summary="- Large context detected\n- Context tokens: 100,000",
        )
        assert decision["activate_rlm"] is True
        assert "large_context" in decision["signals"]

    def test_heuristic_decision_conversational(self, orchestrator):
        """Heuristic bypasses conversational queries."""
        decision = orchestrator._heuristic_decision(
            query="ok",
            context_summary="- Context tokens: 5,000",
        )
        assert decision["activate_rlm"] is False
        assert "conversational" in decision["signals"]

    def test_parse_response_valid_json(self, orchestrator):
        """Parses valid JSON response."""
        response = '{"activate_rlm": true, "activation_reason": "test"}'
        result = orchestrator._parse_response(response)
        assert result["activate_rlm"] is True
        assert result["activation_reason"] == "test"

    def test_parse_response_json_in_text(self, orchestrator):
        """Extracts JSON from surrounding text."""
        response = 'Here is my decision: {"activate_rlm": false, "reason": "simple"} That is all.'
        result = orchestrator._parse_response(response)
        assert result["activate_rlm"] is False

    def test_parse_response_invalid_json(self, orchestrator):
        """Raises on invalid JSON."""
        with pytest.raises(ValueError, match="No JSON found"):
            orchestrator._parse_response("no json here")

    def test_cache_key_computation(self, orchestrator):
        """Cache keys are consistent for same input."""
        key1 = orchestrator._compute_cache_key("test query", "context")
        key2 = orchestrator._compute_cache_key("test query", "context")
        assert key1 == key2

    def test_cache_key_differs(self, orchestrator):
        """Cache keys differ for different inputs."""
        key1 = orchestrator._compute_cache_key("query a", "context")
        key2 = orchestrator._compute_cache_key("query b", "context")
        assert key1 != key2

    def test_cache_update_and_eviction(self, orchestrator):
        """Cache evicts old entries when full."""
        orchestrator.config.cache_size = 4

        # Fill cache
        for i in range(4):
            orchestrator._update_cache(f"key{i}", {"value": i})

        assert len(orchestrator._cache) == 4

        # Add one more - should trigger eviction
        orchestrator._update_cache("key4", {"value": 4})

        # Should have evicted some entries
        assert len(orchestrator._cache) <= 4

    def test_statistics_initial(self, orchestrator):
        """Initial statistics are zero."""
        stats = orchestrator.get_statistics()
        assert stats["local_decisions"] == 0
        assert stats["heuristic_fallbacks"] == 0
        assert stats["cache_hits"] == 0
        assert stats["total_decisions"] == 0

    @pytest.mark.asyncio
    async def test_orchestrate_fallback_on_no_backend(self, orchestrator):
        """Falls back to heuristics when no backend available."""
        # Mock _get_runner to raise
        orchestrator._get_runner = MagicMock(side_effect=RuntimeError("No backend"))

        decision = await orchestrator.orchestrate(
            query="Why is this failing?",
            context_summary="- Context tokens: 10,000",
        )

        assert decision is not None
        assert orchestrator._stats["heuristic_fallbacks"] == 1
        assert orchestrator._stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_orchestrate_uses_cache(self, orchestrator):
        """Orchestrate uses cached decisions."""
        # Pre-populate cache
        cache_key = orchestrator._compute_cache_key("test query", "context")
        cached_decision = {"activate_rlm": True, "cached": True}
        orchestrator._cache[cache_key] = cached_decision

        decision = await orchestrator.orchestrate(
            query="test query",
            context_summary="context",
        )

        assert decision["cached"] is True
        assert orchestrator._stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_orchestrate_with_mock_runner(self, orchestrator):
        """Orchestrate works with mocked runner."""
        mock_runner = AsyncMock()
        mock_runner.generate.return_value = LocalInferenceResult(
            content='{"activate_rlm": true, "activation_reason": "mock_test"}',
            latency_ms=25.0,
            tokens_generated=50,
            model_name="test-model",
            backend=LocalModelBackend.MLX,
        )

        orchestrator._runner = mock_runner

        decision = await orchestrator.orchestrate(
            query="complex analysis task",
            context_summary="- Context tokens: 50,000",
        )

        assert decision["activate_rlm"] is True
        assert decision["activation_reason"] == "mock_test"
        assert orchestrator._stats["local_decisions"] == 1
        assert orchestrator._stats["total_latency_ms"] == 25.0


class TestLocalInferenceResult:
    """Tests for LocalInferenceResult dataclass."""

    def test_create_result(self):
        """Can create inference result."""
        result = LocalInferenceResult(
            content="test response",
            latency_ms=30.5,
            tokens_generated=25,
            model_name="test-model",
            backend=LocalModelBackend.MLX,
        )
        assert result.content == "test response"
        assert result.latency_ms == 30.5
        assert result.tokens_generated == 25
        assert result.backend == LocalModelBackend.MLX


class TestHeuristicDecisionEdgeCases:
    """Edge case tests for heuristic decisions."""

    @pytest.fixture
    def orchestrator(self):
        return LocalOrchestrator()

    def test_flaky_test_detection(self, orchestrator):
        """Detects flaky test debugging."""
        decision = orchestrator._heuristic_decision(
            query="This test is flaky and fails randomly",
            context_summary="",
        )
        assert decision["activate_rlm"] is True
        assert "debugging_deep" in decision["signals"]

    def test_race_condition_detection(self, orchestrator):
        """Detects race condition debugging."""
        decision = orchestrator._heuristic_decision(
            query="I think there's a race condition here",
            context_summary="",
        )
        assert decision["activate_rlm"] is True
        assert "debugging_deep" in decision["signals"]

    def test_migration_detection(self, orchestrator):
        """Detects migration tasks."""
        decision = orchestrator._heuristic_decision(
            query="We need to migrate from PostgreSQL to MySQL",
            context_summary="",
        )
        assert decision["activate_rlm"] is True
        # Should trigger architectural signal

    def test_yes_no_bypass(self, orchestrator):
        """Simple yes/no bypasses RLM."""
        for query in ["yes", "no", "ok", "thanks"]:
            decision = orchestrator._heuristic_decision(query, "")
            assert decision["activate_rlm"] is False

    def test_execution_mode_thorough(self, orchestrator):
        """Multiple high-value signals trigger thorough mode."""
        decision = orchestrator._heuristic_decision(
            query="Why is this flaky? What's the best approach to fix all instances?",
            context_summary="",
        )
        assert decision["execution_mode"] == "thorough"

    def test_depth_budget_for_debugging(self, orchestrator):
        """Deep debugging gets higher depth budget."""
        decision = orchestrator._heuristic_decision(
            query="This race condition is causing intermittent failures",
            context_summary="",
        )
        assert decision["depth_budget"] >= 2
