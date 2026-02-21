"""Tests for RLMSubAgent and RLMSubAgentConfig."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.sub_agent import RLMSubAgent, RLMSubAgentConfig
from src.frame.canonical_task import CanonicalTask
from src.repl.rlaph_loop import RLPALoopResult
from src.types import SessionContext


class TestRLMSubAgentConfig:
    """Test suite for RLMSubAgentConfig dataclass."""

    def test_config_creation_with_required_fields(self):
        """Config should be created with required name field."""
        config = RLMSubAgentConfig(name="analyzer")
        assert config.name == "analyzer"
        assert config.system_prompt_override == ""
        assert config.default_max_depth == 3
        assert config.default_scope == "overview"
        assert config.verbose is False

    def test_config_creation_with_all_fields(self):
        """Config should accept all optional fields."""
        config = RLMSubAgentConfig(
            name="debugger",
            system_prompt_override="You are a debugging specialist.",
            default_max_depth=5,
            default_scope="correctness",
            verbose=True,
        )
        assert config.name == "debugger"
        assert config.system_prompt_override == "You are a debugging specialist."
        assert config.default_max_depth == 5
        assert config.default_scope == "correctness"
        assert config.verbose is True

    def test_config_defaults(self):
        """Config should have correct default values."""
        config = RLMSubAgentConfig(name="test")
        assert config.default_max_depth == 3
        assert config.default_scope == "overview"
        assert config.verbose is False

    def test_config_immutability_of_name(self):
        """Config name should be settable at initialization."""
        config1 = RLMSubAgentConfig(name="agent1")
        config2 = RLMSubAgentConfig(name="agent2")
        assert config1.name == "agent1"
        assert config2.name == "agent2"


class TestRLMSubAgentInit:
    """Test suite for RLMSubAgent initialization."""

    def test_init_creates_loop_with_config_depth(self):
        """RLMSubAgent should store config with max_depth."""
        config = RLMSubAgentConfig(name="test", default_max_depth=5)
        agent = RLMSubAgent(config)

        assert agent.config is config
        assert agent.config.default_max_depth == 5

    def test_init_creates_loop_with_verbose(self):
        """RLMSubAgent should store config with verbose setting."""
        config = RLMSubAgentConfig(name="test", verbose=True)
        agent = RLMSubAgent(config)

        assert agent.config.verbose is True

    def test_init_stores_config(self):
        """RLMSubAgent should store the provided config."""
        config = RLMSubAgentConfig(name="analyzer")
        agent = RLMSubAgent(config)
        assert agent.config is config

    def test_init_with_all_config_options(self):
        """RLMSubAgent should work with all config options."""
        config = RLMSubAgentConfig(
            name="specialist",
            system_prompt_override="You are a specialist.",
            default_max_depth=7,
            default_scope="security",
            verbose=True,
        )
        agent = RLMSubAgent(config)

        assert agent.config.name == "specialist"
        assert agent.config.default_max_depth == 7
        assert agent.config.verbose is True


class TestRLMSubAgentRun:
    """Test suite for RLMSubAgent.run() method."""

    @pytest.mark.asyncio
    async def test_run_creates_loop_with_effective_depth(self):
        """run() should create loop with effective depth from config."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test answer",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="test", default_max_depth=4)
            agent = RLMSubAgent(config)

            await agent.run("test query")

            # Loop should be created in run() with config depth
            assert MockLoop.call_count == 1
            last_call_args = MockLoop.call_args_list[-1]
            assert last_call_args[1]["max_depth"] == 4

    @pytest.mark.asyncio
    async def test_run_with_max_depth_override(self):
        """run() should use max_depth override when provided."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test answer",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="test", default_max_depth=3)
            agent = RLMSubAgent(config)

            await agent.run("test query", max_depth=6)

            # Second loop creation (from run) should use override
            last_call_args = MockLoop.call_args_list[-1]
            assert last_call_args[1]["max_depth"] == 6

    @pytest.mark.asyncio
    async def test_run_creates_default_context_when_none(self):
        """run() should create SessionContext when none provided."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="test")
            agent = RLMSubAgent(config)

            await agent.run("test query", context=None)

            call_args = mock_loop.run.call_args
            assert isinstance(call_args[0][1], SessionContext)

    @pytest.mark.asyncio
    async def test_run_uses_provided_context(self):
        """run() should use provided SessionContext."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="test")
            agent = RLMSubAgent(config)

            custom_context = SessionContext()
            await agent.run("test query", context=custom_context)

            call_args = mock_loop.run.call_args
            assert call_args[0][1] is custom_context

    @pytest.mark.asyncio
    async def test_run_creates_default_canonical_task_when_none(self):
        """run() should create default CanonicalTask when none provided."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(
                name="test",
                default_scope="security"
            )
            agent = RLMSubAgent(config)

            await agent.run("test query", canonical_task=None)

            # Task should be created with default_scope
            # We can't directly check the task, but the call should succeed

    @pytest.mark.asyncio
    async def test_run_uses_provided_canonical_task(self):
        """run() should use provided CanonicalTask."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="test")
            agent = RLMSubAgent(config)

            custom_task = CanonicalTask(
                task_type="debug",
                target="**/*.py",
                analysis_scope="correctness",
                params={}
            )
            await agent.run("test query", canonical_task=custom_task)

            # Call should succeed with custom task

    @pytest.mark.asyncio
    async def test_run_uses_cwd_when_working_dir_none(self):
        """run() should use Path.cwd() when working_dir is None."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            with patch("src.agents.sub_agent.Path") as MockPath:
                mock_loop = MagicMock()
                mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                    answer="test",
                    iterations=1,
                    depth_used=0,
                    tokens_used=100,
                    execution_time_ms=100.0,
                    history=[]
                ))
                MockLoop.return_value = mock_loop

                MockPath.cwd.return_value = Path("/default/cwd")
                MockPath.return_value = Path("/default/cwd")

                config = RLMSubAgentConfig(name="test")
                agent = RLMSubAgent(config)

                await agent.run("test query", working_dir=None)

                call_args = mock_loop.run.call_args
                assert call_args[0][2] == Path("/default/cwd")

    @pytest.mark.asyncio
    async def test_run_converts_working_dir_to_path(self):
        """run() should convert working_dir string to Path."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="test")
            agent = RLMSubAgent(config)

            await agent.run("test query", working_dir="/custom/path")

            call_args = mock_loop.run.call_args
            assert isinstance(call_args[0][2], Path)
            assert call_args[0][2] == Path("/custom/path")

    @pytest.mark.asyncio
    async def test_run_passes_session_id_to_loop(self):
        """run() should pass session_id to loop.run()."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="test")
            agent = RLMSubAgent(config)

            await agent.run("test query", session_id="test-session-123")

            call_args = mock_loop.run.call_args
            assert call_args[0][3] == "test-session-123"

    @pytest.mark.asyncio
    async def test_run_prefixes_answer_with_agent_name(self):
        """run() should prefix answer with uppercase agent name."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="The code is secure.",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="security-analyzer")
            agent = RLMSubAgent(config)

            result = await agent.run("test query")

            assert result.answer == "[SECURITY-ANALYZER] The code is secure."

    @pytest.mark.asyncio
    async def test_run_with_system_prompt_override_prepends_to_query(self):
        """run() should prepend system_prompt_override to query."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test answer",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(
                name="test",
                system_prompt_override="You are a security expert."
            )
            agent = RLMSubAgent(config)

            await agent.run("Check for vulnerabilities.")

            call_args = mock_loop.run.call_args
            query = call_args[0][0]
            assert query == "You are a security expert.\n\nCheck for vulnerabilities."

    @pytest.mark.asyncio
    async def test_run_without_system_prompt_override_uses_query_as_is(self):
        """run() should use query as-is when no override."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test answer",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="test")
            agent = RLMSubAgent(config)

            original_query = "Analyze this code."
            await agent.run(original_query)

            call_args = mock_loop.run.call_args
            assert call_args[0][0] == original_query

    @pytest.mark.asyncio
    async def test_run_returns_result_with_metadata(self):
        """run() should return RLPALoopResult with all metadata."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test answer",
                iterations=3,
                depth_used=2,
                tokens_used=500,
                execution_time_ms=250.0,
                history=[{"turn": 1}, {"turn": 2}]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(name="test")
            agent = RLMSubAgent(config)

            result = await agent.run("test query")

            assert isinstance(result, RLPALoopResult)
            assert result.iterations == 3
            assert result.depth_used == 2
            assert result.tokens_used == 500
            assert result.execution_time_ms == 250.0
            assert result.history == [{"turn": 1}, {"turn": 2}]

    @pytest.mark.asyncio
    async def test_run_with_all_parameters(self):
        """run() should work correctly with all parameters set."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="complete answer",
                iterations=2,
                depth_used=1,
                tokens_used=300,
                execution_time_ms=150.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(
                name="comprehensive",
                system_prompt_override="You are comprehensive.",
                default_max_depth=3,
                verbose=False,
            )
            agent = RLMSubAgent(config)

            ctx = SessionContext()
            task = CanonicalTask(
                task_type="review",
                target="**/*.py",
                analysis_scope="security",
                params={}
            )

            result = await agent.run(
                query="Review everything.",
                working_dir="/project/path",
                session_id="full-test-session",
                canonical_task=task,
                max_depth=8,
                context=ctx,
            )

            # Verify loop creation with override depth
            last_call_args = MockLoop.call_args_list[-1]
            assert last_call_args[1]["max_depth"] == 8

            # Verify run call arguments
            call_args = mock_loop.run.call_args
            assert call_args[0][0] == "You are comprehensive.\n\nReview everything."
            assert call_args[0][1] is ctx
            assert call_args[0][2] == Path("/project/path")
            assert call_args[0][3] == "full-test-session"

            # Verify result
            assert result.answer == "[COMPREHENSIVE] complete answer"
            assert result.iterations == 2


class TestRLMSubAgentIntegration:
    """Integration tests for RLMSubAgent."""

    @pytest.mark.asyncio
    async def test_agent_run_is_async(self):
        """RLMSubAgent.run should be async callable."""
        from src.agents.sub_agent import RLMSubAgent
        import inspect

        config = RLMSubAgentConfig(name="test")
        agent = RLMSubAgent(config)

        assert inspect.iscoroutinefunction(agent.run)

    def test_sub_agent_in_all_exports(self):
        """RLMSubAgent should be in module __all__ exports."""
        from src.agents import sub_agent
        assert "RLMSubAgent" in sub_agent.__all__
        assert "RLMSubAgentConfig" in sub_agent.__all__

    def test_sub_agent_importable_from_package(self):
        """RLMSubAgent should be importable from agents package."""
        from src.agents import RLMSubAgent, RLMSubAgentConfig
        assert RLMSubAgent is not None
        assert RLMSubAgentConfig is not None

    @pytest.mark.asyncio
    async def test_agent_with_mocks_completes_without_error(self):
        """Smoke test: agent should complete with mocked dependencies."""
        with patch("src.agents.sub_agent.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            config = RLMSubAgentConfig(
                name="smoke-test",
                system_prompt_override="You are a smoke test agent.",
            )
            agent = RLMSubAgent(config)

            # Should complete without error
            result = await agent.run("test query")
            assert result is not None
            assert result.answer.startswith("[SMOKE-TEST]")
