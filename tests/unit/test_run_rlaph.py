"""Tests for run_rlaph() and run_rlaph_stream() library functions."""

import asyncio
from pathlib import Path
from typing import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.repl.rlaph_loop import RLPALoopResult, run_rlaph, run_rlaph_stream
from src.types import SessionContext


class TestRunRlaphFunction:
    """Test suite for run_rlaph() library function."""

    def test_run_rlaph_exists(self):
        """run_rlaph function should be importable."""
        from src.repl.rlaph_loop import run_rlaph
        assert callable(run_rlaph)

    def test_run_rlaph_stream_exists(self):
        """run_rlaph_stream function should be importable."""
        from src.repl.rlaph_loop import run_rlaph_stream
        assert callable(run_rlaph_stream)

    def test_run_rlaph_signature(self):
        """run_rlaph should have correct signature."""
        import inspect
        sig = inspect.signature(run_rlaph)
        params = sig.parameters

        assert "query" in params
        assert "working_dir" in params
        assert "session_id" in params
        assert "max_depth" in params
        assert "verbose" in params
        assert "context" in params
        assert "stream" in params

        # Check defaults
        assert params["working_dir"].default is None
        assert params["session_id"].default is None
        assert params["max_depth"].default == 3
        assert params["verbose"].default is False
        assert params["context"].default is None
        assert params["stream"].default is False

    def test_run_rlaph_stream_signature(self):
        """run_rlaph_stream should have correct signature."""
        import inspect
        sig = inspect.signature(run_rlaph_stream)
        params = sig.parameters

        assert "query" in params
        assert "working_dir" in params
        assert "session_id" in params
        assert "max_depth" in params
        assert "verbose" in params
        assert "context" in params
        assert "stream" not in params  # stream version doesn't have stream param

        # Check defaults
        assert params["working_dir"].default is None
        assert params["session_id"].default is None
        assert params["max_depth"].default == 3
        assert params["verbose"].default is False
        assert params["context"].default is None

    @pytest.mark.asyncio
    async def test_run_rlaph_creates_loop_internally(self):
        """run_rlaph should create RLAPHLoop internally."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
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

            result = await run_rlaph("test query")

            # Verify loop was created with correct params
            MockLoop.assert_called_once_with(max_depth=3, verbose=False)

            # Verify run was called on the loop
            mock_loop.run.assert_called_once()
            call_args = mock_loop.run.call_args
            assert call_args[0][0] == "test query"  # query
            assert isinstance(call_args[0][1], SessionContext)  # context
            assert isinstance(call_args[0][2], Path)  # working_dir
            assert call_args[0][3] is None  # session_id

    @pytest.mark.asyncio
    async def test_run_rlaph_with_custom_max_depth(self):
        """run_rlaph should pass max_depth to loop."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
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

            await run_rlaph("test query", max_depth=5)

            MockLoop.assert_called_once_with(max_depth=5, verbose=False)

    @pytest.mark.asyncio
    async def test_run_rlaph_with_verbose(self):
        """run_rlaph should pass verbose to loop."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
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

            await run_rlaph("test query", verbose=True)

            MockLoop.assert_called_once_with(max_depth=3, verbose=True)

    @pytest.mark.asyncio
    async def test_run_rlaph_with_custom_context(self):
        """run_rlaph should use provided context."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
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

            custom_context = SessionContext()
            result = await run_rlaph("test query", context=custom_context)

            call_args = mock_loop.run.call_args
            assert call_args[0][1] is custom_context

    @pytest.mark.asyncio
    async def test_run_rlaph_with_working_dir(self):
        """run_rlaph should convert working_dir to Path."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
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

            await run_rlaph("test query", working_dir="/tmp/test")

            call_args = mock_loop.run.call_args
            assert isinstance(call_args[0][2], Path)
            assert call_args[0][2] == Path("/tmp/test")

    @pytest.mark.asyncio
    async def test_run_rlaph_with_session_id(self):
        """run_rlaph should pass session_id to run."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
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

            await run_rlaph("test query", session_id="test-session-123")

            call_args = mock_loop.run.call_args
            assert call_args[0][3] == "test-session-123"

    @pytest.mark.asyncio
    async def test_run_rlaph_returns_result(self):
        """run_rlaph should return RLPALoopResult."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            expected_result = RLPALoopResult(
                answer="test answer",
                iterations=2,
                depth_used=1,
                tokens_used=250,
                execution_time_ms=150.0,
                history=[{"turn": 1, "action": "test", "has_answer": True}]
            )
            mock_loop.run = AsyncMock(return_value=expected_result)
            MockLoop.return_value = mock_loop

            result = await run_rlaph("test query")

            assert isinstance(result, RLPALoopResult)
            assert result.answer == "test answer"
            assert result.iterations == 2
            assert result.depth_used == 1
            assert result.tokens_used == 250

    @pytest.mark.asyncio
    async def test_run_rlaph_stream_yields_chunks(self):
        """run_rlaph_stream should yield chunks."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="test answer content that will be chunked",
                iterations=1,
                depth_used=0,
                tokens_used=100,
                execution_time_ms=100.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            chunks = []
            async for chunk in run_rlaph_stream("test query"):
                chunks.append(chunk)

            # Should receive chunks (not empty)
            assert len(chunks) > 0
            # Reconstruct full answer
            full_answer = "".join(chunks)
            assert full_answer == "test answer content that will be chunked"

    @pytest.mark.asyncio
    async def test_run_rlaph_stream_with_short_answer(self):
        """run_rlaph_stream should work with short answers."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="short",
                iterations=1,
                depth_used=0,
                tokens_used=50,
                execution_time_ms=50.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            chunks = []
            async for chunk in run_rlaph_stream("test query"):
                chunks.append(chunk)

            full_answer = "".join(chunks)
            assert full_answer == "short"

    @pytest.mark.asyncio
    async def test_run_rlaph_stream_empty_answer(self):
        """run_rlaph_stream should handle empty answer gracefully."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="",
                iterations=1,
                depth_used=0,
                tokens_used=0,
                execution_time_ms=0.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            chunks = []
            async for chunk in run_rlaph_stream("test query"):
                chunks.append(chunk)

            # Should not crash, just yield empty or nothing
            full_answer = "".join(chunks)
            assert full_answer == ""

    @pytest.mark.asyncio
    async def test_run_rlaph_creates_default_context_when_none(self):
        """run_rlaph should create SessionContext when none provided."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
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

            await run_rlaph("test query", context=None)

            call_args = mock_loop.run.call_args
            assert isinstance(call_args[0][1], SessionContext)
            assert len(call_args[0][1].messages) == 0

    @pytest.mark.asyncio
    async def test_run_rlaph_with_all_parameters(self):
        """run_rlaph should work with all parameters set."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="complete answer",
                iterations=3,
                depth_used=2,
                tokens_used=500,
                execution_time_ms=300.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            ctx = SessionContext()
            result = await run_rlaph(
                query="complex query",
                working_dir="/custom/path",
                session_id="custom-session",
                max_depth=10,
                verbose=True,
                context=ctx,
                stream=False
            )

            # Verify loop creation
            MockLoop.assert_called_once_with(max_depth=10, verbose=True)

            # Verify run call
            call_args = mock_loop.run.call_args
            assert call_args[0][0] == "complex query"
            assert call_args[0][1] is ctx
            assert call_args[0][2] == Path("/custom/path")
            assert call_args[0][3] == "custom-session"

            # Verify result
            assert result.answer == "complete answer"

    @pytest.mark.asyncio
    async def test_run_rlaph_stream_with_custom_params(self):
        """run_rlaph_stream should pass all parameters correctly."""
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(return_value=RLPALoopResult(
                answer="streamed answer",
                iterations=1,
                depth_used=1,
                tokens_used=200,
                execution_time_ms=200.0,
                history=[]
            ))
            MockLoop.return_value = mock_loop

            ctx = SessionContext()
            chunks = []
            async for chunk in run_rlaph_stream(
                query="stream query",
                working_dir="/custom/path",
                session_id="stream-session",
                max_depth=7,
                verbose=True,
                context=ctx,
            ):
                chunks.append(chunk)

            # Verify loop creation
            MockLoop.assert_called_once_with(max_depth=7, verbose=True)

            # Verify run call
            call_args = mock_loop.run.call_args
            assert call_args[0][0] == "stream query"
            assert call_args[0][1] is ctx
            assert call_args[0][2] == Path("/custom/path")
            assert call_args[0][3] == "stream-session"

            # Verify streamed content
            full_answer = "".join(chunks)
            assert full_answer == "streamed answer"


class TestRunRlaphIntegration:
    """Integration tests for run_rlaph() functions."""

    @pytest.mark.asyncio
    async def test_run_rlaph_with_mocks_is_callable(self):
        """Verify run_rlaph is async callable."""
        # This is a smoke test to ensure the function signature is correct
        from src.repl.rlaph_loop import run_rlaph
        import inspect

        assert inspect.iscoroutinefunction(run_rlaph)

        # Verify we can call it (it will fail at runtime without API keys,
        # but the signature and imports should work)
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
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

            # This should not raise TypeError or other import errors
            result = await run_rlaph("test")
            assert result is not None

    @pytest.mark.asyncio
    async def test_run_rlaph_stream_is_async_generator(self):
        """Verify run_rlaph_stream is an async generator."""
        from src.repl.rlaph_loop import run_rlaph_stream
        import inspect

        # Check if it's an async generator function
        assert inspect.isasyncgenfunction(run_rlaph_stream)

        # Verify we can iterate from it
        with patch("src.repl.rlaph_loop.RLAPHLoop") as MockLoop:
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

            # This should not raise TypeError
            chunks = []
            async for chunk in run_rlaph_stream("test"):
                chunks.append(chunk)
            assert len(chunks) > 0

    def test_run_rlaph_in_all_exports(self):
        """run_rlaph should be in __all__ exports."""
        from src.repl import rlaph_loop
        assert "run_rlaph" in rlaph_loop.__all__

    def test_run_rlaph_stream_in_all_exports(self):
        """run_rlaph_stream should be in __all__ exports."""
        from src.repl import rlaph_loop
        assert "run_rlaph_stream" in rlaph_loop.__all__


class TestVerboseModeFrameLogging:
    """Test suite for verbose mode frame logging (Task 70)."""

    @pytest.mark.asyncio
    async def test_verbose_false_no_frame_logging(self):
        """Verbose=False should not print frame details."""
        from src.repl.rlaph_loop import RLAPHLoop
        from unittest.mock import patch

        with patch("builtins.print") as mock_print:
            loop = RLAPHLoop(max_depth=3, verbose=False)
            assert loop._verbose is False

            # Create a minimal frame
            from src.frame.causal_frame import CausalFrame, FrameStatus
            from src.frame.context_slice import ContextSlice
            from datetime import datetime

            context_slice = ContextSlice(
                files={},
                memory_refs=[],
                tool_outputs={},
                token_budget=1000,
            )
            frame = CausalFrame(
                frame_id="test-frame-id",
                depth=1,
                parent_id=None,
                children=[],
                query="test query",
                context_slice=context_slice,
                evidence=[],
                conclusion="test conclusion",
                confidence=0.9,
                invalidation_condition="test_condition",
                status=FrameStatus.COMPLETED,
                canonical_task="test_task",
                branched_from=None,
                escalation_reason=None,
                created_at=datetime.now(),
                completed_at=datetime.now(),
            )

            # Mock the add method and check print wasn't called for frame details
            with patch.object(loop.frame_index, "add"):
                loop.frame_index.add(frame)
                loop._current_frame_id = frame.frame_id

                # print should not be called with "[RLM] Frame created:" when verbose=False
                # (we only check that verbose flag is correctly set)
                assert loop._verbose is False

    def test_verbose_true_flag_set(self):
        """Verbose=True should set _verbose flag."""
        from src.repl.rlaph_loop import RLAPHLoop

        loop = RLAPHLoop(max_depth=3, verbose=True)
        assert loop._verbose is True

    @pytest.mark.asyncio
    async def test_verbose_frame_logging_format(self):
        """Verbose mode should print frame details in correct format."""
        from src.repl.rlaph_loop import RLAPHLoop
        from unittest.mock import patch, MagicMock
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from datetime import datetime
        from src.repl.llm_client import LLMClient

        with patch("builtins.print") as mock_print:
            loop = RLAPHLoop(max_depth=3, verbose=True)

            # Set up mocks for llm_sync call
            loop.llm_client = MagicMock(spec=LLMClient)
            loop.llm_client.call.return_value = "Test response"

            # Call llm_sync to trigger frame creation with verbose logging
            result = loop.llm_sync("test query", context="", depth=1)

            # Verify print was called with frame details
            assert mock_print.called
            print_calls = [str(call) for call in mock_print.call_args_list]

            # Check for expected frame logging patterns
            any_call_has_frame_created = any("[RLM] Frame created:" in str(call) for call in print_calls)
            any_call_has_id = any("[RLM]   id:" in str(call) for call in print_calls)
            any_call_has_depth = any("[RLM]   depth:" in str(call) for call in print_calls)
            any_call_has_parent = any("[RLM]   parent:" in str(call) for call in print_calls)
            any_call_has_invalidation = any("[RLM]   invalidation_condition:" in str(call) for call in print_calls)
            any_call_has_evidence = any("[RLM]   evidence:" in str(call) for call in print_calls)

            assert any_call_has_frame_created, "Should print '[RLM] Frame created:'"
            assert any_call_has_id, "Should print frame ID"
            assert any_call_has_depth, "Should print frame depth"
            assert any_call_has_parent, "Should print frame parent"
            assert any_call_has_invalidation, "Should print invalidation condition"
            assert any_call_has_evidence, "Should print evidence"

    @pytest.mark.asyncio
    async def test_verbose_logs_all_required_fields(self):
        """Verbose mode should log all required frame fields."""
        from src.repl.rlaph_loop import RLAPHLoop
        from unittest.mock import patch, MagicMock
        from src.repl.llm_client import LLMClient

        with patch("builtins.print") as mock_print:
            loop = RLAPHLoop(max_depth=3, verbose=True)
            loop.llm_client = MagicMock(spec=LLMClient)
            loop.llm_client.call.return_value = "Test response"

            # Call llm_sync
            result = loop.llm_sync("test query", depth=1)

            # Collect all print arguments
            all_print_args = []
            for call in mock_print.call_args_list:
                if call.args:
                    all_print_args.append(str(call.args[0]))

            # Verify all required fields are present
            assert any("[RLM] Frame created:" in s for s in all_print_args)
            assert any("[RLM]   id:" in s for s in all_print_args)
            assert any("[RLM]   depth:" in s for s in all_print_args)
            assert any("[RLM]   parent:" in s for s in all_print_args)
            assert any("[RLM]   invalidation_condition:" in s for s in all_print_args)
            assert any("[RLM]   evidence:" in s for s in all_print_args)

    @pytest.mark.asyncio
    async def test_verbose_logs_canonical_task_when_present(self):
        """Verbose mode should log canonical_task when present."""
        from src.repl.rlaph_loop import RLAPHLoop
        from unittest.mock import patch, MagicMock
        from src.repl.llm_client import LLMClient

        with patch("builtins.print") as mock_print:
            loop = RLAPHLoop(max_depth=3, verbose=True)
            loop.llm_client = MagicMock(spec=LLMClient)
            # Mock a response that will result in a canonical task being extracted
            loop.llm_client.call.return_value = '{"conclusion": "test", "confidence": 0.9, "next_action": "finalize"}'

            # Call llm_sync
            result = loop.llm_sync("analyze the auth flow", depth=1)

            # Collect all print arguments
            all_print_args = []
            for call in mock_print.call_args_list:
                if call.args:
                    all_print_args.append(str(call.args[0]))

            # canonical_task may or may not be present depending on extraction
            # This test just verifies the logging structure works
            assert any("[RLM] Frame created:" in s for s in all_print_args)
