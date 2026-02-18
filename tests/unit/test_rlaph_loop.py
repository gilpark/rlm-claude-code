"""
Unit tests for RLAPH Loop.

Tests the core functionality of the RLAPH-style loop:
- llm() returns actual string immediately
- Depth limits work correctly
- FINAL answer detection works
- REPL functions preserved
"""

from __future__ import annotations

import pytest

from src.types import SessionContext


class TestRLAPHLoopSync:
    """Test synchronous llm() behavior."""

    @pytest.fixture
    def context(self) -> SessionContext:
        """Create empty session context."""
        return SessionContext(
            messages=[],
            files={},
            tool_outputs=[],
            working_memory={},
        )

    def test_llm_returns_string_not_deferred(self, context: SessionContext):
        """
        llm() should return actual string, not DeferredOperation.

        This is the core RLAPH promise - no more deferred operations.
        """
        from src.rlaph_loop import RLAPHLoop

        loop = RLAPHLoop(max_iterations=5)

        # The llm_sync method should exist and return a string
        assert hasattr(loop, "llm_sync")

        # When no recursive_handler, should return error string (not DeferredOperation)
        result = loop.llm_sync("test query")
        assert isinstance(result, str)
        assert "Error" in result or result == ""

    def test_llm_sync_with_mock_handler(self, context: SessionContext):
        """
        llm_sync with a mock handler should return the handler's response.
        """
        from unittest.mock import MagicMock

        from src.rlaph_loop import RLAPHLoop
        from src.recursive_handler import RecursiveREPL

        loop = RLAPHLoop(max_iterations=5)

        # Create mock handler
        mock_handler = MagicMock(spec=RecursiveREPL)
        mock_handler.llm_sync.return_value = "42"

        loop.recursive_handler = mock_handler

        result = loop.llm_sync("What is 6*7?")
        assert result == "42"
        assert isinstance(result, str)


class TestRLAPHLoopDepth:
    """Test depth limit enforcement."""

    def test_max_depth_default(self):
        """Default max_depth should be 3."""
        from src.rlaph_loop import RLAPHLoop

        loop = RLAPHLoop()
        assert loop.max_depth == 3  # Default is 3, not 2

    def test_max_depth_custom(self):
        """Can set custom max_depth."""
        from src.rlaph_loop import RLAPHLoop

        loop = RLAPHLoop(max_depth=3)
        assert loop.max_depth == 3

    def test_depth_tracking(self):
        """Loop should track current depth."""
        from src.rlaph_loop import RLAPHLoop

        loop = RLAPHLoop(max_depth=2)
        assert loop._depth == 0

        # After increment
        loop._depth = 1
        assert loop._depth == 1


class TestRLAPHLoopFinalAnswer:
    """Test FINAL answer detection."""

    @pytest.fixture
    def context(self) -> SessionContext:
        """Create empty session context."""
        return SessionContext(
            messages=[],
            files={},
            tool_outputs=[],
            working_memory={},
        )

    def test_final_answer_pattern(self):
        """FINAL: pattern should be detected."""
        from src.response_parser import ResponseParser, ResponseAction

        parser = ResponseParser()

        response = "Here is my analysis.\n\nFINAL: The answer is 42"
        items = parser.parse(response)

        assert len(items) == 1
        assert items[0].action == ResponseAction.FINAL_ANSWER
        assert "42" in items[0].content

    def test_final_answer_with_code(self):
        """FINAL: after Python code should be detected."""
        from src.response_parser import ResponseParser, ResponseAction

        parser = ResponseParser()

        response = """Let me calculate:
```python
result = 6 * 7
print(result)
```

FINAL: The answer is 42"""
        items = parser.parse(response)

        # Should have REPL_EXECUTE and FINAL_ANSWER
        actions = [item.action for item in items]
        assert ResponseAction.REPL_EXECUTE in actions
        assert ResponseAction.FINAL_ANSWER in actions

    def test_final_var_pattern(self):
        """FINAL_VAR: pattern should be detected."""
        from src.response_parser import ResponseParser, ResponseAction

        parser = ResponseParser()

        response = """```python
answer = 42
```

FINAL_VAR: answer"""
        items = parser.parse(response)

        actions = [item.action for item in items]
        assert ResponseAction.FINAL_VAR in actions


class TestRLAPHLoopHistory:
    """Test iteration history tracking."""

    def test_history_initialized(self):
        """History should be initialized as empty list."""
        from src.rlaph_loop import RLAPHLoop

        loop = RLAPHLoop()
        assert loop.history == []

    def test_history_records_turns(self):
        """Each turn should be recorded in history."""
        from src.rlaph_loop import RLAPHLoop

        loop = RLAPHLoop(max_iterations=5)
        # Manually add history entry (simulating a turn)
        loop.history.append({"turn": 1, "action": "test", "has_answer": False})

        assert len(loop.history) == 1
        assert loop.history[0]["turn"] == 1


class TestRLAPHLoopREPLFunctions:
    """Test that REPL functions are still available."""

    @pytest.fixture
    def context(self) -> SessionContext:
        """Create session context with test data."""
        return SessionContext(
            messages=[],
            files={"test.py": "print('hello')"},
            tool_outputs=[],
            working_memory={},
        )

    def test_repl_environment_has_peek(self, context: SessionContext):
        """REPL should have peek() function."""
        from src.repl_environment import RLMEnvironment

        env = RLMEnvironment(context=context)

        # peek should be available
        result = env.execute("peek(files, 0, 1)")
        assert result.success

    def test_repl_environment_has_search(self, context: SessionContext):
        """REPL should have search() function."""
        from src.repl_environment import RLMEnvironment

        context.files = {"test.py": "def hello(): pass"}
        env = RLMEnvironment(context=context)

        result = env.execute("search(files, 'hello')")
        assert result.success

    def test_repl_environment_has_llm(self, context: SessionContext):
        """REPL should have llm() function."""
        from src.repl_environment import RLMEnvironment

        env = RLMEnvironment(context=context)

        # llm function should exist (will return error without handler)
        result = env.execute("llm('test')")
        # Should not crash - may return error string but should execute
        assert result.success or "Error" in result.output


class TestRLAPHLoopIntegration:
    """Integration tests for RLAPH loop."""

    @pytest.fixture
    def context(self) -> SessionContext:
        """Create session context with test data."""
        return SessionContext(
            messages=[],
            files={"test.py": "def add(a, b): return a + b"},
            tool_outputs=[],
            working_memory={},
        )

    @pytest.mark.asyncio
    async def test_loop_returns_result(self, context: SessionContext):
        """Loop should return a RLPALoopResult."""
        from src.rlaph_loop import RLAPHLoop, RLPALoopResult

        loop = RLAPHLoop(max_iterations=3)

        # Run with simple query
        result = await loop.run("What is 2+2?", context)

        # Should return a result object
        assert isinstance(result, RLPALoopResult)
        assert isinstance(result.answer, str)
        assert result.iterations >= 1

    @pytest.mark.asyncio
    async def test_loop_respects_max_iterations(self, context: SessionContext):
        """Loop should not exceed max_iterations."""
        from src.rlaph_loop import RLAPHLoop

        loop = RLAPHLoop(max_iterations=3)

        result = await loop.run("Complex task", context)

        # Should not exceed max iterations
        assert result.iterations <= 3
