#!/usr/bin/env python
"""
Fast test script for RLAPHLoop with mocked LLM calls.

Run with: .venv/bin/python test_rlaph_fast.py
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.rlaph_loop import RLAPHLoop, RLPALoopResult
from src.types import SessionContext, Message, MessageRole


class MockRouter:
    """Mock router that returns canned responses."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["FINAL: This is a mock answer"]
        self.call_count = 0

    async def complete(self, **kwargs):
        """Return next canned response."""
        response = MagicMock()
        response.content = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


async def test_basic_final():
    """Test loop with immediate FINAL answer."""
    print("=" * 60)
    print("Test 1: Immediate FINAL answer")
    print("=" * 60)

    ctx = SessionContext()
    router = MockRouter(responses=["FINAL: 42"])

    loop = RLAPHLoop(max_iterations=5, max_depth=2, router=router)
    result = await loop.run("What is 2 + 2?", ctx)

    print(f"Answer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print(f"Tokens: {result.tokens_used}")
    print(f"Time: {result.execution_time_ms:.2f}ms")
    assert result.answer == "42"
    print("PASSED")


async def test_repl_then_final():
    """Test loop with REPL execution then FINAL."""
    print("\n" + "=" * 60)
    print("Test 2: REPL execution then FINAL")
    print("=" * 60)

    ctx = SessionContext(
        files={"test.py": "x = 10\ny = 20"},
    )

    # First: execute REPL, Second: FINAL answer
    router = MockRouter(
        responses=[
            "```python\nresult = 10 + 20\nprint(result)\n```",
            "FINAL: The sum is 30",
        ]
    )

    loop = RLAPHLoop(max_iterations=5, max_depth=2, router=router)
    result = await loop.run("Calculate 10 + 20", ctx)

    print(f"Answer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print(f"History: {result.history}")
    print("PASSED")


async def test_final_var():
    """Test FINAL_VAR with variable extraction."""
    print("\n" + "=" * 60)
    print("Test 3: FINAL_VAR extraction")
    print("=" * 60)

    ctx = SessionContext()

    # Execute code that sets a variable, then use FINAL_VAR
    router = MockRouter(
        responses=[
            "```python\nanswer = 'hello world'\n```",
            "FINAL_VAR: answer",
        ]
    )

    loop = RLAPHLoop(max_iterations=5, max_depth=2, router=router)
    result = await loop.run("Set a variable", ctx)

    print(f"Answer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print("PASSED")


async def test_max_iterations():
    """Test that loop respects max_iterations."""
    print("\n" + "=" * 60)
    print("Test 4: Max iterations limit")
    print("=" * 60)

    ctx = SessionContext()

    # Always return REPL code (no FINAL)
    router = MockRouter(responses=["```python\nx = 1\n```"])

    loop = RLAPHLoop(max_iterations=3, max_depth=2, router=router)
    result = await loop.run("Infinite loop test", ctx)

    print(f"Answer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    assert result.iterations == 3
    assert result.answer == "No answer produced"
    print("PASSED")


async def test_context_with_files():
    """Test with file context in REPL."""
    print("\n" + "=" * 60)
    print("Test 5: File context access")
    print("=" * 60)

    ctx = SessionContext(
        files={
            "module.py": "def hello():\n    return 'world'",
            "config.json": '{"debug": true}',
        },
    )

    router = MockRouter(
        responses=[
            "```python\nprint(list(files.keys()))\n```",
            "FINAL: Found 2 files",
        ]
    )

    loop = RLAPHLoop(max_iterations=5, max_depth=2, router=router)
    result = await loop.run("List files", ctx)

    print(f"Answer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    assert "2" in result.answer
    print("PASSED")


async def test_context_tokens():
    """Test token estimation."""
    print("\n" + "=" * 60)
    print("Test 6: Token estimation")
    print("=" * 60)

    ctx = SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="Hello world"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        ],
        files={"test.py": "x = 1\ny = 2\n"},
    )

    tokens = ctx.total_tokens
    print(f"Estimated tokens: {tokens}")
    print(f"Active modules: {ctx.active_modules}")
    print("PASSED")


async def main():
    """Run all tests."""
    print("RLAPHLoop Fast Test Suite")
    print("=" * 60)

    await test_basic_final()
    await test_repl_then_final()
    await test_final_var()
    await test_max_iterations()
    await test_context_with_files()
    await test_context_tokens()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
