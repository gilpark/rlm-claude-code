#!/usr/bin/env python
"""
Test script for RLAPHLoop.

Run with: .venv/bin/python test_rlaph_loop.py
"""

import asyncio
from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext, Message, MessageRole


async def test_basic_query():
    """Test a basic query without complex context."""
    print("=" * 60)
    print("Test 1: Basic query")
    print("=" * 60)

    # Create context
    ctx = SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="Hello, can you help me?"),
        ],
        files={},
        tool_outputs=[],
        working_memory={},
    )

    print(f"Context tokens: {ctx.total_tokens}")

    # Create loop
    loop = RLAPHLoop(max_iterations=5, max_depth=2)
    print(f"Max iterations: {loop.max_iterations}")
    print(f"Max depth: {loop.max_depth}")

    # Run query
    print("\nRunning query...")
    result = await loop.run("What is 2 + 2?", ctx)

    print(f"\nAnswer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print(f"Depth used: {result.depth_used}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
    print(f"History: {result.history}")


async def test_with_files():
    """Test with file context."""
    print("\n" + "=" * 60)
    print("Test 2: Query with file context")
    print("=" * 60)

    # Create context with a file
    ctx = SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="Analyze this code"),
        ],
        files={
            "example.py": '''
def greet(name):
    """Greet someone."""
    return f"Hello, {name}!"

def add(a, b):
    """Add two numbers."""
    return a + b
''',
        },
        tool_outputs=[],
        working_memory={},
    )

    print(f"Context tokens: {ctx.total_tokens}")
    print(f"Files: {list(ctx.files.keys())}")

    # Create loop
    loop = RLAPHLoop(max_iterations=10, max_depth=2)

    # Run query
    print("\nRunning query...")
    result = await loop.run("What functions are defined in the files?", ctx)

    print(f"\nAnswer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print(f"Tokens used: {result.tokens_used}")


async def test_with_tool_outputs():
    """Test with tool output context."""
    print("\n" + "=" * 60)
    print("Test 3: Query with tool outputs")
    print("=" * 60)

    from src.types import ToolOutput

    ctx = SessionContext(
        messages=[
            Message(role=MessageRole.USER, content="What happened?"),
        ],
        files={},
        tool_outputs=[
            ToolOutput(
                tool_name="bash",
                content="npm install completed successfully\nadded 125 packages",
                exit_code=0,
            ),
        ],
        working_memory={},
    )

    print(f"Context tokens: {ctx.total_tokens}")
    print(f"Tool outputs: {len(ctx.tool_outputs)}")

    # Create loop
    loop = RLAPHLoop(max_iterations=5, max_depth=2)

    # Run query
    print("\nRunning query...")
    result = await loop.run("Summarize what the tools did", ctx)

    print(f"\nAnswer: {result.answer}")
    print(f"Iterations: {result.iterations}")


async def main():
    """Run all tests."""
    print("RLAPHLoop Test Suite")
    print("=" * 60)

    # Check if we have API access
    try:
        from src.config import default_config
        print(f"Config loaded: {type(default_config)}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")

    # Run tests
    try:
        await test_basic_query()
    except Exception as e:
        print(f"Test 1 failed: {e}")

    # Uncomment to run more tests:
    # await test_with_files()
    # await test_with_tool_outputs()


if __name__ == "__main__":
    asyncio.run(main())
