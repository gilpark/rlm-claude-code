"""
Test script demonstrating llm() function in RLM REPL.

This demonstrates how the llm() function works within the RLAPH loop.
"""

import asyncio
from src.types import SessionContext
from src.rlaph_loop import RLAPHLoop


async def test_llm_function():
    """Test the llm() function in REPL."""
    # Create a simple context
    context = SessionContext(
        messages=[],
        files={},
        tool_outputs=[],
        working_memory={},
    )

    # Create RLAPH loop
    loop = RLAPHLoop(max_depth=3)

    # The query will use llm() in the REPL
    query = """Use the Python REPL to call llm() and return the result.

```python
result = llm("What is 3+3? Just say the number.")
print(result)
```

Then use FINAL_VAR: result"""

    print("Running RLAPH loop with llm() query...")
    print(f"Query: {query}")
    print("-" * 60)

    result = await loop.run(query, context)

    print("-" * 60)
    print(f"Answer: {result.answer}")
    print(f"Iterations: {result.iterations}")
    print(f"Tokens used: {result.tokens_used}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
    print(f"History: {result.history}")


if __name__ == "__main__":
    asyncio.run(test_llm_function())
