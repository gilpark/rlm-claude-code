#!/usr/bin/env python3
"""Debug orchestrator step by step."""
import asyncio
from pathlib import Path
from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext

async def debug_run():
    loop = RLAPHLoop(max_iterations=3, max_depth=1)
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})

    # Simple test that requires actual code execution
    query = """
Execute this exact code and show me the result:

```python
x = 10 + 5
print(f"The answer is {x}")
```

After seeing the execution result, say FINAL: <what you saw>
"""

    print("Starting RLAPH loop...")
    print("=" * 50)

    result = await loop.run(query, context)

    print("=" * 50)
    print(f"Iterations: {result.iterations}")
    print(f"Answer: {result.answer}")
    print(f"History: {result.history}")

asyncio.run(debug_run())
