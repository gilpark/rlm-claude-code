#!/usr/bin/env python3
"""Debug - show LLM I/O."""
import asyncio
from pathlib import Path
from src.rlaph_loop import RLAPHLoop, RLPALoopState
from src.types import SessionContext
from src.response_parser import ResponseParser

async def debug_llm_io():
    loop = RLAPHLoop(max_iterations=3, max_depth=1)
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})

    # Monkey-patch to capture I/O
    original_call = loop.llm_client.call
    calls = []

    def tracked_call(**kwargs):
        result = original_call(**kwargs)
        calls.append({"input": kwargs.get("query", "")[:500], "output": result[:500]})
        return result

    loop.llm_client.call = tracked_call

    query = """
Find the line with hash d4fdb540aa5f2d34 in src/*.py files.

```python
target = "d4fdb540aa5f2d34"
files = glob_files("src/*.py")
for filepath in files:
    content = read_file(filepath)
    lines = content.split("\\n")
    for i, line in enumerate(lines):
        h = hashlib.sha256(line.strip().encode()).hexdigest()[:16]
        if h == target:
            print(f"MATCH: {filepath}:{i+1}")
            print(f"LINE: {line}")
```

Say FINAL: <filepath>:<line> - <content>
"""

    result = await loop.run(query, context, working_dir=Path("."))

    print("=" * 60)
    print(f"LLM CALLS: {len(calls)}")
    for i, call in enumerate(calls):
        print(f"\n--- CALL {i+1} INPUT (last 500 chars) ---")
        print(call["input"][-500:])
        print(f"\n--- CALL {i+1} OUTPUT (first 500 chars) ---")
        print(call["output"][:500])
    print("=" * 60)
    print(f"FINAL ANSWER: {result.answer}")

asyncio.run(debug_llm_io())
