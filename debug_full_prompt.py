#!/usr/bin/env python3
"""Debug - show full prompt."""
import asyncio
from pathlib import Path
from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext

async def debug_full():
    loop = RLAPHLoop(max_iterations=2, max_depth=1)
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})

    query = """
Find the line with hash d4fdb540aa5f2d34.

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
```

FINAL: <result>
"""

    # Patch to show FULL prompt
    original_call = loop.llm_client.call
    call_num = 0

    def tracked_call(**kwargs):
        nonlocal call_num
        call_num += 1
        print(f"\n{'='*60}")
        print(f"CALL {call_num} FULL PROMPT:")
        print(kwargs.get('query', ''))
        print(f"{'='*60}")
        return original_call(**kwargs)

    loop.llm_client.call = tracked_call

    result = await loop.run(query, context, working_dir=Path("."))

asyncio.run(debug_full())
