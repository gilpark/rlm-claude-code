#!/usr/bin/env python3
"""Debug needle search - properly structured."""
import asyncio
from pathlib import Path
from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext

async def debug_needle():
    loop = RLAPHLoop(max_iterations=10, max_depth=1)
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})

    # Properly structured needle search
    query = """
I need you to find a specific line in the src/ Python files.

The target hash is: d4fdb540aa5f2d34

This is the SHA256 hash (first 16 chars) of a stripped line.

Execute this code step by step:

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

You MUST execute the code above and show me what prints.
Then say FINAL: <filepath>:<line_number> - <the line content>
"""

    print("Testing needle search...")
    print("=" * 50)

    result = await loop.run(query, context, working_dir=Path("."))

    print("=" * 50)
    print(f"Iterations: {result.iterations}")
    print(f"Answer: {result.answer}")

asyncio.run(debug_needle())
