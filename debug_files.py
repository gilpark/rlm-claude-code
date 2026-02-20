#!/usr/bin/env python3
"""Debug file reading in REPL."""
import asyncio
from pathlib import Path
from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext

async def debug_files():
    loop = RLAPHLoop(max_iterations=5, max_depth=1)
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})

    # Test file reading
    query = """
Use the glob_files function to find Python files in src/.

Then read the first few lines of src/__init__.py using read_file().

Show me what you find. End with FINAL: <first line of the file>
"""

    print("Testing file reading...")
    print("=" * 50)

    result = await loop.run(query, context, working_dir=Path("."))

    print("=" * 50)
    print(f"Iterations: {result.iterations}")
    print(f"Answer: {result.answer}")
    print(f"History: {result.history}")

asyncio.run(debug_files())
