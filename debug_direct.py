#!/usr/bin/env python3
"""Debug - show actual REPL output."""
import asyncio
from pathlib import Path
from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext
from src.repl_environment import RLMEnvironment

async def debug_repl_output():
    # Create REPL directly
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    repl = RLMEnvironment(context)
    repl.enable_file_access(working_dir=Path("."))

    # Run the exact search code
    code = '''
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

print("Search complete")
'''

    print("Executing code directly in REPL...")
    print("=" * 50)

    result = repl.execute(code)

    print("SUCCESS:", result.success)
    print("OUTPUT:")
    print(result.output)
    print("ERROR:", result.error)

asyncio.run(debug_repl_output())
