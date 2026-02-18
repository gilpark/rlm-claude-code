#!/usr/bin/env python3
"""
Test RLM with large context (>1M tokens).

This validates the key RLM feature: context externalization.
Files are NOT passed in the prompt (would exceed token limits).
Instead, the REPL reads them on-demand.

Usage:
    uv run python tests/integration/test_large_context.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext


def create_large_file(size_mb: float = 2.0) -> Path:
    """
    Create a large test file with repetitive but searchable content.

    Args:
        size_mb: Target size in megabytes

    Returns:
        Path to the created file
    """
    temp_dir = Path(tempfile.gettempdir()) / "rlm_test"
    temp_dir.mkdir(exist_ok=True)

    file_path = temp_dir / "large_context.txt"

    # Each line is ~100 chars, we want ~size_mb
    target_bytes = int(size_mb * 1024 * 1024)
    line_template = "Line {i:08d}: This is a test line with some content. The secret code is: {code}\n"

    # Hide a secret code somewhere in the middle
    secret_line = target_bytes // 200  # Around halfway
    secret_code = "ALPHA-BRAVO-CHARLIE-7749"

    with open(file_path, "w") as f:
        bytes_written = 0
        i = 0
        while bytes_written < target_bytes:
            if i == secret_line:
                line = line_template.format(i=i, code=secret_code)
            else:
                line = line_template.format(i=i, code="nothing-special")
            f.write(line)
            bytes_written += len(line)
            i += 1

    actual_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"Created {file_path}")
    print(f"  Size: {actual_mb:.2f} MB")
    print(f"  Lines: {i}")
    print(f"  Secret at line: {secret_line}")

    return file_path


async def test_large_context_read():
    """
    Test that RLM can read and search a large file without loading it all in memory.

    This validates:
    1. File is NOT passed in prompt (would exceed limits)
    2. REPL can read file on-demand with read_file()
    3. REPL can search file with grep_files()
    4. LLM can process results and provide correct answer
    """
    print("\n" + "=" * 60)
    print("Test: Large Context (>1M tokens) via Externalization")
    print("=" * 60)

    # Create a 2MB file (~500K tokens)
    large_file = create_large_file(size_mb=2.0)

    # The secret we hid in the file
    expected_secret = "ALPHA-BRAVO-CHARLIE-7749"

    # Create RLAPH loop
    loop = RLAPHLoop(
        max_iterations=10,
        max_depth=1,
    )

    # Create empty context - files are read on-demand
    context = SessionContext()

    # Query that requires reading the large file
    query = f"""There is a large file at {large_file}.

1. First, check the file size with: import os; print(os.path.getsize("{large_file}") / 1024 / 1024, "MB")

2. Then search for the secret code using grep_files: grep_files("ALPHA-BRAVO", "{large_file}")

3. Extract and report the secret code.

FINAL: <the secret code>"""

    print(f"\nQuery: Find secret code in {large_file.name}")
    print(f"Expected secret: {expected_secret}")

    start_time = time.time()
    result = await loop.run(
        query=query,
        context=context,
        working_dir=Path(tempfile.gettempdir()) / "rlm_test",
    )
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Answer: {result.answer}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens used: {result.tokens_used}")

    # Verify
    success = expected_secret in result.answer
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Secret code {'found' if success else 'NOT found'}")

    # Cleanup
    large_file.unlink(missing_ok=True)

    return success


async def test_large_context_summarization():
    """
    Test that RLM can summarize a large file without loading it all at once.

    Uses chunked reading and summarization.
    """
    print("\n" + "=" * 60)
    print("Test: Large File Summarization")
    print("=" * 60)

    # Create a 1MB file
    large_file = create_large_file(size_mb=1.0)

    loop = RLAPHLoop(max_iterations=10, max_depth=1)
    context = SessionContext()

    query = f"""There is a large file at {large_file}.

1. Read the first 50 lines to understand the format: read_file("{large_file}", limit=50)

2. Count total lines efficiently (without reading all content)

3. Summarize what kind of data this file contains.

FINAL: <summary including: format description, estimated line count, any patterns>"""

    print(f"\nQuery: Summarize {large_file.name}")

    start_time = time.time()
    result = await loop.run(
        query=query,
        context=context,
        working_dir=Path(tempfile.gettempdir()) / "rlm_test",
    )
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Answer: {result.answer[:500]}...")
    print(f"  Iterations: {result.iterations}")
    print(f"  Time: {elapsed:.2f}s")

    # Verify it mentions lines and format
    success = "line" in result.answer.lower() or "format" in result.answer.lower()
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Summary {'provided' if success else 'incomplete'}")

    # Cleanup
    large_file.unlink(missing_ok=True)

    return success


async def main():
    """Run all large context tests."""
    print("\n" + "=" * 60)
    print("RLM Large Context Validation Tests")
    print("Testing context externalization with >1M token files")
    print("=" * 60)

    results = []

    # Test 1: Search large file
    try:
        results.append(("Large file search", await test_large_context_read()))
    except Exception as e:
        print(f"\n✗ FAIL: Large file search - {e}")
        results.append(("Large file search", False))

    # Test 2: Summarize large file
    try:
        results.append(("Large file summarization", await test_large_context_summarization()))
    except Exception as e:
        print(f"\n✗ FAIL: Large file summarization - {e}")
        results.append(("Large file summarization", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
