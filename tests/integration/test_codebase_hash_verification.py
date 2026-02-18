#!/usr/bin/env python3
"""
Cryptographic verification: RLM must find and hash random lines from actual codebase.

The idea:
1. Pick random lines from random Python files
2. Compute their SHA256 hashes (ground truth)
3. Ask RLM to find those lines and compute their hashes
4. Compare hashes - they MUST match exactly

This is cryptographic proof - you cannot guess a SHA256 hash.

Usage:
    uv run python tests/integration/test_codebase_hash_verification.py
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext

PROJECT_ROOT = Path(__file__).parent.parent.parent


def pick_random_lines(num_lines: int = 5) -> list[dict]:
    """
    Pick random lines from random Python files in the codebase.

    Returns:
        List of dicts with: file, line_number, line_content, hash
    """
    src_dir = PROJECT_ROOT / "src"
    py_files = list(src_dir.rglob("*.py"))

    # Filter out __pycache__ and empty files
    py_files = [f for f in py_files if "__pycache__" not in str(f) and f.stat().st_size > 100]

    selected = []

    for _ in range(num_lines):
        # Pick random file
        file_path = random.choice(py_files)

        # Read all lines
        with open(file_path) as f:
            lines = f.readlines()

        # Pick random non-empty, non-comment line
        valid_lines = [
            (i, line.strip())
            for i, line in enumerate(lines)
            if line.strip() and not line.strip().startswith("#") and len(line.strip()) > 20
        ]

        if not valid_lines:
            continue

        line_num, line_content = random.choice(valid_lines)

        # Compute hash
        line_hash = hashlib.sha256(line_content.encode()).hexdigest()[:16]

        selected.append({
            "file": str(file_path.relative_to(PROJECT_ROOT)),
            "line_number": line_num + 1,  # 1-indexed
            "line_content": line_content,
            "hash": line_hash,
        })

    return selected


async def test_find_and_hash_lines():
    """
    Test: RLM must find specific lines and compute their hashes.

    This proves RLM actually reads files because:
    - Hashes cannot be guessed (cryptographically impossible)
    - Must find exact line at exact location
    - Must compute correct hash
    """
    print("\n" + "=" * 60)
    print("Cryptographic Verification: Find & Hash Random Lines")
    print("=" * 60)

    # Pick 3 random lines
    random.seed(42)  # Reproducible
    targets = pick_random_lines(3)

    print("\nGround truth (hidden from RLM):")
    for i, t in enumerate(targets, 1):
        print(f"  {i}. {t['file']}:{t['line_number']}")
        print(f"     Line: {t['line_content'][:50]}...")
        print(f"     Hash: {t['hash']}")
    print()

    loop = RLAPHLoop(max_iterations=10, max_depth=1)
    context = SessionContext()

    # Build query with file paths and line numbers
    query_parts = ["Find these specific lines and compute their SHA256 hash (first 16 chars):"]
    query_parts.append("")

    for i, t in enumerate(targets, 1):
        query_parts.append(
            f"{i}. File: {t['file']}, Line number: {t['line_number']}"
        )

    query_parts.append("")
    query_parts.append("For each line, compute the hash like this:")
    query_parts.append("NOTE: hashlib is already available - do NOT import it.")
    query_parts.append("```python")
    query_parts.append("content = read_file('FILE_PATH', limit=LINE_NUM+1)")
    query_parts.append("lines = content.split('\\n')")
    query_parts.append("line = lines[LINE_NUM-1].strip()  # 0-indexed")
    query_parts.append("h = hashlib.sha256(line.encode()).hexdigest()[:16]")
    query_parts.append("print(f'Hash: {h}')")
    query_parts.append("```")
    query_parts.append("")
    query_parts.append("FINAL: <hash1>, <hash2>, <hash3>")

    query = "\n".join(query_parts)

    print("Running RLM query...")
    result = await loop.run(query=query, context=context, working_dir=PROJECT_ROOT)

    print(f"\nRLM answer: {result.answer}")

    # Extract hashes from answer
    import re
    found_hashes = re.findall(r'[a-f0-9]{16}', result.answer.lower())

    print(f"\n{'=' * 60}")
    print("Verification Results:")
    print("=" * 60)

    matches = 0
    for i, t in enumerate(targets):
        expected = t['hash']
        found = found_hashes[i] if i < len(found_hashes) else "NOT_FOUND"
        match = expected == found
        matches += 1 if match else 0
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"  {i+1}. Expected: {expected}")
        print(f"     Got:      {found}")
        print(f"     {status}")
        print()

    success = matches == len(targets)
    print(f"{'=' * 60}")
    print(f"Result: {matches}/{len(targets)} hashes matched")
    print(f"{'✓ VERIFIED' if success else '✗ FAILED'}: RLM {'actually read' if success else 'did NOT read'} the files")

    return success


async def test_search_and_hash():
    """
    Simpler test: Search for a unique pattern and hash the line it's on.
    """
    print("\n" + "=" * 60)
    print("Search & Hash Verification")
    print("=" * 60)

    # Find a unique line in the codebase
    unique_patterns = [
        "RLAPH Loop - Clean Recursive",
        "Session State Consolidation Plan",
        "Recursive Language Agent with Python Handler",
    ]

    for pattern in unique_patterns:
        # Find the line
        for py_file in (PROJECT_ROOT / "src").rglob("*.py"):
            with open(py_file) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if pattern in line:
                    line_content = line.strip()
                    expected_hash = hashlib.sha256(line_content.encode()).hexdigest()[:16]
                    file_path = str(py_file.relative_to(PROJECT_ROOT))
                    line_num = i + 1

                    print(f"Pattern: '{pattern[:40]}...'")
                    print(f"Location: {file_path}:{line_num}")
                    print(f"Expected hash: {expected_hash}")
                    print()

                    loop = RLAPHLoop(max_iterations=5, max_depth=1)
                    context = SessionContext()

                    query = f'''The line "{pattern}" is in {file_path} at line {line_num}.

Read that file and hash line {line_num}.

NOTE: hashlib is already available - do NOT import it.

```python
content = read_file("{file_path}", limit={line_num + 5})
lines = content.split('\\n')
line = lines[{line_num - 1}].strip()  # 0-indexed
h = hashlib.sha256(line.encode()).hexdigest()[:16]
print(f'HASH: {{h}}')
```

FINAL: <the 16-char hash>'''

                    result = await loop.run(query=query, context=context, working_dir=PROJECT_ROOT)

                    print(f"RLM answer: {result.answer}")

                    import re
                    found_hashes = re.findall(r'[a-f0-9]{16}', result.answer.lower())
                    found = found_hashes[0] if found_hashes else "NOT_FOUND"

                    match = expected_hash == found
                    print(f"\n  Expected: {expected_hash}")
                    print(f"  Got:      {found}")
                    print(f"  {'✓ PASS' if match else '✗ FAIL'}")

                    return match

    return False


async def main():
    """Run all cryptographic verification tests."""
    print("\n" + "=" * 60)
    print("RLM Cryptographic Verification")
    print("Proof of file reading via hash matching")
    print("=" * 60)
    print("\nThis test proves RLM actually reads files by requiring")
    print("it to compute SHA256 hashes of specific lines.")
    print("Hashes cannot be guessed - they MUST be computed from content.")

    results = []

    try:
        results.append(("Find & hash random lines", await test_find_and_hash_lines()))
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        results.append(("Find & hash random lines", False))

    try:
        results.append(("Search & hash pattern", await test_search_and_hash()))
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        results.append(("Search & hash pattern", False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ CRYPTOGRAPHIC PROOF: RLM actually reads and processes files")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
