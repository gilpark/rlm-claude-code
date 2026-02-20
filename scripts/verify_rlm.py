#!/usr/bin/env python3
"""
Quick verification that RLM and RLM Environment are working.

First-time setup:
    # Create virtual environment with Python 3.12
    uv venv --python 3.12

    # Install dependencies
    uv sync --all-extras

    # Run verification
    uv run python scripts/verify_rlm.py

If you already have the venv set up:
    uv run python scripts/verify_rlm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_repl_basics():
    """Test basic REPL execution."""
    from src.repl.repl_environment import RLMEnvironment
    from src.types import SessionContext

    context = SessionContext()
    env = RLMEnvironment(context)

    # Test simple math
    result = env.execute("x = 2 + 2\nprint(f'Result: {x}')")
    assert result.success, f"Basic math failed: {result.error}"
    assert "Result: 4" in result.output, f"Wrong output: {result.output}"
    print("  ✓ Basic Python execution works")
    return True


def test_repl_hashlib():
    """Test hashlib is available."""
    from src.repl.repl_environment import RLMEnvironment
    from src.types import SessionContext
    import hashlib

    context = SessionContext()
    env = RLMEnvironment(context)
    env.globals["hashlib"] = hashlib

    result = env.execute('h = hashlib.sha256(b"test").hexdigest()[:8]\nprint(f"Hash: {h}")')
    assert result.success, f"Hashlib failed: {result.error}"
    assert "Hash:" in result.output, f"No hash output: {result.output}"
    print("  ✓ hashlib available in REPL")
    return True


def test_file_access():
    """Test file access functions."""
    from src.repl.repl_environment import RLMEnvironment
    from src.types import SessionContext

    context = SessionContext()
    env = RLMEnvironment(context)
    env.enable_file_access(working_dir=Path(__file__).parent.parent)

    # Test glob
    result = env.execute('files = glob_files("src/*.py")\nprint(f"Found {len(files)} files")')
    assert result.success, f"glob_files failed: {result.error}"
    assert "Found" in result.output, f"No glob output: {result.output}"
    print("  ✓ glob_files() works")

    # Test read_file
    result = env.execute('content = read_file("src/types.py", limit=10)\nprint(f"Read {len(content)} chars")')
    assert result.success, f"read_file failed: {result.error}"
    assert "Read" in result.output, f"No read output: {result.output}"
    print("  ✓ read_file() works")

    return True


async def test_rlm_loop():
    """Test RLM loop execution."""
    import asyncio
    from src.repl.rlaph_loop import RLAPHLoop
    from src.types import SessionContext

    context = SessionContext()
    loop = RLAPHLoop(max_iterations=3, max_depth=0)

    result = await loop.run(
        query="What is 2 + 2? Just answer the number.",
        context=context,
        working_dir=Path(__file__).parent.parent,
    )

    assert result.answer, "No answer from RLM"
    assert "4" in result.answer, f"Wrong answer: {result.answer}"
    print("  ✓ RLM loop executes and returns answers")
    return True


def main():
    print("=" * 50)
    print("RLM Verification")
    print("=" * 50)

    all_passed = True

    # Test 1: REPL basics
    print("\n1. REPL Environment Basics")
    try:
        test_repl_basics()
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        all_passed = False

    # Test 2: hashlib
    print("\n2. Hashlib in REPL")
    try:
        test_repl_hashlib()
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        all_passed = False

    # Test 3: File access
    print("\n3. File Access Functions")
    try:
        test_file_access()
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        all_passed = False

    # Test 4: RLM loop
    print("\n4. RLM Loop Execution")
    try:
        import asyncio

        asyncio.run(test_rlm_loop())
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ ALL TESTS PASSED - RLM is working!")
        print("=" * 50)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
