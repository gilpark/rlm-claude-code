#!/usr/bin/env python3
"""
Rigorous verification: Proves RLM actually reads file content.

Uses controlled test files with KNOWN content to prove file access.
The RLM must return exact values that can ONLY be obtained by reading the files.

Usage:
    uv run python tests/integration/test_rlm_verification.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext


def create_test_files() -> tuple[Path, dict]:
    """
    Create controlled test files with known content.

    Returns:
        Tuple of (test_dir, ground_truth)
    """
    test_dir = Path(tempfile.mkdtemp(prefix="rlm_verify_"))

    # File 1: Functions file with known number of functions
    file1 = test_dir / "functions.py"
    func_count = 50
    file1_content = "\n".join([f"def func_{i}(x): return x * {i}" for i in range(func_count)])
    file1.write_text(file1_content)

    # File 2: Classes file with known number of classes
    file2 = test_dir / "classes.py"
    class_count = 25
    file2_content = "\n".join([f"class Class{i}: pass" for i in range(class_count)])
    file2.write_text(file2_content)

    # File 3: Data file with known pattern
    file3 = test_dir / "data.txt"
    pattern_count = 100
    secret_code = "VERIFY-ME-12345"
    file3_lines = [f"Line {i}: data" for i in range(1000)]
    # Insert secret at specific locations
    for i in range(pattern_count):
        file3_lines[i * 10] = f"Line {i * 10}: {secret_code}"
    file3.write_text("\n".join(file3_lines))

    # File 4: Large file to test chunked reading
    file4 = test_dir / "large.py"
    large_lines = 5000
    file4_lines = [f"# Line {i}: {'x' * 50}" for i in range(large_lines)]
    # Hide a secret
    file4_lines[2500] = f"# SECRET_LINE: FOUND-AT-2500"
    file4.write_text("\n".join(file4_lines))

    ground_truth = {
        "func_count": func_count,
        "class_count": class_count,
        "pattern_count": pattern_count,
        "secret_code": secret_code,
        "large_lines": large_lines,
        "hidden_secret": "FOUND-AT-2500",
        "hidden_line": 2500,
        "test_dir": test_dir,
    }

    return test_dir, ground_truth


async def test_verify_function_count():
    """Verify RLM correctly counts functions in controlled file."""
    print("\n" + "=" * 60)
    print("Test 1: Function Count Verification")
    print("=" * 60)

    test_dir, ground_truth = create_test_files()

    try:
        loop = RLAPHLoop(max_iterations=5, max_depth=1)
        context = SessionContext()

        query = f'''Read the file at {test_dir}/functions.py and count how many "def " appear.

```python
content = read_file("{test_dir}/functions.py", limit=100)
count = content.count("def ")
print(f"Function count: {{count}}")
```

FINAL: <the exact count>'''

        result = await loop.run(query=query, context=context, working_dir=test_dir)

        print(f"Ground truth: {ground_truth['func_count']} functions")
        print(f"RLM answer: {result.answer}")

        import re
        numbers = re.findall(r'\d+', result.answer)
        if numbers:
            rlm_count = int(numbers[0])
            match = rlm_count == ground_truth['func_count']
            print(f"{'✓ PASS' if match else '✗ FAIL'}: Count is {'correct' if match else 'incorrect'}")
            return match
        else:
            print("✗ FAIL: Could not extract count")
            return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_verify_secret_location():
    """Verify RLM can find a secret at a specific line in a large file."""
    print("\n" + "=" * 60)
    print("Test 2: Secret Location in Large File")
    print("=" * 60)

    test_dir, ground_truth = create_test_files()

    try:
        loop = RLAPHLoop(max_iterations=5, max_depth=1)
        context = SessionContext()

        query = f'''Search for "SECRET_LINE" in {test_dir}/large.py using grep_files:

```python
result = grep_files("SECRET_LINE", "{test_dir}/large.py")
print(result)
```

Tell me what the secret says and which line it's on.

FINAL: <the secret text and line number>'''

        result = await loop.run(query=query, context=context, working_dir=test_dir)

        print(f"Ground truth: '{ground_truth['hidden_secret']}' at line {ground_truth['hidden_line']}")
        print(f"RLM answer: {result.answer}")

        # Check if both the secret and line number are in the answer
        has_secret = ground_truth['hidden_secret'] in result.answer
        has_line = str(ground_truth['hidden_line']) in result.answer

        match = has_secret and has_line
        print(f"  Secret found: {'✓' if has_secret else '✗'}")
        print(f"  Line number correct: {'✓' if has_line else '✗'}")
        print(f"{'✓ PASS' if match else '✗ FAIL'}")

        return match
    finally:
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_verify_pattern_count():
    """Verify RLM correctly counts pattern occurrences."""
    print("\n" + "=" * 60)
    print("Test 3: Pattern Count Verification")
    print("=" * 60)

    test_dir, ground_truth = create_test_files()

    try:
        loop = RLAPHLoop(max_iterations=5, max_depth=1)
        context = SessionContext()

        query = f'''Count how many times "{ground_truth['secret_code']}" appears in {test_dir}/data.txt.

Use grep_files to search:
```python
result = grep_files("{ground_truth['secret_code']}", "{test_dir}/data.txt")
count = result.count("{ground_truth['secret_code']}")
print(f"Count: {{count}}")
```

FINAL: <the exact count>'''

        result = await loop.run(query=query, context=context, working_dir=test_dir)

        print(f"Ground truth: {ground_truth['pattern_count']} occurrences")
        print(f"RLM answer: {result.answer}")

        import re
        numbers = re.findall(r'\d+', result.answer)
        if numbers:
            rlm_count = int(numbers[0])
            match = rlm_count == ground_truth['pattern_count']
            print(f"{'✓ PASS' if match else '✗ FAIL'}: Count is {'correct' if match else 'incorrect'}")
            return match
        else:
            print("✗ FAIL: Could not extract count")
            return False
    finally:
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)


async def test_verify_file_list():
    """Verify RLM correctly lists files."""
    print("\n" + "=" * 60)
    print("Test 4: File Listing Verification")
    print("=" * 60)

    test_dir, ground_truth = create_test_files()

    try:
        loop = RLAPHLoop(max_iterations=5, max_depth=1)
        context = SessionContext()

        query = f'''List all files in {test_dir} using glob_files:

```python
files = glob_files("*.py", "{test_dir}")
print(f"Files: {{files}}")
```

FINAL: <list the Python file names>'''

        result = await loop.run(query=query, context=context, working_dir=test_dir)

        expected_files = {"functions.py", "classes.py", "large.py"}

        print(f"Expected files: {expected_files}")
        print(f"RLM answer: {result.answer}")

        found = sum(1 for f in expected_files if f in result.answer)
        match = found == len(expected_files)

        print(f"  Files found: {found}/{len(expected_files)}")
        print(f"{'✓ PASS' if match else '✗ FAIL'}")

        return match
    finally:
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)


async def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("RLM File Access Verification")
    print("Rigorous tests using controlled file content")
    print("=" * 60)

    results = []

    tests = [
        ("Function count", test_verify_function_count),
        ("Secret location", test_verify_secret_location),
        ("Pattern count", test_verify_pattern_count),
        ("File listing", test_verify_file_list),
    ]

    for name, test_fn in tests:
        try:
            passed = await test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ VERIFIED: RLM actually reads file content")
    else:
        print("\n✗ VERIFICATION FAILED: RLM may not be reading files correctly")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
