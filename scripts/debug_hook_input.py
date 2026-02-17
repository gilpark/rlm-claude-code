#!/usr/bin/env python3
"""
Debug script to see what hooks actually receive.

Run with: echo '{"test": "input"}' | python scripts/debug_hook_input.py
"""

import json
import os
import sys
from pathlib import Path

def debug_hook_input():
    """Print all input sources for debugging."""

    print("=== ENVIRONMENT VARIABLES ===", file=sys.stderr)
    for key, value in os.environ.items():
        if key.startswith("CLAUDE_"):
            print(f"  {key}={value[:200] if len(value) > 200 else value}", file=sys.stderr)

    print("\n=== STDIN ===", file=sys.stderr)
    stdin_data = ""
    try:
        stdin_data = sys.stdin.read()
        if stdin_data:
            print(f"  Raw: {stdin_data[:500]}", file=sys.stderr)
            try:
                parsed = json.loads(stdin_data)
                print(f"  Parsed JSON:", file=sys.stderr)
                print(f"    Keys: {list(parsed.keys())}", file=sys.stderr)
                for key, value in parsed.items():
                    val_str = str(value)[:100]
                    print(f"    {key}: {val_str}", file=sys.stderr)
            except json.JSONDecodeError:
                print("  (Not valid JSON)", file=sys.stderr)
        else:
            print("  (empty)", file=sys.stderr)
    except Exception as e:
        print(f"  Error reading stdin: {e}", file=sys.stderr)

    print("\n=== ARGV ===", file=sys.stderr)
    print(f"  {sys.argv}", file=sys.stderr)

    # Output result
    result = {
        "status": "debug",
        "env_vars": {k: v[:100] for k, v in os.environ.items() if k.startswith("CLAUDE_")},
        "stdin_length": len(stdin_data),
        "argv": sys.argv,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    debug_hook_input()
