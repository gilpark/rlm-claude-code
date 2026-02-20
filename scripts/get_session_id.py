#!/usr/bin/env python3
"""
Get or create session ID for RLM operations.

This script provides a coordination point for session_id between
hooks (which receive it from Claude Code) and the orchestrator
(which needs it for frame persistence).

Usage:
    # Get current session ID (from hook input)
    echo '{"session_id": "abc123"}' | python scripts/get_session_id.py

    # Get or create session ID (for orchestrator)
    python scripts/get_session_id.py --ensure

The session ID is stored in ~/.claude/rlm-frames/.current_session
"""

import json
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime


def get_session_file() -> Path:
    """Get path to current session file."""
    return Path.home() / ".claude" / "rlm-frames" / ".current_session"


def get_session_id() -> str | None:
    """Get current session ID from coordination file."""
    session_file = get_session_file()
    if session_file.exists():
        data = json.loads(session_file.read_text())
        return data.get("session_id")
    return None


def set_session_id(session_id: str) -> None:
    """Set current session ID in coordination file."""
    session_file = get_session_file()
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(json.dumps({
        "session_id": session_id,
        "pid": os.getpid(),
        "updated_at": datetime.now().isoformat(),
    }))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Get or create session ID")
    parser.add_argument("--ensure", action="store_true",
                        help="Create session ID if not exists")
    parser.add_argument("--set", dest="set_id", help="Set session ID")
    args = parser.parse_args()

    # Set session ID from argument
    if args.set_id:
        set_session_id(args.set_id)
        print(json.dumps({"session_id": args.set_id}))
        return

    # Try to read from stdin (hook input)
    stdin_data = ""
    if not sys.stdin.isatty():
        stdin_data = sys.stdin.read().strip()

    if stdin_data:
        try:
            hook_data = json.loads(stdin_data)
            session_id = hook_data.get("session_id")
            if session_id:
                set_session_id(session_id)
                print(json.dumps({"session_id": session_id}))
                return
        except json.JSONDecodeError:
            pass

    # Get existing session ID
    session_id = get_session_id()

    if session_id:
        print(json.dumps({"session_id": session_id}))
        return

    if args.ensure:
        # Generate new session ID
        session_id = str(uuid.uuid4())[:8]
        set_session_id(session_id)
        print(json.dumps({"session_id": session_id}))
        return

    # No session ID available
    print(json.dumps({"session_id": None, "error": "No session ID found"}))


if __name__ == "__main__":
    main()
