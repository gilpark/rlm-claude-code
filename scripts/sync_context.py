#!/usr/bin/env python3
"""
Sync tool context with RLM state before tool execution.

Called by: hooks/hooks.json PreToolUse

This ensures the RLM environment has access to the latest
context before any tool (bash, edit, read) is executed.

Hooks receive JSON via stdin. This script:
1. Reads the hook input from stdin
2. Updates the state persistence layer
3. Persists context to ~/.claude/rlm-state/context.json for repl_bridge.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_state_dir() -> Path:
    """Get RLM state directory."""
    state_dir = Path.home() / ".claude" / "rlm-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def read_hook_input() -> dict[str, Any]:
    """Read hook input from stdin (JSON format per Claude Code docs)."""
    try:
        data = sys.stdin.read()
        if data:
            return json.loads(data)
    except json.JSONDecodeError:
        pass
    return {}


def load_context() -> dict[str, Any]:
    """Load existing context from state file."""
    ctx_file = get_state_dir() / "context.json"
    if ctx_file.exists():
        try:
            return json.loads(ctx_file.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "conversation": [],
        "files": {},
        "tool_outputs": [],
        "working_memory": {},
    }


def save_context(context: dict[str, Any]) -> None:
    """Save context to state file."""
    ctx_file = get_state_dir() / "context.json"
    ctx_file.write_text(json.dumps(context, indent=2, default=str))


def sync_context():
    """
    Sync context from Claude Code to RLM state.

    Reads context from stdin (JSON) and updates the persistence layer.
    """
    # Read hook input from stdin
    hook_input = read_hook_input()

    # Extract fields from hook input
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    # Also check environment variables (fallback for older hook format)
    if not tool_name:
        tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
    if not tool_input:
        tool_input_str = os.environ.get("CLAUDE_TOOL_INPUT", "")
        if tool_input_str:
            try:
                tool_input = json.loads(tool_input_str)
            except json.JSONDecodeError:
                tool_input = {"raw": tool_input_str}

    session_id = hook_input.get("session_id", os.environ.get("CLAUDE_SESSION_ID", "default"))

    try:
        from src.state_persistence import get_persistence

        persistence = get_persistence()

        # Initialize or restore session
        if persistence.current_state is None:
            persistence.init_session(session_id)

        # Log tool invocation to working memory
        if tool_name:
            persistence.update_working_memory(
                "last_tool",
                {"name": tool_name, "input_preview": str(tool_input)[:200]},
            )

        # If Read tool, prepare to cache file content
        if tool_name == "Read":
            file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
            if file_path:
                persistence.update_working_memory("pending_file_read", file_path)

        # Build context for repl_bridge.py
        context = load_context()

        # Update from persistence layer
        if persistence.current_context:
            context["conversation"] = [
                {"role": m.role.value, "content": m.content}
                for m in persistence.current_context.messages
            ]
            context["files"] = persistence.current_context.files
            context["tool_outputs"] = [
                {"tool": o.tool_name, "content": o.content[:1000]}
                for o in persistence.current_context.tool_outputs[-20:]  # Last 20
            ]
            context["working_memory"] = persistence.current_context.working_memory

        context["session_id"] = session_id
        context["rlm_active"] = persistence.current_state.rlm_active if persistence.current_state else False
        context["last_tool"] = {"name": tool_name, "input": tool_input} if tool_name else None

        # Persist state to disk
        persistence.save_state(session_id)

        save_context(context)

        # Output success
        result = {
            "status": "synced",
            "session_id": session_id,
            "tool": tool_name,
            "context_stats": {
                "messages": len(context["conversation"]),
                "files": len(context["files"]),
                "tool_outputs": len(context["tool_outputs"]),
            },
        }
        print(json.dumps(result))

    except ImportError as e:
        # Modules not available - still save basic context
        context = load_context()
        context["session_id"] = session_id
        context["last_tool"] = {"name": tool_name, "input": tool_input} if tool_name else None
        save_context(context)

        result = {"status": "synced_basic", "reason": f"import_error: {e}"}
        print(json.dumps(result))
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
        print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    sync_context()
