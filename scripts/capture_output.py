#!/usr/bin/env python3
"""
Capture tool output for RLM context after tool execution.

Called by: hooks/hooks.json PostToolUse

This captures the output of tools (bash, edit, read) and adds
them to the RLM context for subsequent processing.

Hooks receive JSON via stdin with tool_name, tool_input, and tool_response.
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
        # Debug output
        if os.environ.get("RLM_DEBUG"):
            print(f"DEBUG read_hook_input: stdin length = {len(data)}", file=sys.stderr)
        if data:
            return json.loads(data)
    except json.JSONDecodeError as e:
        if os.environ.get("RLM_DEBUG"):
            print(f"DEBUG read_hook_input: JSON error = {e}", file=sys.stderr)
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


def capture_output():
    """
    Capture tool output and add to RLM context.

    Reads output from stdin (JSON) and updates the persistence layer.
    """
    # Read hook input from stdin
    hook_input = read_hook_input()

    # Extract fields from hook input
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_response = hook_input.get("tool_response", "")

    # Also check environment variables (fallback)
    if not tool_name:
        tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
    if not tool_response:
        tool_response = os.environ.get("CLAUDE_TOOL_OUTPUT", "")

    # Normalize tool_response to string
    if isinstance(tool_response, dict):
        tool_response = json.dumps(tool_response)
    elif not isinstance(tool_response, str):
        tool_response = str(tool_response)

    session_id = hook_input.get("session_id", os.environ.get("CLAUDE_SESSION_ID", "default"))

    try:
        from src.state_persistence import get_persistence

        persistence = get_persistence()

        # Ensure session is initialized
        if persistence.current_state is None:
            persistence.init_session(session_id)

        if tool_name and tool_response:
            # Add tool output to context
            persistence.add_tool_output(
                tool_name=tool_name,
                content=tool_response[:50000],  # Limit size
                exit_code=None,
            )

            # If this was a file read, cache the content
            if tool_name == "Read":
                pending_file = persistence.current_state.working_memory.get("pending_file_read")
                if pending_file:
                    persistence.add_file_to_cache(pending_file, tool_response)
                    persistence.update_working_memory("pending_file_read", None)

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
                for o in persistence.current_context.tool_outputs[-20:]
            ]
            context["working_memory"] = persistence.current_context.working_memory

        context["session_id"] = session_id
        context["rlm_active"] = persistence.current_state.rlm_active if persistence.current_state else False

        # Add current tool output
        if tool_name and tool_response:
            context["tool_outputs"].append({
                "tool": tool_name,
                "content": tool_response[:1000],
            })

        save_context(context)

        # Output success
        result = {
            "status": "captured",
            "tool": tool_name,
            "output_size": len(tool_response) if tool_response else 0,
            "context_stats": {
                "files": len(context["files"]),
                "tool_outputs": len(context["tool_outputs"]),
            },
        }
        print(json.dumps(result))

    except ImportError as e:
        # Modules not available - still update context
        context = load_context()
        if tool_name and tool_response:
            context["tool_outputs"].append({
                "tool": tool_name,
                "content": tool_response[:1000],
            })
            # If Read tool, cache file
            if tool_name == "Read" and isinstance(tool_input, dict):
                file_path = tool_input.get("file_path", "")
                if file_path:
                    context["files"][file_path] = tool_response
        save_context(context)

        result = {"status": "captured_basic", "reason": f"import_error: {e}"}
        print(json.dumps(result))
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
        print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    capture_output()
