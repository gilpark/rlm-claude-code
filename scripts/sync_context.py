#!/usr/bin/env python3
"""
Sync tool context with RLM state before tool execution.

Called by: hooks/hooks.json PreToolUse

This ensures the RLM environment has access to the latest
context before any tool (bash, edit, read) is executed.

Hook input (via stdin JSON):
{
    "session_id": "...",
    "transcript_path": "...",
    "tool_name": "Bash|Read|Edit|Write|...",
    "tool_input": {...},
    "tool_use_id": "..."
}
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def sync_context():
    """
    Sync context from Claude Code to RLM state.

    Reads hook data from stdin JSON (Claude Code hooks protocol)
    and updates the RLM state persistence layer.
    """
    try:
        from src.state_persistence import get_persistence

        persistence = get_persistence()

        # Read hook input from stdin (Claude Code hooks pass JSON via stdin)
        hook_data = {}
        try:
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                hook_data = json.loads(stdin_data)
        except json.JSONDecodeError:
            pass  # Empty or invalid stdin

        # Get session ID from hook data or environment
        session_id = hook_data.get("session_id") or os.environ.get("CLAUDE_SESSION_ID", "default")

        # Initialize or restore session
        if persistence.current_state is None:
            persistence.init_session(session_id)

        # Extract tool info from hook data (stdin JSON)
        tool_name = hook_data.get("tool_name", "")
        tool_input = hook_data.get("tool_input", {})

        if tool_name:
            # Log tool invocation to working memory
            input_preview = json.dumps(tool_input)[:200] if isinstance(tool_input, dict) else str(tool_input)[:200]
            persistence.update_working_memory(
                "last_tool",
                {"name": tool_name, "input_preview": input_preview},
            )

        # If read tool, prepare to cache file content
        if tool_name == "Read" and isinstance(tool_input, dict):
            file_path = tool_input.get("file_path", "")
            if file_path:
                persistence.update_working_memory("pending_file_read", file_path)

        # Output success
        result = {
            "status": "synced",
            "session_id": session_id,
            "rlm_active": persistence.current_state.rlm_active if persistence.current_state else False,
        }
        print(json.dumps(result))

    except ImportError as e:
        # Modules not available - output minimal result
        result = {"status": "skipped", "reason": f"import_error: {e}"}
        print(json.dumps(result))
    except Exception as e:
        # Log error but don't block tool execution
        result = {"status": "error", "reason": str(e)}
        print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    sync_context()
