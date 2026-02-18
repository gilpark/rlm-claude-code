#!/usr/bin/env python3
"""
Capture tool output for RLM context after tool execution.

Called by: hooks/hooks.json PostToolUse

This captures the output of tools (bash, edit, read) and adds
them to the RLM context for subsequent processing.

Uses two sources of data:
1. Hook input via stdin JSON (immediate tool info)
2. Full transcript parsing (complete context)

Hook input (via stdin JSON):
{
    "session_id": "...",
    "transcript_path": "...",
    "tool_name": "Bash|Read|Edit|Write|...",
    "tool_input": {...},
    "tool_response": "...",
    "tool_use_id": "..."
}
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def capture_output():
    """
    Capture tool output and add to RLM context.

    Reads hook data from stdin JSON and augments with full
    transcript data for comprehensive context capture.
    """
    try:
        from src.state_persistence import get_persistence
        from src.types import MessageRole

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

        # Ensure session is initialized
        if persistence.current_state is None:
            persistence.init_session(session_id)

        # Extract tool info from hook data (stdin JSON)
        tool_name = hook_data.get("tool_name", "")
        tool_input = hook_data.get("tool_input", {})
        tool_response = hook_data.get("tool_response", "")

        # Convert tool_response to string if it's a dict
        if isinstance(tool_response, dict):
            tool_response = json.dumps(tool_response, indent=2)

        # Also try to load full context from transcript
        transcript_context = {}
        try:
            from src.transcript_parser import load_session_context
            transcript_context = load_session_context()

            # Update files cache from transcript (more complete)
            for path, content in transcript_context.get("files", {}).items():
                persistence.add_file_to_cache(path, content)

            # Update messages from transcript
            for msg in transcript_context.get("messages", [])[-20:]:  # Last 20 messages
                role_str = msg.get("role", "user")
                try:
                    role = MessageRole.USER if role_str == "user" else MessageRole.ASSISTANT
                    # Don't duplicate - check if already in messages
                    # For now, just add recent messages
                except:
                    pass

        except ImportError:
            pass  # Transcript parser not available
        except Exception as e:
            pass  # Transcript parsing failed, continue with hook data

        if tool_name and tool_response:
            # Add tool output to context
            persistence.add_tool_output(
                tool_name=tool_name,
                content=tool_response[:50000],  # Limit size
                exit_code=None,
            )

            # If this was a file read, cache the content
            if tool_name == "Read":
                file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
                if file_path:
                    persistence.add_file_to_cache(file_path, tool_response)

            # If this was Bash with errors, track for complexity detection
            if tool_name == "Bash":
                if isinstance(tool_response, str) and ("error" in tool_response.lower() or "failed" in tool_response.lower()):
                    error_count = persistence.current_state.working_memory.get(
                        "recent_errors", 0
                    )
                    persistence.update_working_memory("recent_errors", error_count + 1)

            # Always save after capturing tool output
            persistence.save_state()

        # Output success
        result = {
            "status": "captured",
            "tool": tool_name,
            "output_size": len(tool_response) if tool_response else 0,
            "session_id": session_id,
            "transcript_files": len(transcript_context.get("files", {})) if transcript_context else 0,
        }
        print(json.dumps(result))

    except ImportError as e:
        result = {"status": "skipped", "reason": f"import_error: {e}"}
        print(json.dumps(result))
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
        print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    capture_output()
