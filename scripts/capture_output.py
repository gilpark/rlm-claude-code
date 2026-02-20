#!/usr/bin/env python3
"""
Capture tool output into active CausalFrame.

Hook: PostToolUse
Purpose: Record tool outputs in the current frame's context_slice.

Input (stdin JSON from Claude Code):
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
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def capture_output():
    """
    Capture tool output and record in frame context.

    For v2, we track tool outputs in a simple JSONL log that can be
    picked up by the frame extraction process.
    """
    # Read hook input from stdin FIRST
    hook_data = {}
    try:
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            hook_data = json.loads(stdin_data)
    except json.JSONDecodeError:
        pass

    # Get session_id from hook input (not env var)
    session_id = hook_data.get("session_id", "default")
    session_dir = Path.home() / ".claude" / "rlm-frames" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Also save to coordination file for orchestrator
    session_file = Path.home() / ".claude" / "rlm-frames" / ".current_session"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(json.dumps({
        "session_id": session_id,
        "pid": os.getpid(),
        "updated_at": datetime.now().isoformat(),
    }))

    # Extract tool info
    tool_name = hook_data.get("tool_name", "")
    tool_input = hook_data.get("tool_input", {})
    tool_response = hook_data.get("tool_response", "")
    tool_use_id = hook_data.get("tool_use_id", "")

    # Convert response to string if needed
    if isinstance(tool_response, dict):
        tool_response = json.dumps(tool_response, indent=2)

    if not tool_name:
        result = {"status": "skipped", "reason": "no tool_name"}
        print(json.dumps(result))
        return

    # Record tool output for frame capture
    tool_record = {
        "type": "tool_output",
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_response[:10000] if tool_response else "",  # Limit size
        "tool_use_id": tool_use_id,
    }

    # Append to session tool log
    tool_log_path = session_dir / "tools.jsonl"
    with open(tool_log_path, "a") as f:
        f.write(json.dumps(tool_record) + "\n")

    # Track file reads for context_slice
    if tool_name == "Read":
        file_path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
        if file_path and tool_response:
            # Record file read for frame context
            read_record = {
                "type": "file_read",
                "session_id": session_id,
                "file_path": file_path,
                "content_hash": hash(tool_response),  # Just track hash, not content
            }
            with open(tool_log_path, "a") as f:
                f.write(json.dumps(read_record) + "\n")

    # Output success
    result = {
        "status": "captured",
        "tool": tool_name,
        "output_size": len(tool_response) if tool_response else 0,
        "session_id": session_id,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    capture_output()
