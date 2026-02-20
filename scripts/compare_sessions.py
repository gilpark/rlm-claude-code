#!/usr/bin/env python3
"""Compare current session with prior session.

Hook: SessionStart
Purpose: Surface changed files, invalidated frames, resumable branches.
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from session.session_comparison import compare_sessions
from session.session_artifacts import SessionArtifacts
from frame.frame_index import FrameIndex


def find_most_recent_session(current_session_id: str) -> str | None:
    """Find the most recent session (excluding current)."""
    frames_dir = Path.home() / ".claude" / "rlm-frames"

    if not frames_dir.exists():
        return None

    sessions = []
    for session_dir in frames_dir.iterdir():
        if session_dir.is_dir() and session_dir.name != current_session_id:
            artifacts_path = session_dir / "artifacts.json"
            if artifacts_path.exists():
                sessions.append((session_dir.name, artifacts_path.stat().st_mtime))

    if not sessions:
        return None

    # Sort by modification time, most recent first
    sessions.sort(key=lambda x: x[1], reverse=True)
    return sessions[0][0]


def main():
    """Main entry point for hook."""
    # Read hook input from stdin
    hook_data = {}
    try:
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            hook_data = json.loads(stdin_data)
    except json.JSONDecodeError:
        pass

    session_id = hook_data.get("session_id", "unknown")

    # Find prior session
    prior_session_id = find_most_recent_session(session_id)

    if not prior_session_id:
        print(json.dumps({"status": "no_prior_session", "session_id": session_id}))
        sys.exit(0)

    # Load prior session artifacts
    prior = SessionArtifacts.load(prior_session_id)

    if not prior:
        print(json.dumps({"status": "no_prior_artifacts", "prior_session_id": prior_session_id}))
        sys.exit(0)

    # Load prior frame index
    prior_index = FrameIndex.load(prior_session_id)

    # Create current session (placeholder - real data comes from Claude Code)
    current = SessionArtifacts(
        session_id=session_id,
        initial_prompt=os.environ.get("CLAUDE_PROMPT", ""),
        files={},  # Would be populated from actual session
        root_frame_id="",
        conversation_log="",
    )

    # Compare sessions
    diff = compare_sessions(current, prior, index=prior_index)

    # Output results
    output = {
        "status": "compared",
        "prior_session_id": prior_session_id,
        "same_task": diff.same_task,
        "changed_files": diff.changed_files,
        "invalidated_frames": diff.invalidated_frame_ids,
        "resumable": diff.resumable_frames,
    }

    print(json.dumps(output, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
