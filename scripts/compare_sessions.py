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
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.session.session_comparison import compare_sessions
from src.session.session_artifacts import SessionArtifacts, FileRecord
from src.frame.frame_store import FrameStore


def load_prior_session(session_id: str) -> SessionArtifacts | None:
    """Load prior session artifacts if they exist."""
    artifacts_path = Path.home() / ".claude" / "rlm-frames" / f"{session_id}_artifacts.json"

    if not artifacts_path.exists():
        return None

    with open(artifacts_path) as f:
        data = json.load(f)

    # Reconstruct FileRecord dict
    files = {
        path: FileRecord(path=path, hash=rec["hash"], role=rec["role"])
        for path, rec in data.get("files", {}).items()
    }

    return SessionArtifacts(
        session_id=data["session_id"],
        initial_prompt=data.get("initial_prompt", ""),
        files=files,
        root_frame_id=data.get("root_frame_id", ""),
        conversation_log=data.get("conversation_log", ""),
    )


def main():
    """Main entry point for hook."""
    session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")

    # Get prior session ID if provided
    prior_session_id = os.environ.get("CLAUDE_PRIOR_SESSION_ID")

    if not prior_session_id:
        print("No prior session to compare")
        sys.exit(0)

    # Load prior session
    prior = load_prior_session(prior_session_id)

    if not prior:
        print(f"Could not load prior session: {prior_session_id}")
        sys.exit(0)

    # Create current session from environment/args
    # This is a placeholder - real implementation would get actual file info
    current = SessionArtifacts(
        session_id=session_id,
        initial_prompt=os.environ.get("CLAUDE_PROMPT", ""),
        files={},  # Would be populated from actual session
        root_frame_id="",
        conversation_log="",
    )

    # Compare
    diff = compare_sessions(current, prior)

    # Output results for Claude to see
    output = {
        "same_task": diff.same_task,
        "changed_files": diff.changed_files,
        "invalidated_frames": diff.invalidated_frame_ids,
        "resumable": diff.resumable_frames,
    }

    print(json.dumps(output, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
