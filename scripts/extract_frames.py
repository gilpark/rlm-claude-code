#!/usr/bin/env python3
"""Extract frames from session and save to FrameStore.

Hook: Stop
Purpose: Persist CausalFrame tree when session ends.

This hook loads frames saved by RLAPHLoop and ensures they are
persisted to the FrameStore for long-term storage.
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frame.frame_store import FrameStore
from frame.frame_index import FrameIndex


def extract_frames(session_id: str) -> dict:
    """
    Extract frames from saved index and persist to FrameStore.

    Args:
        session_id: The session identifier

    Returns:
        Dict with status and frame count
    """
    # Load frames from index (saved by RLAPHLoop)
    index = FrameIndex.load(session_id)

    if len(index) == 0:
        return {"status": "no_frames", "count": 0, "session_id": session_id}

    # Create session directory
    session_dir = Path.home() / ".claude" / "rlm-frames" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save to FrameStore (JSONL format)
    store_path = session_dir / "frames.jsonl"
    store = FrameStore(path=store_path)

    for frame in index.values():
        store.save(frame)

    return {
        "status": "success",
        "count": len(index),
        "session_id": session_id,
        "store_path": str(store_path),
    }


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

    session_id = hook_data.get("session_id", os.environ.get("CLAUDE_SESSION_ID", "default"))

    # Extract and persist frames
    result = extract_frames(session_id)

    # Output result for debugging
    print(json.dumps(result))

    sys.exit(0)


if __name__ == "__main__":
    main()
