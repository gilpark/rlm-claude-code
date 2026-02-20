#!/usr/bin/env python3
"""Extract frames from session and save to FrameStore.

Hook: Stop
Purpose: Persist CausalFrame tree when session ends.

This hook loads frames saved by RLAPHLoop and ensures they are
persisted to the FrameStore for long-term storage.
"""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frame.frame_store import FrameStore
from frame.frame_index import FrameIndex


def get_session_id_for_frames(hook_session_id: str) -> str:
    """
    Get the session ID that frames were saved with.

    Priority:
    1. Coordination file (set by orchestrator or capture_output)
    2. Hook input session_id
    3. "default"
    """
    # Check coordination file first
    session_file = Path.home() / ".claude" / "rlm-frames" / ".current_session"
    if session_file.exists():
        try:
            data = json.loads(session_file.read_text())
            if data.get("session_id"):
                return data["session_id"]
        except (json.JSONDecodeError, KeyError):
            pass

    return hook_session_id or "default"


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

    hook_session_id = hook_data.get("session_id", "default")

    # Get the session ID that frames were saved with
    session_id = get_session_id_for_frames(hook_session_id)

    # Extract and persist frames
    result = extract_frames(session_id)

    # Output result
    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
