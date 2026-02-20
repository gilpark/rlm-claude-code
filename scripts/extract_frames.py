#!/usr/bin/env python3
"""Extract frames from session and save to FrameStore.

Hook: Stop
Purpose: Persist CausalFrame tree when session ends.
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.frame.frame_store import FrameStore
from src.frame.frame_index import FrameIndex


def extract_frames(session_id: str, index: FrameIndex) -> None:
    """Extract all frames from index and save to FrameStore."""
    session_dir = Path.home() / ".claude" / "rlm-frames" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    store_path = session_dir / "frames.jsonl"
    store = FrameStore(path=store_path)

    for frame in index.values():
        store.save(frame)

    print(f"Saved {len(index)} frames to {store_path}")


def main():
    """Main entry point for hook."""
    # Read hook input from stdin to get session_id
    hook_data = {}
    try:
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            hook_data = json.loads(stdin_data)
    except json.JSONDecodeError:
        pass

    session_id = hook_data.get("session_id", os.environ.get("CLAUDE_SESSION_ID", "unknown"))

    # For now, this is a placeholder - the actual frame extraction
    # would come from the running session's FrameIndex
    # In production, this would receive the frame data via stdin

    print(f"Frame extraction hook called for session: {session_id}")

    # Read frame data from stdin if provided
    try:
        input_data = sys.stdin.read()
        if input_data:
            data = json.loads(input_data)
            print(f"Received {len(data.get('frames', []))} frames")
    except json.JSONDecodeError:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
