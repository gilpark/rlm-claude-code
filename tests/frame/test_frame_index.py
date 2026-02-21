"""Tests for FrameIndex commit_hash support for git-aware invalidation."""

import pytest
from datetime import datetime
from pathlib import Path

from src.frame.frame_index import FrameIndex
from src.frame.context_slice import ContextSlice
from src.frame.causal_frame import CausalFrame, FrameStatus


def test_frame_index_commit_hash_defaults_none():
    """FrameIndex commit_hash should default to None."""
    index = FrameIndex()
    assert index.commit_hash is None


def test_frame_index_save_load_with_commit_hash(tmp_path):
    """FrameIndex should persist and load commit_hash."""
    index = FrameIndex()
    index.commit_hash = "abc12345"

    frame = CausalFrame(
        frame_id="test123",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(
            files={}, memory_refs=[], tool_outputs={}, token_budget=8000
        ),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.8,
        invalidation_condition={},
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(frame)

    save_path = index.save("test-session", tmp_path)
    loaded = FrameIndex.load("test-session", tmp_path)

    assert loaded.commit_hash == "abc12345"


def test_frame_index_load_without_commit_hash(tmp_path):
    """FrameIndex should handle loading old files without commit_hash."""
    # Create index with a frame
    index = FrameIndex()
    frame = CausalFrame(
        frame_id="test456",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(
            files={}, memory_refs=[], tool_outputs={}, token_budget=8000
        ),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.8,
        invalidation_condition={},
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(frame)
    index.save("test-session-old", tmp_path)

    # Manually remove commit_hash from the saved file to simulate old format
    import json

    old_file = tmp_path / "test-session-old" / "index.json"
    with open(old_file) as f:
        data = json.load(f)

    # Remove commit_hash if present (simulating old file format)
    data.pop("commit_hash", None)

    with open(old_file, "w") as f:
        json.dump(data, f, indent=2)

    # Load should still work and default to None
    loaded = FrameIndex.load("test-session-old", tmp_path)
    assert loaded.commit_hash is None
    assert len(loaded) == 1  # Frame should still be loaded
