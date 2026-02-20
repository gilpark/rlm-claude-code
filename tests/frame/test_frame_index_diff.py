"""Tests for cross-session diff-based frame loading."""

import hashlib
from datetime import datetime
from pathlib import Path

import pytest

from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.frame.frame_index import FrameIndex


def test_load_frames_with_git_diff_invalidates_changed(tmp_path, monkeypatch):
    """Loading frames should invalidate those with changed files."""
    test_file = tmp_path / "test.py"
    test_file.write_text("original content")

    index = FrameIndex()
    index.commit_hash = "oldcommit"

    original_hash = hashlib.blake2b(test_file.read_bytes()).hexdigest()[:16]

    frame = CausalFrame(
        frame_id="test123",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(
            files={str(test_file): original_hash},
            memory_refs=[],
            tool_outputs={},
            token_budget=8000,
        ),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(frame)
    index.save("test-session", tmp_path)

    test_file.write_text("modified content")

    # Mock git diff to say file changed
    def mock_detect_changed(*args, **kwargs):
        return {test_file.resolve()}

    import src.frame.context_map as cm

    monkeypatch.setattr(cm, "detect_changed_files", mock_detect_changed)

    loaded = FrameIndex.load_with_validation("test-session", tmp_path, tmp_path)

    assert loaded.get("test123").status == FrameStatus.INVALIDATED


def test_load_frames_without_changes_keeps_valid(tmp_path, monkeypatch):
    """Loading frames with no changes should keep frames valid."""
    test_file = tmp_path / "test.py"
    test_file.write_text("unchanged content")

    index = FrameIndex()
    index.commit_hash = "oldcommit"

    content_hash = hashlib.blake2b(test_file.read_bytes()).hexdigest()[:16]

    frame = CausalFrame(
        frame_id="test456",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(
            files={str(test_file): content_hash},
            memory_refs=[],
            tool_outputs={},
            token_budget=8000,
        ),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(frame)
    index.save("test-session", tmp_path)

    def mock_detect_changed(*args, **kwargs):
        return set()

    import src.frame.context_map as cm

    monkeypatch.setattr(cm, "detect_changed_files", mock_detect_changed)

    loaded = FrameIndex.load_with_validation("test-session", tmp_path, tmp_path)

    assert loaded.get("test456").status == FrameStatus.COMPLETED


def test_load_with_validation_no_commit_hash(tmp_path, monkeypatch):
    """Frames without commit hash should load without validation."""
    test_file = tmp_path / "test.py"
    test_file.write_text("content")

    index = FrameIndex()
    index.commit_hash = None  # No commit hash

    content_hash = hashlib.blake2b(test_file.read_bytes()).hexdigest()[:16]

    frame = CausalFrame(
        frame_id="test789",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(
            files={str(test_file): content_hash},
            memory_refs=[],
            tool_outputs={},
            token_budget=8000,
        ),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(frame)
    index.save("no-commit-session", tmp_path)

    # This would invalidate if called, but shouldn't be called
    def mock_detect_changed(*args, **kwargs):
        raise AssertionError("detect_changed_files should not be called")

    import src.frame.context_map as cm

    monkeypatch.setattr(cm, "detect_changed_files", mock_detect_changed)

    loaded = FrameIndex.load_with_validation(
        "no-commit-session", tmp_path, tmp_path
    )

    # Frame should remain COMPLETED since no commit hash was stored
    assert loaded.get("test789").status == FrameStatus.COMPLETED


def test_load_with_validation_no_current_root(tmp_path, monkeypatch):
    """Frames loaded without current_root should skip validation."""
    test_file = tmp_path / "test.py"
    test_file.write_text("content")

    index = FrameIndex()
    index.commit_hash = "somecommit"

    content_hash = hashlib.blake2b(test_file.read_bytes()).hexdigest()[:16]

    frame = CausalFrame(
        frame_id="test000",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(
            files={str(test_file): content_hash},
            memory_refs=[],
            tool_outputs={},
            token_budget=8000,
        ),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(frame)
    index.save("no-root-session", tmp_path)

    # This would invalidate if called, but shouldn't be called
    def mock_detect_changed(*args, **kwargs):
        raise AssertionError("detect_changed_files should not be called")

    import src.frame.context_map as cm

    monkeypatch.setattr(cm, "detect_changed_files", mock_detect_changed)

    loaded = FrameIndex.load_with_validation("no-root-session", tmp_path, None)

    # Frame should remain COMPLETED since no current_root was provided
    assert loaded.get("test000").status == FrameStatus.COMPLETED


def test_load_with_validation_skips_non_completed(tmp_path, monkeypatch):
    """Only COMPLETED frames should be checked for invalidation."""
    test_file = tmp_path / "test.py"
    test_file.write_text("content")

    index = FrameIndex()
    index.commit_hash = "oldcommit"

    content_hash = hashlib.blake2b(test_file.read_bytes()).hexdigest()[:16]

    # Create a RUNNING frame (should be skipped)
    running_frame = CausalFrame(
        frame_id="running1",
        depth=0,
        parent_id=None,
        children=[],
        query="running query",
        context_slice=ContextSlice(
            files={str(test_file): content_hash},
            memory_refs=[],
            tool_outputs={},
            token_budget=8000,
        ),
        evidence=[],
        conclusion="",
        confidence=0.0,
        invalidation_condition="",
        status=FrameStatus.RUNNING,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=None,
    )
    index.add(running_frame)
    index.save("running-session", tmp_path)

    def mock_detect_changed(*args, **kwargs):
        return {test_file.resolve()}

    import src.frame.context_map as cm

    monkeypatch.setattr(cm, "detect_changed_files", mock_detect_changed)

    loaded = FrameIndex.load_with_validation("running-session", tmp_path, tmp_path)

    # RUNNING frame should remain RUNNING (not invalidated)
    assert loaded.get("running1").status == FrameStatus.RUNNING
