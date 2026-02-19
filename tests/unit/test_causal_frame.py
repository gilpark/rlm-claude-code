"""Tests for CausalFrame and related types."""

from src.causal_frame import FrameStatus


def test_frame_status_has_all_values():
    """FrameStatus enum should have all required status values."""
    assert FrameStatus.CREATED.value == "created"
    assert FrameStatus.RUNNING.value == "running"
    assert FrameStatus.COMPLETED.value == "completed"
    assert FrameStatus.VERIFIED.value == "verified"
    assert FrameStatus.PROMOTED.value == "promoted"
    assert FrameStatus.INVALIDATED.value == "invalidated"
    assert FrameStatus.SUSPENDED.value == "suspended"
    assert FrameStatus.UNCERTAIN.value == "uncertain"


def test_frame_status_count():
    """Should have exactly 8 status values."""
    assert len(FrameStatus) == 8
