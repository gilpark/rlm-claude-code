"""Tests for FrameLifecycle class."""

from src.causal_frame import FrameStatus
from src.frame_lifecycle import FrameLifecycle


def test_lifecycle_valid_transitions():
    """Valid transitions should return True."""
    lifecycle = FrameLifecycle()

    # CREATED → RUNNING is valid
    assert lifecycle.can_transition(FrameStatus.CREATED, FrameStatus.RUNNING)

    # RUNNING → COMPLETED is valid
    assert lifecycle.can_transition(FrameStatus.RUNNING, FrameStatus.COMPLETED)

    # RUNNING → INVALIDATED is valid
    assert lifecycle.can_transition(FrameStatus.RUNNING, FrameStatus.INVALIDATED)

    # COMPLETED → VERIFIED is valid
    assert lifecycle.can_transition(FrameStatus.COMPLETED, FrameStatus.VERIFIED)

    # VERIFIED → PROMOTED is valid
    assert lifecycle.can_transition(FrameStatus.VERIFIED, FrameStatus.PROMOTED)


def test_lifecycle_terminal_states():
    """Terminal states should not allow transitions."""
    lifecycle = FrameLifecycle()

    # PROMOTED is terminal
    assert not lifecycle.can_transition(FrameStatus.PROMOTED, FrameStatus.RUNNING)
    assert not lifecycle.can_transition(FrameStatus.PROMOTED, FrameStatus.INVALIDATED)

    # INVALIDATED is terminal
    assert not lifecycle.can_transition(FrameStatus.INVALIDATED, FrameStatus.RUNNING)


def test_lifecycle_invalid_transitions():
    """Invalid transitions should return False."""
    lifecycle = FrameLifecycle()

    # CREATED → COMPLETED is invalid (skip RUNNING)
    assert not lifecycle.can_transition(FrameStatus.CREATED, FrameStatus.COMPLETED)

    # RUNNING → CREATED is invalid (no backward)
    assert not lifecycle.can_transition(FrameStatus.RUNNING, FrameStatus.CREATED)

    # COMPLETED → CREATED is invalid (no backward)
    assert not lifecycle.can_transition(FrameStatus.COMPLETED, FrameStatus.CREATED)


def test_lifecycle_suspended_can_resume():
    """SUSPENDED frames can resume to RUNNING."""
    lifecycle = FrameLifecycle()

    assert lifecycle.can_transition(FrameStatus.SUSPENDED, FrameStatus.RUNNING)
    assert lifecycle.can_transition(FrameStatus.SUSPENDED, FrameStatus.INVALIDATED)


def test_lifecycle_running_to_suspended():
    """RUNNING can transition to SUSPENDED."""
    lifecycle = FrameLifecycle()

    assert lifecycle.can_transition(FrameStatus.RUNNING, FrameStatus.SUSPENDED)
