"""Tests for simplified CausalFrame."""

import pytest
from src.frame.causal_frame import CausalFrame, FrameStatus, compute_frame_id
from src.frame.context_slice import ContextSlice


class TestSimplifiedFrameStatus:
    """Test simplified FrameStatus enum."""

    def test_status_running_exists(self):
        """FrameStatus.RUNNING exists."""
        assert FrameStatus.RUNNING.value == "running"

    def test_status_completed_exists(self):
        """FrameStatus.COMPLETED exists."""
        assert FrameStatus.COMPLETED.value == "completed"

    def test_status_suspended_exists(self):
        """FrameStatus.SUSPENDED exists."""
        assert FrameStatus.SUSPENDED.value == "suspended"

    def test_status_invalidated_exists(self):
        """FrameStatus.INVALIDATED exists."""
        assert FrameStatus.INVALIDATED.value == "invalidated"

    def test_status_promoted_exists(self):
        """FrameStatus.PROMOTED exists."""
        assert FrameStatus.PROMOTED.value == "promoted"

    def test_status_count(self):
        """FrameStatus has exactly 5 values."""
        assert len(FrameStatus) == 5
