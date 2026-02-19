"""Tests for plugin interface."""

from src.plugin_interface import CoreContext, PluginError


def test_plugin_error_creation():
    """PluginError should be creatable with all fields."""
    error = PluginError("Something went wrong")
    error.recoverable = True
    error.frame_id = "frame-123"
    error.reason = "Test error"

    assert str(error) == "Something went wrong"
    assert error.recoverable is True
    assert error.frame_id == "frame-123"
    assert error.reason == "Test error"


def test_plugin_error_default_values():
    """PluginError should have sensible defaults."""
    error = PluginError("Error")

    assert error.recoverable is False
    assert error.frame_id is None
    assert error.reason is None


def test_core_context_typedef():
    """CoreContext should be a valid TypedDict."""
    ctx: CoreContext = {
        "current_frame": None,
        "index": {},
        "artifacts": None,
        "changed_files": [],
        "invalidated_frames": [],
        "suspended_frames": [],
        "confidence_threshold": 0.7
    }
    assert ctx["confidence_threshold"] == 0.7
    assert ctx["changed_files"] == []
    assert ctx["index"] == {}


def test_core_context_with_data():
    """CoreContext should accept frame data."""
    ctx: CoreContext = {
        "current_frame": None,
        "index": {"frame1": None},  # Would be CausalFrame in practice
        "artifacts": None,
        "changed_files": ["file1.py", "file2.py"],
        "invalidated_frames": ["old_frame"],
        "suspended_frames": ["suspended_frame"],
        "confidence_threshold": 0.5
    }
    assert len(ctx["changed_files"]) == 2
    assert "frame1" in ctx["index"]
