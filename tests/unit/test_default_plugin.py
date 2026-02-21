"""Tests for default plugin."""

from datetime import datetime

from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.plugins.default_plugin import DefaultRLMPlugin
from src.plugin_interface import CoreContext
from src.session.session_artifacts import SessionArtifacts, FileRecord


def make_test_frame() -> CausalFrame:
    """Helper to create a CausalFrame for testing."""
    context = ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=1000)
    return CausalFrame(
        frame_id="test",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=context,
        evidence=[],
        conclusion=None,
        confidence=0.8,
        invalidation_condition={},
        status=FrameStatus.RUNNING,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=None
    )


def make_test_context() -> CoreContext:
    """Helper to create a CoreContext for testing."""
    return {
        "current_frame": make_test_frame(),
        "index": {},
        "artifacts": None,
        "changed_files": [],
        "invalidated_frames": [],
        "suspended_frames": [],
        "confidence_threshold": 0.7
    }


def test_default_plugin_transform_input():
    """Default plugin should transform input."""
    plugin = DefaultRLMPlugin()
    ctx = make_test_context()

    result = plugin.transform_input({"query": "test"}, ctx)

    assert isinstance(result, dict)
    assert "query" in result


def test_default_plugin_transform_input_adds_suspended():
    """Default plugin should add suspended_frames to input."""
    plugin = DefaultRLMPlugin()
    ctx = make_test_context()
    ctx["suspended_frames"] = ["frame1", "frame2"]

    result = plugin.transform_input({"query": "test"}, ctx)

    assert "suspended_frames" in result
    assert result["suspended_frames"] == ["frame1", "frame2"]


def test_default_plugin_parse_output():
    """Default plugin should parse output."""
    plugin = DefaultRLMPlugin()
    frame = make_test_frame()

    result = plugin.parse_output("Conclusion: test result", frame)

    assert isinstance(result, dict)
    assert "raw_output" in result
    assert "conclusion" in result


def test_default_plugin_store():
    """Default plugin should store parsed output."""
    plugin = DefaultRLMPlugin()
    frame = make_test_frame()
    frame.conclusion = None

    plugin.store({"conclusion": "new conclusion"}, frame)

    assert frame.conclusion == "new conclusion"


def test_default_plugin_store_ignores_non_dict():
    """Default plugin should ignore non-dict parsed output."""
    plugin = DefaultRLMPlugin()
    frame = make_test_frame()
    frame.conclusion = "original"

    plugin.store("not a dict", frame)

    assert frame.conclusion == "original"
