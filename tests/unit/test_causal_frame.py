"""Tests for CausalFrame and related types."""

from datetime import datetime

from src.causal_frame import CausalFrame, FrameStatus
from src.context_slice import ContextSlice


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


def test_causal_frame_creation():
    """CausalFrame should be creatable with all fields."""
    context = ContextSlice(
        files={"test.py": "abc123"},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000
    )
    frame = CausalFrame(
        frame_id="test123",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=context,
        evidence=[],
        conclusion=None,
        confidence=0.8,
        invalidation_condition="always valid",
        status=FrameStatus.CREATED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=None
    )
    assert frame.frame_id == "test123"
    assert frame.depth == 0
    assert frame.status == FrameStatus.CREATED
    assert frame.confidence == 0.8


def test_causal_frame_with_branch_fields():
    """CausalFrame should support branch management fields."""
    context = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000
    )
    frame = CausalFrame(
        frame_id="child1",
        depth=1,
        parent_id="root",
        children=[],
        query="child query",
        context_slice=context,
        evidence=["root"],
        conclusion="result",
        confidence=0.6,
        invalidation_condition="test",
        status=FrameStatus.SUSPENDED,
        branched_from="root",
        escalation_reason="low confidence",
        created_at=datetime.now(),
        completed_at=datetime.now()
    )
    assert frame.branched_from == "root"
    assert frame.escalation_reason == "low confidence"
    assert frame.status == FrameStatus.SUSPENDED


# Tests for compute_frame_id
from src.causal_frame import compute_frame_id


def test_compute_frame_id_is_deterministic():
    """Same inputs should produce same frame_id."""
    context = ContextSlice(
        files={"a.py": "hash1"},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000
    )
    id1 = compute_frame_id("parent1", "query1", context)
    id2 = compute_frame_id("parent1", "query1", context)
    assert id1 == id2
    assert len(id1) == 16  # 16 hex chars


def test_compute_frame_id_differs_with_query():
    """Different query should produce different frame_id."""
    context = ContextSlice(
        files={"a.py": "hash1"},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000
    )
    id1 = compute_frame_id("parent1", "query1", context)
    id2 = compute_frame_id("parent1", "query2", context)
    assert id1 != id2


def test_compute_frame_id_handles_none_parent():
    """Root frames (None parent) should work."""
    context = ContextSlice(
        files={"a.py": "hash1"},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000
    )
    id1 = compute_frame_id(None, "query1", context)
    assert len(id1) == 16
