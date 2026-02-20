"""Tests for FrameIndex class."""

from datetime import datetime

from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.frame.frame_index import FrameIndex


def make_frame(
    frame_id: str,
    parent_id: str | None = None,
    status: FrameStatus = FrameStatus.RUNNING,
    branched_from: str | None = None
) -> CausalFrame:
    """Helper to create a CausalFrame for testing."""
    context = ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=1000)
    return CausalFrame(
        frame_id=frame_id,
        depth=0 if parent_id is None else 1,
        parent_id=parent_id,
        children=[],
        query=f"query for {frame_id}",
        context_slice=context,
        evidence=[],
        conclusion=None,
        confidence=0.8,
        invalidation_condition="test",
        status=status,
        branched_from=branched_from,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=None
    )


def test_frame_index_add_and_get():
    """FrameIndex should add and retrieve frames."""
    index = FrameIndex()
    frame = make_frame("frame1")
    index.add(frame)
    assert index.get("frame1") == frame
    assert index.get("nonexistent") is None


def test_frame_index_get_active_frames():
    """FrameIndex should filter frames by RUNNING status."""
    index = FrameIndex()
    index.add(make_frame("f1", status=FrameStatus.RUNNING))
    index.add(make_frame("f2", status=FrameStatus.COMPLETED))
    index.add(make_frame("f3", status=FrameStatus.RUNNING))

    active = index.get_active_frames()
    assert len(active) == 2
    assert all(f.status == FrameStatus.RUNNING for f in active)


def test_frame_index_get_suspended_frames():
    """FrameIndex should filter frames by SUSPENDED status."""
    index = FrameIndex()
    index.add(make_frame("f1", status=FrameStatus.SUSPENDED))
    index.add(make_frame("f2", status=FrameStatus.COMPLETED))
    index.add(make_frame("f3", status=FrameStatus.SUSPENDED))

    suspended = index.get_suspended_frames()
    assert len(suspended) == 2


def test_frame_index_get_pivots():
    """FrameIndex should find frames that branched from another frame."""
    index = FrameIndex()
    f1 = make_frame("f1")
    f2 = make_frame("f2", branched_from="f1")  # This is a pivot
    index.add(f1)
    index.add(f2)
    index.add(make_frame("f3"))  # Not a pivot

    pivots = index.get_pivots()
    assert len(pivots) == 1
    assert pivots[0].frame_id == "f2"


def test_frame_index_contains():
    """FrameIndex should support 'in' operator."""
    index = FrameIndex()
    frame = make_frame("frame1")
    index.add(frame)
    assert "frame1" in index
    assert "nonexistent" not in index


def test_frame_index_len():
    """FrameIndex should support len()."""
    index = FrameIndex()
    assert len(index) == 0
    index.add(make_frame("f1"))
    index.add(make_frame("f2"))
    assert len(index) == 2


def test_frame_index_find_by_parent():
    """FrameIndex.find_by_parent returns children of a frame."""
    index = FrameIndex()
    parent = make_frame("parent", status=FrameStatus.COMPLETED)
    child1 = make_frame("child1", parent_id="parent", status=FrameStatus.COMPLETED)
    child2 = make_frame("child2", parent_id="parent", status=FrameStatus.RUNNING)
    other = make_frame("other", status=FrameStatus.COMPLETED)

    index.add(parent)
    index.add(child1)
    index.add(child2)
    index.add(other)

    children = index.find_by_parent("parent")
    assert len(children) == 2
    child_ids = {f.frame_id for f in children}
    assert child_ids == {"child1", "child2"}


def test_frame_index_find_by_parent_no_children():
    """FrameIndex.find_by_parent returns empty list if no children."""
    index = FrameIndex()
    frame = make_frame("frame1")
    index.add(frame)

    children = index.find_by_parent("frame1")
    assert children == []


def test_frame_index_find_promoted():
    """FrameIndex.find_promoted returns PROMOTED frames."""
    index = FrameIndex()
    promoted1 = make_frame("p1", status=FrameStatus.PROMOTED)
    promoted2 = make_frame("p2", status=FrameStatus.PROMOTED)
    running = make_frame("r1", status=FrameStatus.RUNNING)

    index.add(promoted1)
    index.add(promoted2)
    index.add(running)

    promoted = index.find_promoted()
    assert len(promoted) == 2
    promoted_ids = {f.frame_id for f in promoted}
    assert promoted_ids == {"p1", "p2"}
