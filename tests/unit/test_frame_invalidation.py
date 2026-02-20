"""Tests for frame invalidation."""

from datetime import datetime

from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.frame.frame_index import FrameIndex
from src.frame.frame_invalidation import propagate_invalidation


def make_frame_with_children(
    frame_id: str,
    parent_id: str | None = None,
    children: list[str] | None = None,
    evidence: list[str] | None = None
) -> CausalFrame:
    """Helper to create a CausalFrame for testing."""
    context = ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=1000)
    return CausalFrame(
        frame_id=frame_id,
        depth=0 if parent_id is None else 1,
        parent_id=parent_id,
        children=children or [],
        query=f"query for {frame_id}",
        context_slice=context,
        evidence=evidence or [],
        conclusion=None,
        confidence=0.8,
        invalidation_condition="test",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now()
    )


def test_invalidation_cascades_to_children():
    """Invalidating a frame should cascade to all children."""
    index = FrameIndex()
    root = make_frame_with_children("root", children=["child1", "child2"])
    child1 = make_frame_with_children("child1", parent_id="root", children=["grandchild"])
    child2 = make_frame_with_children("child2", parent_id="root")
    grandchild = make_frame_with_children("grandchild", parent_id="child1")

    for f in [root, child1, child2, grandchild]:
        index.add(f)

    invalidated = propagate_invalidation("root", "test reason", index)

    assert "root" in invalidated
    assert "child1" in invalidated
    assert "child2" in invalidated
    assert "grandchild" in invalidated


def test_invalidation_cascades_to_evidence_users():
    """Invalidating a frame should cascade to frames using it as evidence."""
    index = FrameIndex()
    f1 = make_frame_with_children("f1")
    f2 = make_frame_with_children("f2", evidence=["f1"])  # f2 uses f1 as evidence
    f3 = make_frame_with_children("f3", evidence=["f2"])  # f3 uses f2 as evidence

    for f in [f1, f2, f3]:
        index.add(f)

    invalidated = propagate_invalidation("f1", "test reason", index)

    assert "f1" in invalidated
    assert "f2" in invalidated  # Used f1 as evidence
    assert "f3" in invalidated  # Used f2 as evidence


def test_invalidation_sets_status():
    """Invalidated frames should have INVALIDATED status."""
    index = FrameIndex()
    root = make_frame_with_children("root", children=["child"])
    child = make_frame_with_children("child", parent_id="root")

    index.add(root)
    index.add(child)

    propagate_invalidation("root", "test reason", index)

    assert index.get("root").status == FrameStatus.INVALIDATED
    assert index.get("child").status == FrameStatus.INVALIDATED


def test_invalidation_sets_escalation_reason():
    """Invalidated frames should have escalation_reason set."""
    index = FrameIndex()
    frame = make_frame_with_children("frame")
    index.add(frame)

    propagate_invalidation("frame", "test reason", index)

    assert index.get("frame").escalation_reason == "test reason"
