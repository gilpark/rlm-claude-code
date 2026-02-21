"""Tests for frame tree structure."""
import pytest
from src.frame.frame_index import FrameIndex
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from datetime import datetime


def make_frame(frame_id: str, parent_id: str = None) -> CausalFrame:
    """Helper to create test frames."""
    return CausalFrame(
        frame_id=frame_id,
        depth=0 if not parent_id else 1,
        parent_id=parent_id,
        children=[],
        query=f"query_{frame_id}",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion=f"conclusion_{frame_id}",
        confidence=0.8,
        invalidation_condition={},
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )


def test_add_child_updates_parent_children_list():
    """FrameIndex.add should update parent's children list."""
    index = FrameIndex()

    parent = make_frame("parent789")
    index.add(parent)

    child = make_frame("child111", parent_id="parent789")
    index.add(child)

    # Parent's children should now include child
    parent_after = index.get("parent789")
    assert "child111" in parent_after.children


def test_tree_structure_multiple_children():
    """Parent should track all children."""
    index = FrameIndex()

    parent = make_frame("root")
    index.add(parent)

    for i in range(3):
        child = make_frame(f"child_{i}", parent_id="root")
        index.add(child)

    parent_after = index.get("root")
    assert len(parent_after.children) == 3
    assert "child_0" in parent_after.children
    assert "child_1" in parent_after.children
    assert "child_2" in parent_after.children


def test_add_child_idempotent():
    """Adding same child twice shouldn't duplicate."""
    index = FrameIndex()

    parent = make_frame("parent_dup")
    index.add(parent)

    child = make_frame("child_dup", parent_id="parent_dup")
    index.add(child)
    index.add(child)  # Add again (shouldn't duplicate)

    parent_after = index.get("parent_dup")
    assert parent_after.children.count("child_dup") == 1
