"""Tests for evidence tracking in FrameIndex.add."""
import pytest
from src.frame.frame_index import FrameIndex
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from datetime import datetime


def make_frame(frame_id: str, parent_id: str = None, status: FrameStatus = FrameStatus.COMPLETED) -> CausalFrame:
    """Helper to create test frames with configurable status."""
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
        status=status,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now() if status == FrameStatus.COMPLETED else None,
    )


def test_child_frame_added_to_parent_evidence():
    """Completed child should be added to parent's evidence list."""
    index = FrameIndex()

    parent = make_frame("parent123")
    index.add(parent)

    child = make_frame("child456", parent_id="parent123", status=FrameStatus.COMPLETED)
    index.add(child)

    # Parent's evidence should now include the completed child
    parent_after = index.get("parent123")
    assert "child456" in parent_after.evidence


def test_invalidated_child_not_added_to_evidence():
    """Child with non-COMPLETED status should NOT be added to parent's evidence."""
    index = FrameIndex()

    parent = make_frame("parent789")
    index.add(parent)

    # Test each non-COMPLETED status
    non_completed_statuses = [
        FrameStatus.RUNNING,
        FrameStatus.SUSPENDED,
        FrameStatus.INVALIDATED,
        FrameStatus.PROMOTED,
    ]

    for status in non_completed_statuses:
        child = make_frame(f"child_{status.value}", parent_id="parent789", status=status)
        index.add(child)

    parent_after = index.get("parent789")
    # None of the non-completed children should be in evidence
    assert "child_RUNNING" not in parent_after.evidence
    assert "child_suspended" not in parent_after.evidence
    assert "child_invalidated" not in parent_after.evidence
    assert "child_promoted" not in parent_after.evidence


def test_evidence_tracking_idempotent():
    """Adding the same completed child twice shouldn't duplicate in evidence."""
    index = FrameIndex()

    parent = make_frame("parent_dup")
    index.add(parent)

    child = make_frame("child_dup", parent_id="parent_dup", status=FrameStatus.COMPLETED)
    index.add(child)
    index.add(child)  # Add again (shouldn't duplicate)

    parent_after = index.get("parent_dup")
    # Child should appear exactly once in evidence
    assert parent_after.evidence.count("child_dup") == 1


def test_evidence_separate_from_children():
    """Evidence list should be separate from children list."""
    index = FrameIndex()

    parent = make_frame("parent_sep")
    index.add(parent)

    completed_child = make_frame("child_completed", parent_id="parent_sep", status=FrameStatus.COMPLETED)
    running_child = make_frame("child_running", parent_id="parent_sep", status=FrameStatus.RUNNING)
    invalidated_child = make_frame("child_invalidated", parent_id="parent_sep", status=FrameStatus.INVALIDATED)

    index.add(completed_child)
    index.add(running_child)
    index.add(invalidated_child)

    parent_after = index.get("parent_sep")

    # All children should be in children list
    assert len(parent_after.children) == 3
    assert "child_completed" in parent_after.children
    assert "child_running" in parent_after.children
    assert "child_invalidated" in parent_after.children

    # Only COMPLETED child should be in evidence
    assert len(parent_after.evidence) == 1
    assert "child_completed" in parent_after.evidence


def test_nested_evidence_tracking():
    """Evidence should track correctly across multiple levels."""
    index = FrameIndex()

    root = make_frame("root")
    index.add(root)

    parent = make_frame("parent", parent_id="root", status=FrameStatus.COMPLETED)
    index.add(parent)

    child = make_frame("child", parent_id="parent", status=FrameStatus.COMPLETED)
    index.add(child)

    # Root should have parent in evidence
    root_after = index.get("root")
    assert "parent" in root_after.evidence

    # Parent should have child in evidence
    parent_after = index.get("parent")
    assert "child" in parent_after.evidence

    # But root should NOT have grandchild directly in evidence
    assert "child" not in root_after.evidence


def test_child_updated_to_completed_adds_to_evidence():
    """When a child frame's status changes to COMPLETED, it should be added to evidence."""
    index = FrameIndex()

    parent = make_frame("parent_update")
    index.add(parent)

    # Add child as RUNNING first
    child = make_frame("child_update", parent_id="parent_update", status=FrameStatus.RUNNING)
    index.add(child)

    parent_after = index.get("parent_update")
    assert "child_update" not in parent_after.evidence

    # Update child to COMPLETED and re-add
    child.status = FrameStatus.COMPLETED
    child.completed_at = datetime.now()
    index.add(child)

    parent_after = index.get("parent_update")
    # Now the child should be in evidence
    assert "child_update" in parent_after.evidence
