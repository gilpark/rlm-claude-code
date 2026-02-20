"""Tests for find_dependent_frames with caching."""

import pytest
from datetime import datetime

from src.frame.frame_index import FrameIndex
from src.frame.context_slice import ContextSlice
from src.frame.causal_frame import CausalFrame, FrameStatus


def _make_frame(
    frame_id: str,
    parent_id: str | None = None,
    evidence: list[str] | None = None,
    status: FrameStatus = FrameStatus.COMPLETED,
) -> CausalFrame:
    """Helper to create a test frame."""
    return CausalFrame(
        frame_id=frame_id,
        depth=0,
        parent_id=parent_id,
        children=[],
        query=f"query for {frame_id}",
        context_slice=ContextSlice(
            files={}, memory_refs=[], tool_outputs={}, token_budget=8000
        ),
        evidence=evidence or [],
        conclusion=f"conclusion for {frame_id}",
        confidence=0.8,
        invalidation_condition="",
        status=status,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )


def test_find_dependent_frames_by_evidence():
    """Finds frames citing this frame in evidence."""
    index = FrameIndex()

    # Create a parent frame
    parent = _make_frame("parent1")
    index.add(parent)

    # Create child frames that cite parent in evidence
    child1 = _make_frame("child1", parent_id="parent1", evidence=["parent1"])
    child2 = _make_frame("child2", parent_id="parent1", evidence=["parent1", "other"])

    index.add(child1)
    index.add(child2)

    dependents = index.find_dependent_frames("parent1")

    # Should include both children (they cite parent in evidence)
    assert "child1" in dependents
    assert "child2" in dependents
    assert len(dependents) == 2


def test_find_dependent_frames_includes_children():
    """Finds child frames even if not in evidence."""
    index = FrameIndex()

    parent = _make_frame("parent2")
    index.add(parent)

    # Create a child that doesn't cite parent in evidence
    child = _make_frame("child3", parent_id="parent2", evidence=[])

    index.add(child)

    dependents = index.find_dependent_frames("parent2")

    # Should include child by parent relationship
    assert "child3" in dependents
    assert len(dependents) == 1


def test_find_dependent_frames_none():
    """Returns empty set if no dependents."""
    index = FrameIndex()

    isolated = _make_frame("isolated")
    index.add(isolated)

    dependents = index.find_dependent_frames("isolated")

    assert dependents == set()


def test_find_dependent_frames_caching():
    """Caches results and returns copy (cache isolation)."""
    index = FrameIndex()

    parent = _make_frame("parent3")
    index.add(parent)

    child = _make_frame("child4", parent_id="parent3", evidence=["parent3"])
    index.add(child)

    # First call populates cache
    dependents1 = index.find_dependent_frames("parent3")
    assert dependents1 == {"child4"}

    # Second call should return cached value
    dependents2 = index.find_dependent_frames("parent3")
    assert dependents2 == {"child4"}

    # Modifying returned set should not affect cache
    dependents1.add("fake_id")
    dependents3 = index.find_dependent_frames("parent3")
    assert dependents3 == {"child4"}  # Cache unchanged

    # Adding a new frame should invalidate cache
    new_child = _make_frame("child5", parent_id="parent3", evidence=["parent3"])
    index.add(new_child)

    dependents4 = index.find_dependent_frames("parent3")
    assert dependents4 == {"child4", "child5"}  # Cache rebuilt
