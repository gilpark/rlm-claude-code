"""End-to-End integration test for tree cascade, evidence tracking, and query persistence.

This module tests the full integration of:
- Frame tree construction with parent-child relationships
- Evidence tracking (frames citing other frames as evidence)
- Cascade invalidation (downward to children, sideways to evidence users)
- Query tracking persistence across save/load

The test scenarios cover:
1. Full tree cascade invalidation - invalidate root, entire tree invalidated
2. Partial tree invalidation - invalidate one branch, sibling unaffected
3. Query tracking persistence - initial_query and query_summary persist across save/load
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.frame.frame_index import FrameIndex
from src.frame.frame_invalidation import propagate_invalidation


def make_frame(
    frame_id: str,
    parent_id: str | None = None,
    children: list[str] | None = None,
    evidence: list[str] | None = None,
    depth: int = 0,
) -> CausalFrame:
    """Helper to create a CausalFrame for testing.

    Args:
        frame_id: Unique identifier for the frame
        parent_id: Optional parent frame ID
        children: Optional list of child frame IDs
        evidence: Optional list of evidence frame IDs this frame cites
        depth: Depth in the tree (0 = root)

    Returns:
        A CausalFrame with default values for testing
    """
    context = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000
    )
    return CausalFrame(
        frame_id=frame_id,
        depth=depth,
        parent_id=parent_id,
        children=children or [],
        query=f"query for {frame_id}",
        context_slice=context,
        evidence=evidence or [],
        conclusion=f"conclusion for {frame_id}",
        confidence=0.8,
        invalidation_condition=f"test invalidation for {frame_id}",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now()
    )


def test_full_tree_cascade_invalidation():
    """Test cascade invalidation propagates through entire tree and evidence network.

    Tree structure:
        root
        ├── child1
        │   └── leaf1
        └── child2
        evidence_user (cites leaf1 as evidence)

    Invalidating root should cascade to:
    - child1 (child of root)
    - leaf1 (child of child1)
    - child2 (child of root)
    - evidence_user (uses leaf1 as evidence)

    Total: 5 frames invalidated
    """
    index = FrameIndex()

    # Build tree: root -> child1 -> leaf1
    root = make_frame("root", children=["child1", "child2"], depth=0)
    child1 = make_frame("child1", parent_id="root", children=["leaf1"], depth=1)
    leaf1 = make_frame("leaf1", parent_id="child1", depth=2)
    child2 = make_frame("child2", parent_id="root", depth=1)

    # evidence_user cites leaf1 as evidence
    evidence_user = make_frame(
        "evidence_user",
        evidence=["leaf1"],
        depth=0  # Can be at any depth, evidence is independent
    )

    # Add all frames to index
    for f in [root, child1, leaf1, child2, evidence_user]:
        index.add(f)

    # Invalidate root
    invalidated = propagate_invalidation("root", "root changed", index)

    # Assert all 5 frames are invalidated
    assert len(invalidated) == 5
    assert "root" in invalidated
    assert "child1" in invalidated
    assert "leaf1" in invalidated
    assert "child2" in invalidated
    assert "evidence_user" in invalidated

    # Verify all frames have INVALIDATED status
    assert index.get("root").status == FrameStatus.INVALIDATED
    assert index.get("child1").status == FrameStatus.INVALIDATED
    assert index.get("leaf1").status == FrameStatus.INVALIDATED
    assert index.get("child2").status == FrameStatus.INVALIDATED
    assert index.get("evidence_user").status == FrameStatus.INVALIDATED

    # Verify escalation reasons propagate
    # Root has direct reason
    assert index.get("root").escalation_reason == "root changed"

    # Descendants have cascaded reason (includes "Parent invalidated:" prefix)
    assert "root changed" in index.get("child1").escalation_reason
    assert "root changed" in index.get("leaf1").escalation_reason
    assert "root changed" in index.get("child2").escalation_reason

    # Evidence user gets invalidated because leaf1 (which it cites) was invalidated
    assert "leaf1" in index.get("evidence_user").escalation_reason


def test_partial_tree_invalidation():
    """Test invalidating middle frame cascades both down and up via evidence.

    Tree structure:
        root
        ├── branch1
        │   ├── leaf1a
        │   └── leaf1b
        └── branch2
            └── leaf2a

    With auto-evidence tracking (B3), parents automatically track children as evidence.

    Invalidating branch1 should:
    - Invalidate branch1 (direct)
    - Invalidate leaf1a, leaf1b (children of branch1)
    - Invalidate root (parent, via auto-evidence tracking)
    - Invalidate branch2 (root's child, via root's invalidation)
    - Invalidate leaf2a (child of branch2, via branch2's invalidation)

    Total: all 6 frames invalidated due to evidence-based cascade.
    """
    index = FrameIndex()

    # Build tree with two branches
    root = make_frame("root", children=["branch1", "branch2"], depth=0)
    branch1 = make_frame("branch1", parent_id="root", children=["leaf1a", "leaf1b"], depth=1)
    leaf1a = make_frame("leaf1a", parent_id="branch1", depth=2)
    leaf1b = make_frame("leaf1b", parent_id="branch1", depth=2)
    branch2 = make_frame("branch2", parent_id="root", children=["leaf2a"], depth=1)
    leaf2a = make_frame("leaf2a", parent_id="branch2", depth=2)

    # Add all frames (this also auto-populates evidence for parents)
    for f in [root, branch1, leaf1a, leaf1b, branch2, leaf2a]:
        index.add(f)

    # Verify auto-evidence tracking: parents have children in evidence
    assert "branch1" in index.get("root").evidence
    assert "branch2" in index.get("root").evidence
    assert "leaf1a" in index.get("branch1").evidence
    assert "leaf1b" in index.get("branch1").evidence
    assert "leaf2a" in index.get("branch2").evidence

    # Invalidate branch1 (not root)
    invalidated = propagate_invalidation("branch1", "branch1 changed", index)

    # All frames invalidated due to evidence cascade
    assert len(invalidated) == 6
    assert "root" in invalidated
    assert "branch1" in invalidated
    assert "branch2" in invalidated
    assert "leaf1a" in invalidated
    assert "leaf1b" in invalidated
    assert "leaf2a" in invalidated

    # Verify all statuses
    assert index.get("root").status == FrameStatus.INVALIDATED
    assert index.get("branch1").status == FrameStatus.INVALIDATED
    assert index.get("leaf1a").status == FrameStatus.INVALIDATED
    assert index.get("leaf1b").status == FrameStatus.INVALIDATED
    assert index.get("branch2").status == FrameStatus.INVALIDATED
    assert index.get("leaf2a").status == FrameStatus.INVALIDATED


def test_query_tracking_persists(tmp_path: Path):
    """Test that initial_query and query_summary persist across save/load.

    This verifies the query tracking feature added to FrameIndex.
    """
    # Create index with query fields
    index = FrameIndex(
        initial_query="Analyze the authentication module for security vulnerabilities",
        query_summary="Auth security analysis",
        commit_hash="abc123"
    )

    # Add a frame
    frame = make_frame("frame1")
    index.add(frame)

    # Save to temp directory
    session_id = "test_query_session"
    index.save(session_id, tmp_path)

    # Load from temp directory
    loaded_index = FrameIndex.load(session_id, tmp_path)

    # Assert query fields persisted
    assert loaded_index.initial_query == "Analyze the authentication module for security vulnerabilities"
    assert loaded_index.query_summary == "Auth security analysis"
    assert loaded_index.commit_hash == "abc123"

    # Assert frame also persisted
    assert loaded_index.get("frame1") is not None
    assert loaded_index.get("frame1").query == "query for frame1"


def test_evidence_only_invalidation():
    """Test invalidation propagates through evidence chain without parent-child relationship.

    Chain:
        f1 (independent)
        f2 (cites f1 as evidence)
        f3 (cites f2 as evidence)
        f4 (independent, unrelated)

    Invalidating f1 should cascade to f2 and f3, but not f4.
    """
    index = FrameIndex()

    # Create evidence chain
    f1 = make_frame("f1")
    f2 = make_frame("f2", evidence=["f1"])
    f3 = make_frame("f3", evidence=["f2"])
    f4 = make_frame("f4")  # Unrelated

    for f in [f1, f2, f3, f4]:
        index.add(f)

    # Invalidate f1
    invalidated = propagate_invalidation("f1", "f1 changed", index)

    # Assert f1, f2, f3 invalidated, but not f4
    assert len(invalidated) == 3
    assert "f1" in invalidated
    assert "f2" in invalidated
    assert "f3" in invalidated
    assert "f4" not in invalidated

    # Verify statuses
    assert index.get("f1").status == FrameStatus.INVALIDATED
    assert index.get("f2").status == FrameStatus.INVALIDATED
    assert index.get("f3").status == FrameStatus.INVALIDATED
    assert index.get("f4").status == FrameStatus.COMPLETED


def test_mixed_tree_and_evidence_cascade():
    """Test cascade with both parent-child and evidence relationships.

    Tree:
        root
        ├── child1
        │   └── leaf1
        └── child2

    Evidence:
        evidence_user1 cites leaf1
        evidence_user2 cites child2

    Invalidating root should cascade to:
    - child1, leaf1 (via parent-child)
    - child2 (via parent-child)
    - evidence_user1 (via leaf1 evidence)
    - evidence_user2 (via child2 evidence)
    """
    index = FrameIndex()

    # Build tree
    root = make_frame("root", children=["child1", "child2"], depth=0)
    child1 = make_frame("child1", parent_id="root", children=["leaf1"], depth=1)
    leaf1 = make_frame("leaf1", parent_id="child1", depth=2)
    child2 = make_frame("child2", parent_id="root", depth=1)

    # Evidence users
    evidence_user1 = make_frame("evidence_user1", evidence=["leaf1"])
    evidence_user2 = make_frame("evidence_user2", evidence=["child2"])

    for f in [root, child1, leaf1, child2, evidence_user1, evidence_user2]:
        index.add(f)

    # Invalidate root
    invalidated = propagate_invalidation("root", "root changed", index)

    # All frames should be invalidated
    assert len(invalidated) == 6
    assert "root" in invalidated
    assert "child1" in invalidated
    assert "leaf1" in invalidated
    assert "child2" in invalidated
    assert "evidence_user1" in invalidated
    assert "evidence_user2" in invalidated

    # All should have INVALIDATED status
    for frame_id in ["root", "child1", "leaf1", "child2", "evidence_user1", "evidence_user2"]:
        assert index.get(frame_id).status == FrameStatus.INVALIDATED


def test_leaf_invalidation_with_evidence():
    """Test invalidating a leaf frame cascades up via evidence.

    Tree:
        root
        ├── child1
        │   └── leaf1 (has evidence user)
        └── child2

    With auto-evidence tracking (B3), parents automatically track children as evidence.

    Invalidating leaf1 should:
    - Invalidate leaf1 (direct)
    - Invalidate child1 (parent, via auto-evidence tracking)
    - Invalidate root (via child1's invalidation and auto-evidence)
    - Invalidate child2 (via root's invalidation)
    - Invalidate evidence_user1 (cites leaf1 as evidence)

    Total: 5 frames invalidated due to evidence-based cascade.
    """
    index = FrameIndex()

    root = make_frame("root", children=["child1", "child2"], depth=0)
    child1 = make_frame("child1", parent_id="root", children=["leaf1"], depth=1)
    leaf1 = make_frame("leaf1", parent_id="child1", depth=2)
    child2 = make_frame("child2", parent_id="root", depth=1)
    evidence_user1 = make_frame("evidence_user1", evidence=["leaf1"])

    for f in [root, child1, leaf1, child2, evidence_user1]:
        index.add(f)

    # Invalidate leaf1
    invalidated = propagate_invalidation("leaf1", "leaf1 changed", index)

    # All frames except evidence_user1 (which cites leaf1) cascade via evidence
    assert len(invalidated) == 5
    assert "leaf1" in invalidated
    assert "child1" in invalidated  # Parent, via auto-evidence
    assert "root" in invalidated  # Via child1's invalidation
    assert "child2" in invalidated  # Via root's invalidation
    assert "evidence_user1" in invalidated  # Explicit evidence

    # Verify all statuses
    assert index.get("root").status == FrameStatus.INVALIDATED
    assert index.get("child1").status == FrameStatus.INVALIDATED
    assert index.get("leaf1").status == FrameStatus.INVALIDATED
    assert index.get("child2").status == FrameStatus.INVALIDATED
    assert index.get("evidence_user1").status == FrameStatus.INVALIDATED


def test_cascade_without_auto_evidence():
    """Test cascade behavior when frames are added manually without auto-evidence.

    This test creates frames and directly sets them in the index without using
    the add() method, to demonstrate the difference in cascade behavior.

    Without auto-evidence tracking:
    - Invalidating a child only cascades downward to its children
    - Parents are NOT affected (cascade is downward-only)
    """
    index = FrameIndex()

    # Create frames with manual children lists (no auto-evidence)
    root = make_frame("root", children=["child1", "child2"], depth=0)
    child1 = make_frame("child1", parent_id="root", children=["leaf1"], depth=1)
    leaf1 = make_frame("leaf1", parent_id="child1", depth=2)
    child2 = make_frame("child2", parent_id="root", children=["leaf2"], depth=1)
    leaf2 = make_frame("leaf2", parent_id="child2", depth=2)

    # Manually add to index (bypassing add() which sets auto-evidence)
    for f in [root, child1, leaf1, child2, leaf2]:
        index._frames[f.frame_id] = f

    # Verify no auto-evidence was set
    assert index.get("root").evidence == []
    assert index.get("child1").evidence == []
    assert index.get("child2").evidence == []

    # Invalidate leaf1
    invalidated = propagate_invalidation("leaf1", "leaf1 changed", index)

    # Only leaf1 should be invalidated (no children, no evidence users)
    assert len(invalidated) == 1
    assert "leaf1" in invalidated

    # Parent and sibling should NOT be invalidated
    assert index.get("root").status == FrameStatus.COMPLETED
    assert index.get("child1").status == FrameStatus.COMPLETED
    assert index.get("child2").status == FrameStatus.COMPLETED
    assert index.get("leaf2").status == FrameStatus.COMPLETED
    assert index.get("leaf1").status == FrameStatus.INVALIDATED
