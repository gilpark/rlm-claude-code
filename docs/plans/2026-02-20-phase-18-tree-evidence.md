# Phase 18: Tree Structure + Evidence + Cascade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform linear frame chain into proper tree structure with evidence tracking and cascade invalidation.

**Architecture:** Populate `children` when creating frames, auto-track evidence from child frames and tool outputs, auto-generate `invalidation_condition`, implement full cascade propagation with dependent frame discovery.

**Tech Stack:** Python dataclasses, existing frame infrastructure

---

## Task 1: Populate Children in Frame Tree

**Files:**
- Modify: `src/repl/rlaph_loop.py:255-295` (frame creation section)
- Test: `tests/repl/test_frame_tree.py`

**Step 1: Write the failing test**

```python
"""Tests for frame tree structure."""
import pytest
from pathlib import Path
from src.repl.rlaph_loop import RLAPHLoop
from src.frame.frame_index import FrameIndex
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.types import SessionContext
from datetime import datetime


def test_frame_children_populated_on_creation():
    """When a child frame is created, parent.children should be updated."""
    index = FrameIndex()

    # Create parent frame
    parent = CausalFrame(
        frame_id="parent123",
        depth=0,
        parent_id=None,
        children=[],  # Empty initially
        query="parent query",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="parent conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(parent)

    # Create child frame
    child = CausalFrame(
        frame_id="child456",
        depth=1,
        parent_id="parent123",
        children=[],
        query="child query",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="child conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(child)

    # Update parent's children list (this is what we're testing should happen automatically)
    # index.add should update parent.children
    # But currently it doesn't - this test documents the expected behavior

    # For now, manually verify the expected state
    # After implementation, this should pass automatically


def test_add_child_updates_parent_children_list():
    """FrameIndex.add should update parent's children list."""
    from src.frame.frame_index import FrameIndex

    index = FrameIndex()

    parent = CausalFrame(
        frame_id="parent789",
        depth=0,
        parent_id=None,
        children=[],
        query="parent",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="p",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(parent)

    child = CausalFrame(
        frame_id="child111",
        depth=1,
        parent_id="parent789",
        children=[],
        query="child",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="c",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(child)

    # Parent's children should now include child
    parent_after = index.get("parent789")
    assert "child111" in parent_after.children


def test_tree_structure_multiple_children():
    """Parent should track all children."""
    from src.frame.frame_index import FrameIndex

    index = FrameIndex()

    parent = CausalFrame(
        frame_id="root",
        depth=0,
        parent_id=None,
        children=[],
        query="root",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="r",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(parent)

    for i in range(3):
        child = CausalFrame(
            frame_id=f"child_{i}",
            depth=1,
            parent_id="root",
            children=[],
            query=f"child {i}",
            context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
            evidence=[],
            conclusion=f"c{i}",
            confidence=0.8,
            invalidation_condition="",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )
        index.add(child)

    parent_after = index.get("root")
    assert len(parent_after.children) == 3
    assert "child_0" in parent_after.children
    assert "child_1" in parent_after.children
    assert "child_2" in parent_after.children
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/gilpark/.dotfiles/.claude/plugins/marketplaces/causeway
uv run pytest tests/repl/test_frame_tree.py -v
```

Expected: FAIL with "AssertionError: assert 'child111' in []"

**Step 3: Write minimal implementation**

In `src/frame/frame_index.py`, modify `add` method:

```python
def add(self, frame: "CausalFrame") -> None:
    """Add a frame to the index and update parent's children list."""
    self._frames[frame.frame_id] = frame

    # Update parent's children list
    if frame.parent_id and frame.parent_id in self._frames:
        parent = self._frames[frame.parent_id]
        if frame.frame_id not in parent.children:
            parent.children.append(frame.frame_id)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/repl/test_frame_tree.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/frame/frame_index.py tests/repl/test_frame_tree.py
git commit -m "feat: populate children list when adding frames to index

- FrameIndex.add updates parent's children list
- Enables proper tree structure traversal
- Foundation for cascade invalidation"
```

---

## Task 2: Auto-Generate invalidation_condition

**Files:**
- Modify: `src/repl/rlaph_loop.py:263-283` (frame creation)
- Modify: `src/frame/causal_frame.py` (add helper function)
- Test: `tests/frame/test_invalidation_condition.py`

**Step 1: Write the failing test**

```python
"""Tests for auto-generated invalidation_condition."""
import pytest
from src.frame.causal_frame import generate_invalidation_condition
from src.frame.context_slice import ContextSlice


def test_generate_invalidation_condition_with_files():
    """Should generate condition from files in context_slice."""
    context_slice = ContextSlice(
        files={
            "/path/to/file.py": "abc123",
            "/path/to/other.py": "def456",
        },
        memory_refs=[],
        tool_outputs={},
        token_budget=8000,
    )

    condition = generate_invalidation_condition(context_slice)

    assert "file.py" in condition or "file changes" in condition.lower()
    assert len(condition) > 0


def test_generate_invalidation_condition_empty():
    """Should return empty string for empty context_slice."""
    context_slice = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={},
        token_budget=8000,
    )

    condition = generate_invalidation_condition(context_slice)

    assert condition == ""


def test_generate_invalidation_condition_includes_tool_outputs():
    """Should mention tool outputs in condition."""
    context_slice = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={"Read": "hash123", "Bash": "hash456"},
        token_budget=8000,
    )

    condition = generate_invalidation_condition(context_slice)

    assert "tool" in condition.lower()


def test_invalidation_condition_format():
    """Should produce readable condition string."""
    context_slice = ContextSlice(
        files={"/src/main.py": "abc123"},
        memory_refs=[],
        tool_outputs={},
        token_budget=8000,
    )

    condition = generate_invalidation_condition(context_slice)

    # Should be a clear, readable statement
    assert isinstance(condition, str)
    assert len(condition) < 200  # Not too long
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/frame/test_invalidation_condition.py -v
```

Expected: FAIL with "ImportError: cannot import name 'generate_invalidation_condition'"

**Step 3: Write minimal implementation**

In `src/frame/causal_frame.py`, add function:

```python
def generate_invalidation_condition(context_slice: "ContextSlice") -> str:
    """
    Generate default invalidation condition from context_slice.

    The condition describes what would make this frame's conclusion
    no longer valid.

    Args:
        context_slice: The frame's context slice

    Returns:
        Human-readable invalidation condition string
    """
    parts = []

    if context_slice.files:
        if len(context_slice.files) == 1:
            file_path = list(context_slice.files.keys())[0]
            parts.append(f"Invalid if {Path(file_path).name} changes")
        else:
            parts.append(f"Invalid if any of {len(context_slice.files)} files change")

    if context_slice.tool_outputs:
        tool_names = list(context_slice.tool_outputs.keys())
        if len(tool_names) == 1:
            parts.append(f"tool {tool_names[0]} result changes")
        else:
            parts.append(f"{len(tool_names)} tool results change")

    if context_slice.memory_refs:
        parts.append(f"memory entries {context_slice.memory_refs} change")

    if not parts:
        return ""

    return "; ".join(parts)
```

Add import at top:
```python
from pathlib import Path
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/frame/test_invalidation_condition.py -v
```

Expected: PASS

**Step 5: Use in RLAPHLoop frame creation**

In `src/repl/rlaph_loop.py`, update frame creation:

```python
from ..frame.causal_frame import CausalFrame, FrameStatus, compute_frame_id, generate_invalidation_condition

# In frame creation:
frame = CausalFrame(
    # ... existing fields ...
    invalidation_condition=generate_invalidation_condition(context_slice),
    # ...
)
```

**Step 6: Commit**

```bash
git add src/frame/causal_frame.py src/repl/rlaph_loop.py tests/frame/test_invalidation_condition.py
git commit -m "feat: auto-generate invalidation_condition from context_slice

- generate_invalidation_condition helper function
- Produces human-readable condition string
- Called automatically in frame creation"
```

---

## Task 3: Auto Evidence Tracking

**Files:**
- Modify: `src/repl/rlaph_loop.py:121-131` (_collect_evidence method)
- Test: `tests/repl/test_evidence_tracking.py`

**Step 1: Write the failing test**

```python
"""Tests for automatic evidence tracking."""
import pytest
from pathlib import Path
from src.repl.rlaph_loop import RLAPHLoop
from src.frame.frame_index import FrameIndex
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.types import SessionContext
from datetime import datetime


def test_child_frame_added_to_parent_evidence():
    """When child frame completes, it should be added to parent's evidence."""
    index = FrameIndex()

    # Create parent frame
    parent = CausalFrame(
        frame_id="parent_evidence",
        depth=0,
        parent_id=None,
        children=[],
        query="parent query",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],  # Empty initially
        conclusion="parent conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(parent)

    # Simulate creating child frame (this happens in rlaph_loop)
    # After implementation, parent.evidence should include child frame_id

    # For now, document expected behavior:
    # When a child frame is added with parent_id pointing to parent:
    # 1. parent.children gets updated (Task 1)
    # 2. parent.evidence should also get updated

    # After implementation:
    # assert "child_frame_id" in parent.evidence


def test_evidence_tracking_in_frame_index():
    """FrameIndex should support evidence tracking on add."""
    from src.frame.frame_index import FrameIndex

    index = FrameIndex()

    parent = CausalFrame(
        frame_id="ev_parent",
        depth=0,
        parent_id=None,
        children=[],
        query="p",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="p",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(parent)

    child = CausalFrame(
        frame_id="ev_child",
        depth=1,
        parent_id="ev_parent",
        children=[],
        query="c",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="c",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(child)

    # Parent's evidence should include child
    parent_after = index.get("ev_parent")
    assert "ev_child" in parent_after.evidence
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/repl/test_evidence_tracking.py -v
```

Expected: FAIL with "AssertionError: assert 'ev_child' in []"

**Step 3: Write minimal implementation**

In `src/frame/frame_index.py`, update `add` method:

```python
def add(self, frame: "CausalFrame") -> None:
    """Add a frame to the index, update parent's children and evidence."""
    self._frames[frame.frame_id] = frame

    # Update parent's children list
    if frame.parent_id and frame.parent_id in self._frames:
        parent = self._frames[frame.parent_id]

        # Add to children
        if frame.frame_id not in parent.children:
            parent.children.append(frame.frame_id)

        # Add to evidence (child conclusion is evidence for parent)
        if frame.frame_id not in parent.evidence and frame.status != FrameStatus.INVALIDATED:
            parent.evidence.append(frame.frame_id)
```

Add import:
```python
from .causal_frame import FrameStatus
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/repl/test_evidence_tracking.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/frame/frame_index.py tests/repl/test_evidence_tracking.py
git commit -m "feat: auto-track evidence when adding child frames

- Child frame IDs added to parent's evidence list
- Enables cascade invalidation across evidence links
- Only non-invalidated frames added as evidence"
```

---

## Task 4: Add find_dependent_frames to FrameIndex

**Files:**
- Modify: `src/frame/frame_index.py`
- Test: `tests/frame/test_find_dependent.py`

**Step 1: Write the failing test**

```python
"""Tests for find_dependent_frames."""
import pytest
from src.frame.frame_index import FrameIndex
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from datetime import datetime


def test_find_dependent_frames_by_evidence():
    """Should find frames that cite a frame as evidence."""
    index = FrameIndex()

    # Create source frame
    source = CausalFrame(
        frame_id="source_frame",
        depth=0,
        parent_id=None,
        children=[],
        query="source",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="source conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(source)

    # Create dependent frame that cites source as evidence
    dependent = CausalFrame(
        frame_id="dependent_frame",
        depth=1,
        parent_id=None,  # Not a child relationship
        children=[],
        query="dependent",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=["source_frame"],  # Cites source as evidence
        conclusion="dependent conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(dependent)

    # Find frames that depend on source
    dependents = index.find_dependent_frames("source_frame")

    assert "dependent_frame" in dependents
    assert len(dependents) == 1


def test_find_dependent_frames_multiple():
    """Should find all frames that cite a frame."""
    index = FrameIndex()

    source = CausalFrame(
        frame_id="multi_source",
        depth=0,
        parent_id=None,
        children=[],
        query="s",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="s",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(source)

    for i in range(3):
        dep = CausalFrame(
            frame_id=f"dep_{i}",
            depth=1,
            parent_id=None,
            children=[],
            query=f"d{i}",
            context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
            evidence=["multi_source"],
            conclusion=f"d{i}",
            confidence=0.8,
            invalidation_condition="",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )
        index.add(dep)

    dependents = index.find_dependent_frames("multi_source")

    assert len(dependents) == 3
    assert "dep_0" in dependents
    assert "dep_1" in dependents
    assert "dep_2" in dependents


def test_find_dependent_frames_none():
    """Should return empty set if no dependents."""
    index = FrameIndex()

    orphan = CausalFrame(
        frame_id="orphan",
        depth=0,
        parent_id=None,
        children=[],
        query="o",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="o",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(orphan)

    dependents = index.find_dependent_frames("orphan")

    assert dependents == set()


def test_find_dependent_frames_includes_children():
    """Should include children as dependents (they inherit parent's evidence)."""
    index = FrameIndex()

    parent = CausalFrame(
        frame_id="parent_dep",
        depth=0,
        parent_id=None,
        children=[],
        query="p",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="p",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(parent)

    child = CausalFrame(
        frame_id="child_dep",
        depth=1,
        parent_id="parent_dep",
        children=[],
        query="c",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="c",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(child)

    # Children are dependents of parent
    dependents = index.find_dependent_frames("parent_dep")

    # Should include child (via children list)
    assert "child_dep" in dependents
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/frame/test_find_dependent.py -v
```

Expected: FAIL with "AttributeError: 'FrameIndex' object has no attribute 'find_dependent_frames'"

**Step 3: Write minimal implementation**

In `src/frame/frame_index.py`, add method:

```python
def find_dependent_frames(self, frame_id: str) -> set[str]:
    """
    Find all frames that depend on a given frame.

    Dependents include:
    - Children (frames with this frame as parent_id)
    - Evidence consumers (frames citing this frame in evidence list)

    Args:
        frame_id: Frame to find dependents for

    Returns:
        Set of frame IDs that depend on this frame
    """
    dependents = set()

    for fid, frame in self._frames.items():
        # Check if child
        if frame.parent_id == frame_id:
            dependents.add(fid)

        # Check if cites as evidence
        if frame_id in frame.evidence:
            dependents.add(fid)

    return dependents
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/frame/test_find_dependent.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/frame/frame_index.py tests/frame/test_find_dependent.py
git commit -m "feat: add find_dependent_frames for cascade invalidation

- Find frames that cite a frame as evidence
- Include children as dependents
- O(n) scan, suitable for 10-20 frames"
```

---

## Task 5: Strengthen Cascade Propagation

**Files:**
- Modify: `src/frame/frame_invalidation.py`
- Test: `tests/frame/test_cascade_invalidation.py`

**Step 1: Write the failing test**

```python
"""Tests for cascade invalidation."""
import pytest
from src.frame.frame_index import FrameIndex
from src.frame.frame_invalidation import propagate_invalidation
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from datetime import datetime


def make_frame(frame_id: str, parent_id: str = None, evidence: list = None) -> CausalFrame:
    """Helper to create test frames."""
    return CausalFrame(
        frame_id=frame_id,
        depth=0 if not parent_id else 1,
        parent_id=parent_id,
        children=[],
        query=f"query_{frame_id}",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=evidence or [],
        conclusion=f"conclusion_{frame_id}",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )


def test_propagate_invalidates_children():
    """Invalidating a frame should invalidate its children."""
    index = FrameIndex()

    parent = make_frame("parent_cascade")
    index.add(parent)

    child = make_frame("child_cascade", parent_id="parent_cascade")
    index.add(child)

    grandchild = make_frame("grandchild_cascade", parent_id="child_cascade")
    index.add(grandchild)

    # Invalidate parent
    invalidated = propagate_invalidation("parent_cascade", "test reason", index)

    assert "parent_cascade" in invalidated
    assert "child_cascade" in invalidated
    assert "grandchild_cascade" in invalidated

    assert index.get("parent_cascade").status == FrameStatus.INVALIDATED
    assert index.get("child_cascade").status == FrameStatus.INVALIDATED
    assert index.get("grandchild_cascade").status == FrameStatus.INVALIDATED


def test_propagate_invalidates_evidence_consumers():
    """Invalidating a frame should invalidate frames citing it as evidence."""
    index = FrameIndex()

    source = make_frame("source_evidence")
    index.add(source)

    consumer1 = make_frame("consumer1", evidence=["source_evidence"])
    index.add(consumer1)

    consumer2 = make_frame("consumer2", evidence=["source_evidence"])
    index.add(consumer2)

    # Invalidate source
    invalidated = propagate_invalidation("source_evidence", "evidence changed", index)

    assert "source_evidence" in invalidated
    assert "consumer1" in invalidated
    assert "consumer2" in invalidated


def test_propagate_handles_circular_evidence():
    """Should handle cycles without infinite loop."""
    index = FrameIndex()

    # Create frames that cite each other (edge case)
    frame_a = make_frame("frame_a", evidence=["frame_b"])
    frame_b = make_frame("frame_b", evidence=["frame_a"])

    index.add(frame_a)
    index.add(frame_b)

    # Should not hang
    invalidated = propagate_invalidation("frame_a", "test cycle", index)

    assert "frame_a" in invalidated
    assert "frame_b" in invalidated


def test_propagate_records_escalation_reason():
    """Invalidated frames should have escalation_reason set."""
    index = FrameIndex()

    parent = make_frame("parent_reason")
    index.add(parent)

    child = make_frame("child_reason", parent_id="parent_reason")
    index.add(child)

    propagate_invalidation("parent_reason", "original reason", index)

    parent_frame = index.get("parent_reason")
    child_frame = index.get("child_reason")

    assert parent_frame.escalation_reason == "original reason"
    assert "original reason" in child_frame.escalation_reason


def test_propagate_returns_all_invalidated():
    """Should return set of all invalidated frame IDs."""
    index = FrameIndex()

    # Create a small tree
    root = make_frame("root")
    index.add(root)

    child1 = make_frame("child1", parent_id="root")
    index.add(child1)

    child2 = make_frame("child2", parent_id="root")
    index.add(child2)

    evidence_user = make_frame("evidence_user", evidence=["root"])
    index.add(evidence_user)

    invalidated = propagate_invalidation("root", "test", index)

    assert invalidated == {"root", "child1", "child2", "evidence_user"}


def test_propagate_skips_already_invalidated():
    """Should not re-process already invalidated frames."""
    index = FrameIndex()

    parent = make_frame("parent_skip")
    index.add(parent)

    child = make_frame("child_skip", parent_id="parent_skip")
    child.status = FrameStatus.INVALIDATED  # Already invalidated
    index.add(child)

    invalidated = propagate_invalidation("parent_skip", "test", index)

    # Should still include already-invalidated child in results
    assert "child_skip" in invalidated
```

**Step 2: Run test to verify current state**

```bash
uv run pytest tests/frame/test_cascade_invalidation.py -v
```

The existing `propagate_invalidation` should handle most of these. Let's verify.

**Step 3: Enhance implementation if needed**

Current implementation in `src/frame/frame_invalidation.py` should already work. If any tests fail, enhance:

```python
def propagate_invalidation(
    frame_id: str,
    reason: str,
    index: "FrameIndex"
) -> list[str]:
    """
    Invalidate a frame and all its dependents.

    Propagation direction:
    - DOWN: to all children (tree walk)
    - SIDEWAYS: to frames using this as evidence (via find_dependent_frames)

    At 10-20 frames, O(n) scan is instant. No DAG structure needed.
    """
    from .causal_frame import FrameStatus

    invalidated = set()

    def _invalidate(fid: str, current_reason: str):
        if fid in invalidated:
            return  # Already processed (handles cycles)
        invalidated.add(fid)

        frame = index.get(fid)
        if frame is None:
            return

        frame.status = FrameStatus.INVALIDATED
        frame.escalation_reason = current_reason

        # CASCADE DOWN to children
        for child_id in frame.children:
            _invalidate(child_id, f"Parent invalidated: {reason}")

        # CASCADE SIDEWAYS to evidence consumers
        dependents = index.find_dependent_frames(fid)
        for dep_id in dependents:
            if dep_id not in invalidated:  # Skip if already being processed
                _invalidate(dep_id, f"Evidence invalidated: {fid}")

    _invalidate(frame_id, reason)
    return list(invalidated)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/frame/test_cascade_invalidation.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/frame/frame_invalidation.py tests/frame/test_cascade_invalidation.py
git commit -m "feat: strengthen cascade invalidation with find_dependent_frames

- Use find_dependent_frames for sideways propagation
- Clear escalation_reason chain
- Handle circular evidence gracefully"
```

---

## Task 6: Integration Test - Full Tree with Cascade

**Files:**
- Create: `tests/integration/test_tree_cascade_integration.py`

**Step 1: Write integration test**

```python
"""Integration test for tree structure + evidence + cascade."""
import pytest
from src.frame.frame_index import FrameIndex
from src.frame.frame_invalidation import propagate_invalidation
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from datetime import datetime


def make_frame(
    frame_id: str,
    parent_id: str = None,
    evidence: list = None,
    files: dict = None
) -> CausalFrame:
    """Helper to create test frames."""
    return CausalFrame(
        frame_id=frame_id,
        depth=0 if not parent_id else 1,
        parent_id=parent_id,
        children=[],
        query=f"query_{frame_id}",
        context_slice=ContextSlice(
            files=files or {},
            memory_refs=[],
            tool_outputs={},
            token_budget=8000,
        ),
        evidence=evidence or [],
        conclusion=f"conclusion_{frame_id}",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )


def test_full_tree_cascade_invalidation():
    """
    Integration test: Full tree with children and evidence links.

    Tree structure:
              root
             /    \
          child1  child2
            |       |
          leaf1  evidence_user (cites leaf1)

    When root is invalidated:
    - child1, child2 should be invalidated (children)
    - leaf1 should be invalidated (grandchild)
    - evidence_user should be invalidated (cites leaf1 as evidence)
    """
    index = FrameIndex()

    # Build tree
    root = make_frame("root", files={"/src/main.py": "hash1"})
    index.add(root)

    child1 = make_frame("child1", parent_id="root")
    index.add(child1)

    child2 = make_frame("child2", parent_id="root")
    index.add(child2)

    leaf1 = make_frame("leaf1", parent_id="child1")
    index.add(leaf1)

    # evidence_user cites leaf1 but is not a child
    evidence_user = make_frame("evidence_user", evidence=["leaf1"])
    index.add(evidence_user)

    # Verify tree structure was built correctly
    assert "child1" in index.get("root").children
    assert "child2" in index.get("root").children
    assert "leaf1" in index.get("child1").children

    # Verify evidence tracking
    assert "leaf1" in index.get("child1").evidence

    # Now invalidate root
    invalidated = propagate_invalidation(
        "root",
        "main.py changed",
        index
    )

    # All frames should be invalidated
    assert len(invalidated) == 5
    assert "root" in invalidated
    assert "child1" in invalidated
    assert "child2" in invalidated
    assert "leaf1" in invalidated
    assert "evidence_user" in invalidated

    # Verify status
    for fid in invalidated:
        assert index.get(fid).status == FrameStatus.INVALIDATED

    # Verify escalation reasons propagate
    root_frame = index.get("root")
    assert "main.py changed" in root_frame.escalation_reason

    evidence_user_frame = index.get("evidence_user")
    assert "leaf1" in evidence_user_frame.escalation_reason


def test_partial_tree_invalidation():
    """
    Only invalidating a subtree should not affect siblings.
    """
    index = FrameIndex()

    root = make_frame("root_partial")
    index.add(root)

    child1 = make_frame("child1_partial", parent_id="root_partial")
    index.add(child1)

    child1_leaf = make_frame("child1_leaf", parent_id="child1_partial")
    index.add(child1_leaf)

    child2 = make_frame("child2_partial", parent_id="root_partial")
    index.add(child2)

    child2_leaf = make_frame("child2_leaf", parent_id="child2_partial")
    index.add(child2_leaf)

    # Invalidate only child1
    invalidated = propagate_invalidation(
        "child1_partial",
        "child1 reason",
        index
    )

    # Only child1 subtree should be invalidated
    assert "child1_partial" in invalidated
    assert "child1_leaf" in invalidated

    # Sibling subtree should NOT be invalidated
    assert "child2_partial" not in invalidated
    assert "child2_leaf" not in invalidated
    assert "root_partial" not in invalidated

    # Verify statuses
    assert index.get("child1_partial").status == FrameStatus.INVALIDATED
    assert index.get("child1_leaf").status == FrameStatus.INVALIDATED
    assert index.get("child2_partial").status == FrameStatus.COMPLETED
    assert index.get("child2_leaf").status == FrameStatus.COMPLETED
```

**Step 2: Run integration test**

```bash
uv run pytest tests/integration/test_tree_cascade_integration.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_tree_cascade_integration.py
git commit -m "test: add integration test for tree + evidence + cascade

- Full tree cascade scenario
- Partial subtree invalidation
- Evidence propagation across branches"
```

---

## Verification

Run all tests to verify Phase 18 is complete:

```bash
uv run pytest tests/ -v --tb=short
```

All tests should pass.

---

## Summary

Phase 18 adds:

1. **Children population** - FrameIndex.add updates parent.children
2. **Auto evidence tracking** - Child frames added to parent.evidence
3. **Auto invalidation_condition** - Generated from context_slice
4. **find_dependent_frames** - Discover evidence consumers
5. **Stronger cascade** - Full tree walk + evidence propagation

Result: Proper tree structure with cascade invalidation when premises change.
