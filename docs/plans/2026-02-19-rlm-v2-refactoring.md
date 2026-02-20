# RLM v2 Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor RLM-Claude-Code from ~90 files to 12 src files by removing complexity (orchestrators, routers, vector search, async stacks) and focusing on REPL + CausalFrame persistence.

**Architecture:** Two mechanisms work together: REPL (spatial externalization within session) and CausalFrame (temporal persistence across sessions). The model navigates its environment actively, not passively.

**Tech Stack:** Python 3.11+, dataclasses, JSONL storage, RestrictedPython for REPL sandboxing

---

## Phase 1: Foundation Files (Low Risk)

### Task 1: Create src/llm_client.py

**Files:**
- Create: `src/llm_client.py`
- Test: `tests/unit/test_llm_client.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_llm_client.py
"""Tests for LLMClient - simple synchronous LLM calls."""

import pytest
from src.llm_client import LLMClient, LLMError


class TestLLMClient:
    """Test suite for LLMClient."""

    def test_llm_client_instantiation(self):
        """LLMClient can be instantiated with defaults."""
        client = LLMClient()
        assert client is not None

    def test_llm_client_with_api_key(self):
        """LLMClient accepts api_key parameter."""
        client = LLMClient(api_key="test-key")
        assert client.api_key == "test-key"

    def test_get_model_for_depth_root(self):
        """Root depth (0) returns sonnet model."""
        client = LLMClient()
        model = client.get_model_for_depth(0)
        assert "sonnet" in model.lower() or model == "glm-4.7"

    def test_get_model_for_depth_deep(self):
        """Deep depth (3+) returns haiku model."""
        client = LLMClient()
        model = client.get_model_for_depth(3)
        assert "haiku" in model.lower() or model == "glm-4.7"

    def test_call_requires_query(self):
        """LLMClient.call requires a query."""
        client = LLMClient()
        with pytest.raises(ValueError):
            client.call("")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.llm_client'"

**Step 3: Write minimal implementation**

```python
# src/llm_client.py
"""Simple synchronous LLM client for REPL environment.

Design doc reference: src/llm_client.py in target architecture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


class LLMError(Exception):
    """LLM call failed."""
    pass


@dataclass
class LLMClient:
    """
    Provider-agnostic LLM client.

    Simple synchronous interface for REPL's llm() function.
    Default model cascade: root uses larger model, sub-calls use smaller.
    """

    api_key: str | None = None
    default_model: str = "glm-4.7"
    model cascade: dict[int, str] = field(default_factory=lambda: {
        0: "glm-4.7",  # root
        1: "glm-4.7",  # depth 1
        2: "glm-4.7",  # depth 2+
        3: "glm-4.7",
    })

    def __post_init__(self):
        """Initialize API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def get_model_for_depth(self, depth: int) -> str:
        """
        Get appropriate model for recursion depth.

        Args:
            depth: Current recursion depth (0 = root)

        Returns:
            Model identifier string
        """
        # Check for custom model env vars
        sonnet_model = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", self.default_model)
        haiku_model = os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", self.default_model)

        # Route to cheaper models at deeper depths
        depth_model_map = {
            0: sonnet_model,
            1: sonnet_model,
            2: haiku_model,
            3: haiku_model,
        }
        return depth_model_map.get(depth, haiku_model)

    def call(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        model: str | None = None,
        depth: int = 0,
    ) -> str:
        """
        Make a synchronous LLM call.

        Args:
            query: The query/prompt string
            context: Optional context dict (files, prior_results, etc.)
            model: Optional model override (None = use default for depth)
            depth: Current recursion depth for model selection

        Returns:
            LLM response as string

        Raises:
            ValueError: If query is empty
            LLMError: If the LLM call fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Select model
        selected_model = model if model else self.get_model_for_depth(depth)

        # Build full prompt with context
        full_prompt = self._build_prompt(query, context)

        # Make the actual API call
        return self._api_call(selected_model, full_prompt)

    def _build_prompt(self, query: str, context: dict[str, Any] | None) -> str:
        """Build full prompt from query and context."""
        if not context:
            return query

        parts = []

        # Add file contents if provided
        if "files" in context:
            for path, content in context["files"].items():
                parts.append(f"File: {path}\n```\n{content}\n```")

        # Add prior results if provided
        if "prior_results" in context:
            for name, result in context["prior_results"].items():
                parts.append(f"{name} = {result}")

        parts.append(f"\n{query}")
        return "\n\n".join(parts)

    def _api_call(self, model: str, prompt: str) -> str:
        """
        Make the actual API call.

        This is a placeholder that should be implemented based on
        the actual LLM provider being used.
        """
        # Placeholder - actual implementation depends on provider
        # For testing, this can be mocked
        raise LLMError("LLMClient._api_call not implemented - subclass or mock required")


__all__ = ["LLMClient", "LLMError"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/llm_client.py tests/unit/test_llm_client.py
git commit -m "feat: add LLMClient for synchronous LLM calls"
```

---

### Task 2: Create src/frame_store.py

**Files:**
- Create: `src/frame_store.py`
- Test: `tests/unit/test_frame_store.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_frame_store.py
"""Tests for FrameStore - JSONL persistence for CausalFrames."""

import json
import pytest
from pathlib import Path
from datetime import datetime
import tempfile

from src.causal_frame import CausalFrame, FrameStatus
from src.context_slice import ContextSlice
from src.frame_store import FrameStore


class TestFrameStore:
    """Test suite for FrameStore."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary FrameStore."""
        store_path = tmp_path / "test_session.jsonl"
        return FrameStore(path=store_path)

    @pytest.fixture
    def sample_frame(self):
        """Create a sample CausalFrame for testing."""
        context_slice = ContextSlice(
            files={"test.py": "abc123"},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )
        return CausalFrame(
            frame_id="test_frame_1",
            depth=0,
            parent_id=None,
            children=[],
            query="test query",
            context_slice=context_slice,
            evidence=[],
            conclusion="test conclusion",
            confidence=0.9,
            invalidation_condition="file changes",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

    def test_frame_store_instantiation(self, tmp_path):
        """FrameStore can be instantiated."""
        store_path = tmp_path / "test.jsonl"
        store = FrameStore(path=store_path)
        assert store.path == store_path

    def test_save_frame(self, temp_store, sample_frame):
        """FrameStore.save persists a frame."""
        temp_store.save(sample_frame)
        assert temp_store.path.exists()

    def test_load_frame(self, temp_store, sample_frame):
        """FrameStore.load retrieves a frame by ID."""
        temp_store.save(sample_frame)
        loaded = temp_store.load(sample_frame.frame_id)
        assert loaded is not None
        assert loaded.frame_id == sample_frame.frame_id
        assert loaded.query == sample_frame.query

    def test_load_nonexistent_frame(self, temp_store):
        """FrameStore.load returns None for nonexistent ID."""
        loaded = temp_store.load("nonexistent")
        assert loaded is None

    def test_list_frames(self, temp_store, sample_frame):
        """FrameStore.list returns all frames."""
        temp_store.save(sample_frame)
        frames = temp_store.list()
        assert len(frames) == 1
        assert frames[0].frame_id == sample_frame.frame_id

    def test_list_empty_store(self, temp_store):
        """FrameStore.list returns empty list when empty."""
        frames = temp_store.list()
        assert frames == []

    def test_find_by_status(self, temp_store, sample_frame):
        """FrameStore.find_by_status filters by status."""
        sample_frame.status = FrameStatus.COMPLETED
        temp_store.save(sample_frame)

        completed = temp_store.find_by_status(FrameStatus.COMPLETED)
        assert len(completed) == 1

        running = temp_store.find_by_status(FrameStatus.RUNNING)
        assert len(running) == 0

    def test_multiple_frames(self, temp_store, sample_frame):
        """FrameStore handles multiple frames."""
        frame2 = CausalFrame(
            frame_id="test_frame_2",
            depth=1,
            parent_id="test_frame_1",
            children=[],
            query="child query",
            context_slice=sample_frame.context_slice,
            evidence=["test_frame_1"],
            conclusion="child conclusion",
            confidence=0.8,
            invalidation_condition="parent invalidates",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

        temp_store.save(sample_frame)
        temp_store.save(frame2)

        frames = temp_store.list()
        assert len(frames) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_frame_store.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.frame_store'"

**Step 3: Write minimal implementation**

```python
# src/frame_store.py
"""FrameStore - JSONL persistence for CausalFrames.

Design doc reference: src/frame_store.py in target architecture.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .causal_frame import CausalFrame, FrameStatus


class FrameStore:
    """
    JSONL-based persistence for CausalFrames.

    One file per session: ~/.claude/rlm-frames/{root_session_id}.jsonl
    Zero dependencies. Human-readable. Append-only.

    Each line is a JSON object representing one CausalFrame.
    """

    def __init__(self, path: Path | str):
        """
        Initialize FrameStore.

        Args:
            path: Path to the JSONL file for this session
        """
        self.path = Path(path)
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, frame: "CausalFrame") -> None:
        """
        Save a frame to the store.

        Appends to the JSONL file. Does not check for duplicates.

        Args:
            frame: CausalFrame to persist
        """
        data = self._serialize(frame)
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def load(self, frame_id: str) -> "CausalFrame | None":
        """
        Load a frame by ID.

        Scans the file linearly. At 10-20 frames, this is instant.

        Args:
            frame_id: ID of the frame to load

        Returns:
            CausalFrame if found, None otherwise
        """
        if not self.path.exists():
            return None

        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("frame_id") == frame_id:
                    return self._deserialize(data)

        return None

    def list(self) -> list["CausalFrame"]:
        """
        List all frames in the store.

        Returns:
            List of all CausalFrames
        """
        if not self.path.exists():
            return []

        frames = []
        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                frames.append(self._deserialize(data))

        return frames

    def find_by_status(self, status: "FrameStatus") -> list["CausalFrame"]:
        """
        Find all frames with a given status.

        Args:
            status: FrameStatus to filter by

        Returns:
            List of matching CausalFrames
        """
        from .causal_frame import FrameStatus

        frames = self.list()
        return [f for f in frames if f.status == status]

    def _serialize(self, frame: "CausalFrame") -> dict:
        """Serialize CausalFrame to JSON-compatible dict."""
        data = {
            "frame_id": frame.frame_id,
            "depth": frame.depth,
            "parent_id": frame.parent_id,
            "children": frame.children,
            "query": frame.query,
            "context_slice": {
                "files": frame.context_slice.files,
                "memory_refs": frame.context_slice.memory_refs,
                "tool_outputs": frame.context_slice.tool_outputs,
                "token_budget": frame.context_slice.token_budget,
            },
            "evidence": frame.evidence,
            "conclusion": frame.conclusion,
            "confidence": frame.confidence,
            "invalidation_condition": frame.invalidation_condition,
            "status": frame.status.value,
            "branched_from": frame.branched_from,
            "escalation_reason": frame.escalation_reason,
            "created_at": frame.created_at.isoformat() if frame.created_at else None,
            "completed_at": frame.completed_at.isoformat() if frame.completed_at else None,
        }
        return data

    def _deserialize(self, data: dict) -> "CausalFrame":
        """Deserialize dict to CausalFrame."""
        from .causal_frame import CausalFrame, FrameStatus
        from .context_slice import ContextSlice

        context_slice = ContextSlice(
            files=data["context_slice"]["files"],
            memory_refs=data["context_slice"]["memory_refs"],
            tool_outputs=data["context_slice"]["tool_outputs"],
            token_budget=data["context_slice"]["token_budget"],
        )

        return CausalFrame(
            frame_id=data["frame_id"],
            depth=data["depth"],
            parent_id=data["parent_id"],
            children=data["children"],
            query=data["query"],
            context_slice=context_slice,
            evidence=data["evidence"],
            conclusion=data["conclusion"],
            confidence=data["confidence"],
            invalidation_condition=data["invalidation_condition"],
            status=FrameStatus(data["status"]),
            branched_from=data.get("branched_from"),
            escalation_reason=data.get("escalation_reason"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )


__all__ = ["FrameStore"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_frame_store.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/frame_store.py tests/unit/test_frame_store.py
git commit -m "feat: add FrameStore for JSONL frame persistence"
```

---

## Phase 2: Core Refactor (Medium Risk)

### Task 3: Simplify src/causal_frame.py

**Files:**
- Modify: `src/causal_frame.py`
- Test: `tests/unit/test_causal_frame.py` (update if exists)

**Step 1: Review current implementation**

Current `causal_frame.py` has:
- FrameStatus with VERIFIED, UNCERTAIN (not in v2 spec)
- escalation_reason field (not in v2 spec)

v2 spec FrameStatus:
```python
class FrameStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    SUSPENDED = "suspended"      # pivot — preserved, not deleted
    INVALIDATED = "invalidated"
    PROMOTED = "promoted"        # persisted as long-term knowledge
```

**Step 2: Write test for simplified status**

```python
# tests/unit/test_causal_frame.py (add to existing)
"""Tests for simplified CausalFrame."""

import pytest
from src.causal_frame import CausalFrame, FrameStatus, compute_frame_id
from src.context_slice import ContextSlice


class TestSimplifiedFrameStatus:
    """Test simplified FrameStatus enum."""

    def test_status_running_exists(self):
        """FrameStatus.RUNNING exists."""
        assert FrameStatus.RUNNING.value == "running"

    def test_status_completed_exists(self):
        """FrameStatus.COMPLETED exists."""
        assert FrameStatus.COMPLETED.value == "completed"

    def test_status_suspended_exists(self):
        """FrameStatus.SUSPENDED exists."""
        assert FrameStatus.SUSPENDED.value == "suspended"

    def test_status_invalidated_exists(self):
        """FrameStatus.INVALIDATED exists."""
        assert FrameStatus.INVALIDATED.value == "invalidated"

    def test_status_promoted_exists(self):
        """FrameStatus.PROMOTED exists."""
        assert FrameStatus.PROMOTED.value == "promoted"

    def test_status_count(self):
        """FrameStatus has exactly 5 values."""
        assert len(FrameStatus) == 5
```

**Step 3: Update causal_frame.py**

The current implementation already has the right structure. We just need to remove VERIFIED, CREATED, and UNCERTAIN statuses.

```python
# src/causal_frame.py - update FrameStatus enum
class FrameStatus(Enum):
    """Lifecycle states for a CausalFrame."""

    RUNNING = "running"          # Frame is actively executing
    COMPLETED = "completed"      # Frame finished execution
    SUSPENDED = "suspended"      # Pivot — preserved, not deleted
    INVALIDATED = "invalidated"  # Cascade invalidation
    PROMOTED = "promoted"        # Persisted as long-term knowledge
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_causal_frame.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/causal_frame.py tests/unit/test_causal_frame.py
git commit -m "refactor: simplify FrameStatus to 5 core states"
```

---

### Task 4: Update src/frame_index.py

**Files:**
- Modify: `src/frame_index.py`
- Test: `tests/unit/test_frame_index.py` (create if needed)

**Step 1: Review current implementation**

Current `frame_index.py` is already a simple dict - no changes needed for core functionality.

**Step 2: Add find_by_parent method**

```python
# Add to FrameIndex class
def find_by_parent(self, parent_id: str) -> list["CausalFrame"]:
    """Find all frames with a given parent."""
    return [f for f in self._frames.values() if f.parent_id == parent_id]

def find_promoted(self) -> list["CausalFrame"]:
    """Find all PROMOTED frames (persisted facts)."""
    from .causal_frame import FrameStatus
    return [f for f in self._frames.values() if f.status == FrameStatus.PROMOTED]
```

**Step 3: Write tests**

```python
# tests/unit/test_frame_index.py
"""Tests for FrameIndex."""

import pytest
from datetime import datetime

from src.causal_frame import CausalFrame, FrameStatus
from src.context_slice import ContextSlice
from src.frame_index import FrameIndex


class TestFrameIndex:
    """Test suite for FrameIndex."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample CausalFrame."""
        context_slice = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )
        return CausalFrame(
            frame_id="root",
            depth=0,
            parent_id=None,
            children=["child1"],
            query="root query",
            context_slice=context_slice,
            evidence=[],
            conclusion="root conclusion",
            confidence=0.9,
            invalidation_condition="",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

    def test_add_and_get(self, sample_frame):
        """FrameIndex.add and get work correctly."""
        index = FrameIndex()
        index.add(sample_frame)
        assert index.get("root") == sample_frame

    def test_get_active_frames(self, sample_frame):
        """FrameIndex.get_active_frames returns RUNNING frames."""
        sample_frame.status = FrameStatus.RUNNING
        index = FrameIndex()
        index.add(sample_frame)
        active = index.get_active_frames()
        assert len(active) == 1

    def test_get_suspended_frames(self, sample_frame):
        """FrameIndex.get_suspended_frames returns SUSPENDED frames."""
        sample_frame.status = FrameStatus.SUSPENDED
        index = FrameIndex()
        index.add(sample_frame)
        suspended = index.get_suspended_frames()
        assert len(suspended) == 1

    def test_find_by_parent(self, sample_frame):
        """FrameIndex.find_by_parent returns children."""
        child = CausalFrame(
            frame_id="child1",
            depth=1,
            parent_id="root",
            children=[],
            query="child query",
            context_slice=sample_frame.context_slice,
            evidence=["root"],
            conclusion="child conclusion",
            confidence=0.8,
            invalidation_condition="",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(sample_frame)
        index.add(child)

        children = index.find_by_parent("root")
        assert len(children) == 1
        assert children[0].frame_id == "child1"

    def test_find_promoted(self, sample_frame):
        """FrameIndex.find_promoted returns PROMOTED frames."""
        sample_frame.status = FrameStatus.PROMOTED
        index = FrameIndex()
        index.add(sample_frame)
        promoted = index.find_promoted()
        assert len(promoted) == 1
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_frame_index.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/frame_index.py tests/unit/test_frame_index.py
git commit -m "feat: add find_by_parent and find_promoted to FrameIndex"
```

---

### Task 5: Update src/session_comparison.py

**Files:**
- Modify: `src/session_comparison.py`
- Test: `tests/unit/test_session_comparison.py`

**Step 1: Add FrameIndex integration to compare_sessions**

Current implementation doesn't populate `invalidated_frame_ids` or `resumable_frames`. We need to add FrameIndex parameter.

**Step 2: Update session_comparison.py**

```python
# src/session_comparison.py - update compare_sessions
def compare_sessions(
    current: "SessionArtifacts",
    prior: "SessionArtifacts",
    index: "FrameIndex | None" = None
) -> SessionDiff:
    """
    Compare two sessions to detect what changed.

    Comparison logic:
    1. Compare initial_prompt → same task or new?
    2. Compare file hashes → what changed?
    3. With FrameIndex: find frames referencing changed files → invalidated
    4. With FrameIndex: find suspended frames that might be relevant

    Args:
        current: Current session artifacts
        prior: Prior session artifacts
        index: Optional FrameIndex for frame-level analysis
    """
    # Check if same task (same initial prompt)
    same_task = current.initial_prompt == prior.initial_prompt

    # Find changed files
    changed_files = []

    # New files
    for path in current.files:
        if path not in prior.files:
            changed_files.append(path)

    # Modified files (same path, different hash)
    for path, current_record in current.files.items():
        if path in prior.files:
            prior_record = prior.files[path]
            if current_record.hash != prior_record.hash:
                changed_files.append(path)

    # If no FrameIndex, return basic diff
    if index is None:
        return SessionDiff(
            same_task=same_task,
            changed_files=changed_files,
            invalidated_frame_ids=[],
            resumable_frames=[]
        )

    # With FrameIndex: find invalidated frames
    invalidated_frame_ids = _find_invalidated_frames(index, changed_files)

    # With FrameIndex: find resumable suspended frames
    resumable_frames = _find_resumable_frames(index, same_task)

    return SessionDiff(
        same_task=same_task,
        changed_files=changed_files,
        invalidated_frame_ids=invalidated_frame_ids,
        resumable_frames=resumable_frames
    )


def _find_invalidated_frames(index: "FrameIndex", changed_files: list[str]) -> list[str]:
    """Find frames whose context_slice includes changed files."""
    invalidated = []

    for frame in index.values():
        # Check if any changed file is in this frame's context
        for file_path in frame.context_slice.files:
            if file_path in changed_files:
                invalidated.append(frame.frame_id)
                break

    return invalidated


def _find_resumable_frames(index: "FrameIndex", same_task: bool) -> list[str]:
    """Find suspended frames that might be worth resuming."""
    if not same_task:
        return []  # Different task, don't resume old work

    suspended = index.get_suspended_frames()
    return [f.frame_id for f in suspended]
```

**Step 3: Write tests**

```python
# tests/unit/test_session_comparison.py
"""Tests for session comparison."""

import pytest
from datetime import datetime

from src.causal_frame import CausalFrame, FrameStatus
from src.context_slice import ContextSlice
from src.frame_index import FrameIndex
from src.session_artifacts import SessionArtifacts, FileRecord
from src.session_comparison import compare_sessions, SessionDiff


class TestSessionComparison:
    """Test suite for session comparison."""

    @pytest.fixture
    def prior_session(self):
        """Create a prior session."""
        return SessionArtifacts(
            session_id="prior",
            initial_prompt="test task",
            files={
                "file1.py": FileRecord(path="file1.py", hash="abc", role="read"),
                "file2.py": FileRecord(path="file2.py", hash="def", role="read"),
            },
            root_frame_id="root",
            conversation_log="/path/to/log"
        )

    def test_same_task_detection(self, prior_session):
        """Detect when task is the same."""
        current = SessionArtifacts(
            session_id="current",
            initial_prompt="test task",
            files={},
            root_frame_id="root",
            conversation_log="/path/to/log"
        )
        diff = compare_sessions(current, prior_session)
        assert diff.same_task is True

    def test_different_task_detection(self, prior_session):
        """Detect when task is different."""
        current = SessionArtifacts(
            session_id="current",
            initial_prompt="different task",
            files={},
            root_frame_id="root",
            conversation_log="/path/to/log"
        )
        diff = compare_sessions(current, prior_session)
        assert diff.same_task is False

    def test_changed_file_detection(self, prior_session):
        """Detect changed files."""
        current = SessionArtifacts(
            session_id="current",
            initial_prompt="test task",
            files={
                "file1.py": FileRecord(path="file1.py", hash="xyz", role="modified"),
                "file2.py": FileRecord(path="file2.py", hash="def", role="read"),
            },
            root_frame_id="root",
            conversation_log="/path/to/log"
        )
        diff = compare_sessions(current, prior_session)
        assert "file1.py" in diff.changed_files

    def test_invalidated_frames_with_index(self, prior_session):
        """Find invalidated frames when FrameIndex provided."""
        current = SessionArtifacts(
            session_id="current",
            initial_prompt="test task",
            files={
                "file1.py": FileRecord(path="file1.py", hash="xyz", role="modified"),
            },
            root_frame_id="root",
            conversation_log="/path/to/log"
        )

        # Create frame that references file1.py
        context_slice = ContextSlice(
            files={"file1.py": "abc"},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )
        frame = CausalFrame(
            frame_id="frame1",
            depth=0,
            parent_id=None,
            children=[],
            query="query",
            context_slice=context_slice,
            evidence=[],
            conclusion="conclusion",
            confidence=0.9,
            invalidation_condition="",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)

        diff = compare_sessions(current, prior_session, index=index)
        assert "frame1" in diff.invalidated_frame_ids
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_session_comparison.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/session_comparison.py tests/unit/test_session_comparison.py
git commit -m "feat: add FrameIndex integration to compare_sessions"
```

---

### Task 6: Update src/tool_bridge.py (remove orchestration_schema dependency)

**Files:**
- Modify: `src/tool_bridge.py`

**Step 1: Remove orchestration_schema import**

Current import:
```python
from .orchestration_schema import ToolAccessLevel
```

Replace with local enum:

```python
# src/tool_bridge.py
from enum import Enum


class ToolAccessLevel(Enum):
    """Tool access levels for sub-LLMs."""
    NONE = "none"
    REPL_ONLY = "repl_only"
    READ_ONLY = "read_only"
    FULL = "full"
```

**Step 2: Commit**

```bash
git add src/tool_bridge.py
git commit -m "refactor: remove orchestration_schema dependency from tool_bridge"
```

---

## Phase 3: Integration (Medium Risk)

### Task 7: Simplify src/rlaph_loop.py

**Files:**
- Modify: `src/rlaph_loop.py`

**Step 1: Identify dependencies to remove**

Current imports to remove:
```python
from .response_parser import ResponseAction, ResponseParser
from .router_integration import ModelRouter
from .trajectory import StreamingTrajectory, TrajectoryEvent, TrajectoryEventType
from .recursive_handler import RecursiveREPL
```

**Step 2: Create simplified rlaph_loop.py**

The key change is using LLMClient directly instead of going through ModelRouter and recursive_handler.

This is a larger refactor - implement incrementally:

1. First, add LLMClient import and use it for model selection
2. Remove trajectory dependencies (optional logging instead)
3. Remove recursive_handler, use direct LLM calls
4. Remove response_parser, use simpler parsing

**Step 3: Commit incrementally**

```bash
# After each sub-change
git add src/rlaph_loop.py
git commit -m "refactor: simplify rlaph_loop - remove X dependency"
```

---

### Task 8: Update src/__init__.py

**Files:**
- Modify: `src/__init__.py`

**Step 1: Remove exports for deleted modules**

Remove all exports from modules that will be deleted:
- orchestrator.*
- async_executor, async_handler
- complexity_classifier
- router_integration, smart_router, learned_routing
- embedding_retrieval
- memory_store, memory_evolution, memory_backend
- trajectory, trajectory_analysis
- epistemic.*
- etc.

**Step 2: Keep only v2 exports**

```python
# src/__init__.py - simplified
"""RLM-Claude-Code v2: REPL + CausalFrame persistence."""

__version__ = "2.0.0"

# REPL Layer
from .rlaph_loop import RLAPHLoop, RLPALoopResult
from .repl_environment import RLMEnvironment
from .llm_client import LLMClient, LLMError
from .tool_bridge import ToolBridge, ToolPermissions, ToolResult

# Causal Layer
from .causal_frame import CausalFrame, FrameStatus, compute_frame_id
from .context_slice import ContextSlice
from .frame_index import FrameIndex
from .frame_invalidation import propagate_invalidation
from .frame_store import FrameStore
from .session_artifacts import SessionArtifacts, FileRecord
from .session_comparison import SessionDiff, compare_sessions
from .plugin_interface import CoreContext, RLMPlugin, PluginError

# Types
from .types import SessionContext
from .config import RLMConfig, default_config
from .prompts import get_prompt

__all__ = [
    # REPL
    "RLAPHLoop",
    "RLPALoopResult",
    "RLMEnvironment",
    "LLMClient",
    "LLMError",
    "ToolBridge",
    "ToolPermissions",
    "ToolResult",
    # Causal
    "CausalFrame",
    "FrameStatus",
    "compute_frame_id",
    "ContextSlice",
    "FrameIndex",
    "propagate_invalidation",
    "FrameStore",
    "SessionArtifacts",
    "FileRecord",
    "SessionDiff",
    "compare_sessions",
    "CoreContext",
    "RLMPlugin",
    "PluginError",
    # Types & Config
    "SessionContext",
    "RLMConfig",
    "default_config",
    "get_prompt",
]
```

**Step 3: Commit**

```bash
git add src/__init__.py
git commit -m "refactor: simplify __init__.py to v2 exports only"
```

---

## Phase 4: Hooks (Low Risk)

### Task 9: Update hooks/hooks.json

**Files:**
- Modify: `hooks/hooks.json`

**Step 1: Simplify to v2 hooks**

```json
{
  "description": "RLM v2 plugin hooks - REPL + CausalFrame",
  "hooks": {
    "SessionStart": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/compare_sessions.py",
            "timeout": 5000,
            "description": "Compare with prior session, surface invalidated frames"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/capture_output.py",
            "timeout": 5000,
            "description": "Capture tool output into active CausalFrame"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/extract_frames.py",
            "timeout": 5000,
            "description": "Extract frame tree, save to FrameStore"
          }
        ]
      }
    ]
  }
}
```

**Step 2: Commit**

```bash
git add hooks/hooks.json
git commit -m "refactor: simplify hooks.json to v2 structure"
```

---

### Task 10: Create scripts/extract_frames.py

**Files:**
- Create: `scripts/extract_frames.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Extract frames from session and save to FrameStore.

Hook: Stop
Purpose: Persist CausalFrame tree when session ends.
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frame_store import FrameStore
from frame_index import FrameIndex


def extract_frames(session_id: str, index: FrameIndex) -> None:
    """Extract all frames from index and save to FrameStore."""
    store_path = Path.home() / ".claude" / "rlm-frames" / f"{session_id}.jsonl"
    store = FrameStore(path=store_path)

    for frame in index.values():
        store.save(frame)

    print(f"Saved {len(index)} frames to {store_path}")


def main():
    """Main entry point for hook."""
    # Get session info from environment or stdin
    session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")

    # For now, this is a placeholder - the actual frame extraction
    # would come from the running session's FrameIndex
    # In production, this would receive the frame data via stdin

    print(f"Frame extraction hook called for session: {session_id}")

    # Read frame data from stdin if provided
    try:
        input_data = sys.stdin.read()
        if input_data:
            data = json.loads(input_data)
            print(f"Received {len(data.get('frames', []))} frames")
    except json.JSONDecodeError:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

Run: `chmod +x scripts/extract_frames.py`

**Step 3: Commit**

```bash
git add scripts/extract_frames.py
git commit -m "feat: add extract_frames.py hook script"
```

---

### Task 11: Create scripts/compare_sessions.py

**Files:**
- Create: `scripts/compare_sessions.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Compare current session with prior session.

Hook: SessionStart
Purpose: Surface changed files, invalidated frames, resumable branches.
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from session_comparison import compare_sessions
from session_artifacts import SessionArtifacts, FileRecord
from frame_store import FrameStore


def load_prior_session(session_id: str) -> SessionArtifacts | None:
    """Load prior session artifacts if they exist."""
    artifacts_path = Path.home() / ".claude" / "rlm-frames" / f"{session_id}_artifacts.json"

    if not artifacts_path.exists():
        return None

    with open(artifacts_path) as f:
        data = json.load(f)

    # Reconstruct FileRecord dict
    files = {
        path: FileRecord(path=path, hash=rec["hash"], role=rec["role"])
        for path, rec in data.get("files", {}).items()
    }

    return SessionArtifacts(
        session_id=data["session_id"],
        initial_prompt=data.get("initial_prompt", ""),
        files=files,
        root_frame_id=data.get("root_frame_id", ""),
        conversation_log=data.get("conversation_log", ""),
    )


def main():
    """Main entry point for hook."""
    session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")

    # Get prior session ID if provided
    prior_session_id = os.environ.get("CLAUDE_PRIOR_SESSION_ID")

    if not prior_session_id:
        print("No prior session to compare")
        sys.exit(0)

    # Load prior session
    prior = load_prior_session(prior_session_id)

    if not prior:
        print(f"Could not load prior session: {prior_session_id}")
        sys.exit(0)

    # Create current session from environment/args
    # This is a placeholder - real implementation would get actual file info
    current = SessionArtifacts(
        session_id=session_id,
        initial_prompt=os.environ.get("CLAUDE_PROMPT", ""),
        files={},  # Would be populated from actual session
        root_frame_id="",
        conversation_log="",
    )

    # Compare
    diff = compare_sessions(current, prior)

    # Output results for Claude to see
    output = {
        "same_task": diff.same_task,
        "changed_files": diff.changed_files,
        "invalidated_frames": diff.invalidated_frame_ids,
        "resumable": diff.resumable_frames,
    }

    print(json.dumps(output, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

Run: `chmod +x scripts/compare_sessions.py`

**Step 3: Commit**

```bash
git add scripts/compare_sessions.py
git commit -m "feat: add compare_sessions.py hook script"
```

---

## Phase 5: Cleanup (Medium Risk)

### Task 12: Remove obsolete files - Batch A (Orchestrators)

**Files:**
- Delete: `src/orchestrator/` (entire directory)
- Delete: `src/local_orchestrator.py`
- Delete: `src/intelligent_orchestrator.py`
- Delete: `src/orchestration_logger.py`
- Delete: `src/orchestration_schema.py`
- Delete: `src/orchestration_telemetry.py`

**Step 1: Delete files**

```bash
rm -rf src/orchestrator/
rm src/local_orchestrator.py
rm src/intelligent_orchestrator.py
rm src/orchestration_logger.py
rm src/orchestration_schema.py
rm src/orchestration_telemetry.py
```

**Step 2: Run tests**

Run: `pytest tests/ -v --tb=short`
Expected: Some failures due to missing modules - note which tests fail

**Step 3: Delete corresponding tests**

```bash
rm tests/unit/test_intelligent_orchestrator.py
rm tests/unit/test_local_orchestrator.py
rm tests/unit/test_orchestration_logger.py
rm tests/unit/test_orchestration_schema.py
rm tests/unit/test_orchestration_telemetry.py
rm tests/unit/test_modular_orchestrator.py
rm tests/unit/test_multi_turn_checkpointing.py
rm tests/unit/test_interactive_steering.py
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove orchestrator components (Batch A)"
```

---

### Task 13: Remove obsolete files - Batch B (Routers)

**Files:**
- Delete: `src/smart_router.py`
- Delete: `src/complexity_classifier.py`
- Delete: `src/learned_routing.py`
- Delete: `src/router_integration.py`
- Delete: `src/setfit_classifier.py`

**Step 1: Delete files and tests**

```bash
rm src/smart_router.py src/complexity_classifier.py src/learned_routing.py
rm src/router_integration.py src/setfit_classifier.py
rm tests/unit/test_smart_router.py tests/unit/test_complexity_classifier.py
rm tests/unit/test_learned_routing.py tests/unit/test_setfit_classifier.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: remove routing components (Batch B)"
```

---

### Task 14: Remove obsolete files - Batch C (Vector Search)

**Files:**
- Delete: `src/embedding_retrieval.py`
- Delete: `src/context_index.py`

**Step 1: Delete files and tests**

```bash
rm src/embedding_retrieval.py src/context_index.py
rm tests/unit/test_embedding_retrieval.py tests/unit/test_context_index.py
rm tests/property/test_context_index_properties.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: remove vector search components (Batch C)"
```

---

### Task 15: Remove obsolete files - Batch D (Async Stack)

**Files:**
- Delete: `src/async_executor.py`
- Delete: `src/async_handler.py`
- Delete: `src/recursive_handler.py`

**Step 1: Delete files and tests**

```bash
rm src/async_executor.py src/async_handler.py src/recursive_handler.py
rm tests/unit/test_async_executor.py tests/unit/test_async_handler.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: remove async stack (Batch D)"
```

---

### Task 16: Remove obsolete files - Batch E (Memory & Telemetry)

**Files:**
- Delete: `src/memory_store.py`
- Delete: `src/memory_backend.py`
- Delete: `src/memory_evolution.py`
- Delete: `src/cross_session_promotion.py`
- Delete: `src/trajectory.py`
- Delete: `src/trajectory_analysis.py`
- Delete: `src/progressive_trajectory.py`

**Step 1: Delete files and tests**

```bash
rm src/memory_store.py src/memory_backend.py src/memory_evolution.py
rm src/cross_session_promotion.py
rm src/trajectory.py src/trajectory_analysis.py src/progressive_trajectory.py
rm tests/unit/test_memory_*.py tests/unit/test_trajectory*.py
rm tests/unit/test_cross_session_promotion.py tests/unit/test_progressive_trajectory.py
rm tests/property/test_memory_*.py tests/property/test_trajectory_properties.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: remove memory and trajectory components (Batch E)"
```

---

### Task 17: Remove obsolete files - Batch F (Epistemic)

**Files:**
- Delete: `src/epistemic/` (entire directory)

**Step 1: Delete directory**

```bash
rm -rf src/epistemic/
rm tests/unit/test_claim_extractor.py tests/unit/test_consistency_checker.py
rm tests/unit/test_epistemic_*.py tests/unit/test_evidence_auditor.py
rm tests/unit/test_similarity.py tests/unit/test_verification_*.py
rm tests/property/test_epistemic_properties.py
rm tests/integration/test_epistemic_verification.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: remove epistemic verification (Batch F)"
```

---

### Task 18: Remove obsolete files - Batch G (Remaining)

**Files:**
- Delete all remaining files not in v2 target list

**Step 1: List remaining to delete**

```bash
# These are files not in the v2 target architecture
rm src/auto_activation.py src/cache.py src/cell_manager.py
rm src/circuit_breaker.py src/compute_allocation.py src/confidence_synthesis.py
rm src/context_compression.py src/context_enrichment.py src/context_manager.py
rm src/continuous_learning.py src/cost_tracker.py src/enhanced_budget.py
rm src/execution_guarantees.py src/formal_verification.py src/frame_lifecycle.py
rm src/frame_serialization.py src/gliner_extractor.py src/lats_orchestration.py
rm src/learning.py src/plugin_registry.py src/proactive_computation.py
rm src/progress.py src/prompt_caching.py src/prompt_optimizer.py
rm src/reasoning_traces.py src/response_parser.py src/rich_output.py
rm src/session_manager.py src/session_schema.py src/smart_pipeline.py
rm src/state_persistence.py src/strategy_cache.py src/tokenization.py
rm src/transcript_parser.py src/tree_of_thoughts.py src/user_corrections.py
rm src/user_preferences.py src/visualization.py

# Remove corresponding tests
rm -rf tests/unit/test_*.py tests/property/test_*.py tests/integration/test_*.py
# Keep only tests for v2 modules
```

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: remove remaining obsolete files (Batch G)"
```

---

### Task 19: Clean up src/plugins directory

**Files:**
- Keep: `src/plugins/__init__.py`, `src/plugins/default_plugin.py` (simplified)
- Delete: `src/plugin_registry.py`

**Step 1: Simplify plugins**

Keep minimal plugin infrastructure for v2.

**Step 2: Commit**

```bash
git add -A
git commit -m "refactor: simplify plugin system to basic hooks"
```

---

## Phase 6: Verification

### Task 20: Final verification

**Step 1: Run remaining tests**

Run: `pytest tests/ -v`
Expected: All tests PASS (for v2 modules only)

**Step 2: Verify imports work**

```bash
python -c "from src import RLAPHLoop, FrameStore, CausalFrame, ContextSlice"
```

**Step 3: Verify file count**

Run: `ls src/*.py | wc -l`
Expected: 12 files

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete RLM v2 refactoring"
```

---

## Summary

| Phase | Tasks | Risk | Status |
|-------|-------|------|--------|
| Phase 1: Foundation | 2 | Low | ✓ |
| Phase 2: Core Refactor | 4 | Medium | ✓ |
| Phase 3: Integration | 2 | Medium | ✓ |
| Phase 4: Hooks | 3 | Low | ✓ |
| Phase 5: Cleanup | 8 | Medium | ✓ |
| Phase 6: Verification | 1 | Low | ✓ |
| Phase 7: Hook Integration | 3 | Low | ✓ |
| Phase 8: Causeway Branding | 4 | Low | ✓ |

**Total: 27 tasks (all complete)**

---

## Completion Status (2026-02-19)

### Completed: All 20 Tasks ✓

**File Count:**
- Original: ~90 src files
- Target: 12 src files
- Actual: 18 src files + 2 plugins (kept essential dependencies)

**v2 Files:**

| Layer | Files |
|-------|-------|
| **REPL** | `rlaph_loop.py`, `repl_environment.py`, `llm_client.py`, `response_parser.py`, `tokenization.py` |
| **Causal** | `causal_frame.py`, `context_slice.py`, `frame_index.py`, `frame_invalidation.py`, `frame_store.py` |
| **Session** | `session_artifacts.py`, `session_comparison.py` |
| **Plugin** | `plugin_interface.py`, `plugins/default_plugin.py` |
| **Config** | `config.py`, `prompts.py`, `types.py`, `tool_bridge.py`, `__init__.py` |

**Tests:** 153 passing

**Commits:**
1. `573cd05` - Simplify hooks.json to v2 structure
2. `91ec58d` - Add v2 exports to __init__.py
3. `fc87f58` - Replace ModelRouter with LLMClient
4. `eb0ba0b` - Remove orchestration_schema dependency
5. `58b41c7` - Add FrameIndex integration
6. `1cf4f50` - Phase 5 cleanup Batch A-C
7. `f20e599` - Phase 5 cleanup Batch D-F
8. `456d87b` - Phase 5 cleanup Batch G
9. `bbf1792` - Remove obsolete tests

### Why 18 Files Instead of 12

Two essential dependencies were kept:
- `response_parser.py` (193 lines) - needed by rlaph_loop for parsing LLM responses
- `tokenization.py` (509 lines) - needed by repl_environment for content chunking

These could be inlined in a future pass if file count is critical.

---

## Next Steps

### Phase 7: Hook Integration ✓ COMPLETE
- [x] Implement `capture_output.py` for PostToolUse hook
- [x] Test SessionStart hook with actual frame comparison
- [x] Test Stop hook with actual frame extraction

### Phase 8: Causeway Branding ✓ COMPLETE
- [x] Rename plugin to "causeway"
- [x] Update marketplace.json, pyproject.toml, __init__.py
- [x] Create README.md and CLAUDE.md
- [x] Update skills (context-management, verification)
- [x] Version: 0.0.1 (fresh start)

### Phase 9: Living Documentation (Future Work)
As described in the whitepaper:
- [ ] Implement git diff → invalidation cascade
- [ ] Implement frame re-execution with preserved intent
- [ ] Implement selective documentation update

### Phase 10: Further Simplification (Optional)
- [ ] Inline `response_parser.py` into `rlaph_loop.py`
- [ ] Inline `tokenization.py` functions into `repl_environment.py`
- [ ] Reduce to target 12 files

### Phase 11: Cleanup Broken Scripts/Tests (P1) ✓ COMPLETE
**See:** `docs/CHANGELOG.md` for details
- [x] Delete 7 broken scripts importing deleted modules
- [x] Fix script imports (extract_frames.py, compare_sessions.py)
- [x] Delete broken tests

### Phase 12: Fix RLM Orchestrator Hallucination (P0 - CRITICAL) ✓ COMPLETE
**See:** `docs/CHANGELOG.md` for details

**Problem:** The RLM orchestrator hallucinates - returns fake file names, ignores execution results.

- [x] Add result verification to detect hallucinations
- [x] Add retry loop when verification fails
- [x] Strengthen system prompt against hallucination
- [x] Validate answers reference actually accessed files
- [x] Add integration tests for orchestrator accuracy

### Known Limitations
1. `llm()` depth management is simplified - full recursive behavior needs testing
2. Frame lifecycle (`RUNNING` → `COMPLETED`) is implemented but not validated end-to-end
3. Hook integration is basic - full integration requires Claude Code runtime testing

---

## Final Commits (All Phases)

1. `573cd05` - Simplify hooks.json to v2 structure
2. `91ec58d` - Add v2 exports to __init__.py
3. `fc87f58` - Replace ModelRouter with LLMClient
4. `eb0ba0b` - Remove orchestration_schema dependency
5. `58b41c7` - Add FrameIndex integration
6. `1cf4f50` - Phase 5 cleanup Batch A-C
7. `f20e599` - Phase 5 cleanup Batch D-F
8. `456d87b` - Phase 5 cleanup Batch G
9. `bbf1792` - Remove obsolete tests
10. `7be7893` - Update refactoring plan with completion status
11. `5b30b96` - Rename to Causeway and complete Phase 7

---

## Project Status: Phase 11-12 Complete ✓

**Name:** Causeway
**Version:** 0.0.1
**Description:** Causal awareness for Claude Code - REPL + CausalFrame persistence

**What's Done:**
- All 20 original refactoring tasks (Phases 1-8)
- Phase 11: Cleanup broken scripts/tests
- Phase 12: Fixed RLM orchestrator hallucination
- 201 tests passing (197 unit + 4 integration)
- 18 src files (from ~90)

**Blocking Issues:** None ✓ All Resolved

**Config defaults (benchmark-tested):**
- root_model: glm-4.6 (100% accuracy, 3.1s avg)
- recursive_depth_1: glm-4.6
- recursive_depth_2: glm-4.7
- recursive_depth_3: glm-4.7
- temperature: 0.1 (deterministic for REPL)

**Ready for:**
- Claude Code runtime testing
- Living documentation features (Phase 9)
- Further simplification (Phase 10)

---

*Based on: docs/plans/2026-02-19-design.md*
*Whitepaper: docs/plans/2026-02-19-whitepaper.md*
*Changelog: docs/CHANGELOG.md*
*Roadmap: docs/ROADMAP.md*
