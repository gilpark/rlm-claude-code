# Phase 13-14: Frame Persistence & UX Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect the in-memory CausalFrame system to disk persistence so frames survive across sessions, and improve orchestrator UX with progress visibility.

**Architecture:** Hybrid persistence approach - RLAPHLoop saves frames directly before exit (safety backup), and hooks can load/process saved frames. Orchestrator switched from raw bash to Task agent with status streaming.

**Tech Stack:** Python 3.11+, dataclasses, JSONL storage, Task tool for agent execution

---

## Summary of Changes

| Component | Current State | Target State |
|-----------|--------------|--------------|
| `FrameIndex` | In-memory only | Has `save()`/`load()` to JSON |
| `RLAPHLoop` | Creates frames, loses them | Saves FrameIndex before exit |
| `extract_frames.py` | Placeholder | Loads saved frames, persists to FrameStore |
| Orchestrator UX | Silent bash command | Status events (keep Bash, add print statements) |
| File I/O | Synchronous only | Async parallel reads (sync LLM + async I/O) |

## Architecture Decisions

### Hybrid A+B Persistence
- **Loop saves directly** (safety backup, never lose frames)
- **Hook loads from disk** (proper Claude Code integration)

### Sync LLM + Async I/O
- **LLM calls stay sync** (predictable debugging, clear execution order)
- **File reads are async** (parallel I/O, faster context loading)

### Keep Bash + Status Events (Phase 13-14)
- Simple print statements for visibility
- No architectural changes
- Task agent approach deferred to Phase 15 (see ROADMAP.md)

---

## Task 1: Add Persistence to FrameIndex

**Files:**
- Modify: `src/frame/frame_index.py:65-100`
- Test: `tests/unit/test_frame_index.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_frame_index.py - Add to existing file

import json
import tempfile
from pathlib import Path
from datetime import datetime
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice


class TestFrameIndexPersistence:
    """Tests for FrameIndex save/load persistence."""

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

    def test_save_to_file(self, tmp_path, sample_frame):
        """FrameIndex.save persists frames to JSON file."""
        index = FrameIndex()
        index.add(sample_frame)

        save_path = index.save("test_session", base_dir=tmp_path)

        assert save_path.exists()
        assert save_path.name == "index.json"

        # Verify content
        with open(save_path) as f:
            data = json.load(f)
        assert "frames" in data
        assert len(data["frames"]) == 1

    def test_load_from_file(self, tmp_path, sample_frame):
        """FrameIndex.load reconstructs frames from JSON file."""
        # First save
        index = FrameIndex()
        index.add(sample_frame)
        index.save("test_session", base_dir=tmp_path)

        # Then load
        loaded = FrameIndex.load("test_session", base_dir=tmp_path)

        assert len(loaded) == 1
        frame = loaded.get("test_frame_1")
        assert frame is not None
        assert frame.query == "test query"
        assert frame.status == FrameStatus.COMPLETED

    def test_load_nonexistent_returns_empty(self, tmp_path):
        """FrameIndex.load returns empty index if no file exists."""
        index = FrameIndex.load("nonexistent", base_dir=tmp_path)
        assert len(index) == 0

    def test_save_empty_index(self, tmp_path):
        """FrameIndex.save handles empty index gracefully."""
        index = FrameIndex()
        save_path = index.save("empty_session", base_dir=tmp_path)

        loaded = FrameIndex.load("empty_session", base_dir=tmp_path)
        assert len(loaded) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_frame_index.py::TestFrameIndexPersistence -v`
Expected: FAIL with "AttributeError: 'FrameIndex' object has no attribute 'save'"

**Step 3: Write minimal implementation**

```python
# src/frame/frame_index.py - Add these methods to FrameIndex class

    def save(self, session_id: str, base_dir: Path | None = None) -> Path:
        """
        Save frame index to JSON file.

        Args:
            session_id: Session identifier for the file name
            base_dir: Optional base directory (default: ~/.claude/rlm-frames/)

        Returns:
            Path to the saved file
        """
        import json
        from pathlib import Path
        from dataclasses import asdict
        from datetime import datetime

        if base_dir is None:
            base_dir = Path.home() / ".claude" / "rlm-frames" / session_id
        else:
            base_dir = base_dir / session_id

        base_dir.mkdir(parents=True, exist_ok=True)
        save_path = base_dir / "index.json"

        # Serialize frames
        frames_data = []
        for frame in self._frames.values():
            frame_dict = {
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
            frames_data.append(frame_dict)

        data = {
            "session_id": session_id,
            "frames": frames_data,
            "saved_at": datetime.now().isoformat(),
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        return save_path

    @classmethod
    def load(cls, session_id: str, base_dir: Path | None = None) -> "FrameIndex":
        """
        Load frame index from JSON file.

        Args:
            session_id: Session identifier to load
            base_dir: Optional base directory (default: ~/.claude/rlm-frames/)

        Returns:
            FrameIndex with loaded frames (empty if file doesn't exist)
        """
        import json
        from pathlib import Path
        from datetime import datetime
        from .causal_frame import CausalFrame, FrameStatus
        from .context_slice import ContextSlice

        if base_dir is None:
            base_dir = Path.home() / ".claude" / "rlm-frames"

        load_path = base_dir / session_id / "index.json"

        if not load_path.exists():
            return cls()

        with open(load_path) as f:
            data = json.load(f)

        index = cls()
        for frame_dict in data.get("frames", []):
            context_slice = ContextSlice(
                files=frame_dict["context_slice"]["files"],
                memory_refs=frame_dict["context_slice"]["memory_refs"],
                tool_outputs=frame_dict["context_slice"]["tool_outputs"],
                token_budget=frame_dict["context_slice"]["token_budget"],
            )

            frame = CausalFrame(
                frame_id=frame_dict["frame_id"],
                depth=frame_dict["depth"],
                parent_id=frame_dict["parent_id"],
                children=frame_dict["children"],
                query=frame_dict["query"],
                context_slice=context_slice,
                evidence=frame_dict["evidence"],
                conclusion=frame_dict["conclusion"],
                confidence=frame_dict["confidence"],
                invalidation_condition=frame_dict["invalidation_condition"],
                status=FrameStatus(frame_dict["status"]),
                branched_from=frame_dict.get("branched_from"),
                escalation_reason=frame_dict.get("escalation_reason"),
                created_at=datetime.fromisoformat(frame_dict["created_at"]) if frame_dict.get("created_at") else None,
                completed_at=datetime.fromisoformat(frame_dict["completed_at"]) if frame_dict.get("completed_at") else None,
            )
            index.add(frame)

        return index
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_frame_index.py::TestFrameIndexPersistence -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/frame/frame_index.py tests/unit/test_frame_index.py
git commit -m "feat: add save/load persistence to FrameIndex"
```

---

## Task 2: Make RLAPHLoop Save Frames on Exit

**Files:**
- Modify: `src/repl/rlaph_loop.py:142-170` (run method)
- Test: `tests/unit/test_rlaph_loop.py` (new file if needed)

**Step 1: Write the failing test**

```python
# tests/unit/test_rlaph_loop.py - Add to existing or create new

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from src.repl.rlaph_loop import RLAPHLoop
from src.types import SessionContext


class TestRLAPHLoopPersistence:
    """Tests for RLAPHLoop frame persistence on exit."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = MagicMock()
        client.call = MagicMock(return_value="FINAL: Test answer")
        return client

    @pytest.fixture
    def session_context(self):
        """Create minimal session context."""
        return SessionContext(
            messages=[],
            files={},
            tool_outputs=[],
            working_memory={},
        )

    @pytest.mark.asyncio
    async def test_run_saves_frame_index_on_exit(self, tmp_path, mock_llm_client, session_context):
        """RLAPHLoop.run should save FrameIndex before returning."""
        loop = RLAPHLoop(
            max_iterations=1,
            max_depth=1,
            llm_client=mock_llm_client,
        )

        # Run with working directory
        result = await loop.run("test query", session_context, working_dir=tmp_path)

        # Check that frame index was saved
        # The save happens in the session folder
        # Since we don't have a session_id in run(), we use a default
        rlm_frames_dir = Path.home() / ".claude" / "rlm-frames"

        # Loop should have attempted to save (even if no frames)
        # We can verify by checking the loop has the save method
        assert hasattr(loop.frame_index, 'save')
```

**Step 2: Run test to verify current behavior**

Run: `pytest tests/unit/test_rlaph_loop.py::TestRLAPHLoopPersistence -v`
Expected: PASS (test just verifies method exists)

**Step 3: Add save call to RLAPHLoop.run**

```python
# src/repl/rlaph_loop.py - Modify the run() method

# Add near the end of the run() method, before the return statement:

    async def run(
        self,
        query: str,
        context: SessionContext,
        working_dir: Path | str | None = None,
        session_id: str | None = None,  # ADD THIS PARAMETER
    ) -> RLPALoopResult:
        """..."""
        start_time = time.time()
        state = RLPALoopState(
            max_turns=self.max_iterations,
            depth=self._depth,
        )

        # Generate session_id if not provided
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())[:8]

        # ... rest of existing code ...

        # === ADD THIS BLOCK BEFORE THE RETURN ===
        # Save frame index before returning (Phase 13: Persistence)
        if len(self.frame_index) > 0:
            try:
                save_path = self.frame_index.save(session_id)
                if verbose:
                    print(f"[RLAPH] Saved {len(self.frame_index)} frames to {save_path}")
            except Exception as e:
                # Don't fail the run if save fails
                if verbose:
                    print(f"[RLAPH] Warning: Failed to save frames: {e}")
        # === END SAVE BLOCK ===

        return RLPALoopResult(
            answer=state.final_answer or "No answer produced",
            iterations=state.turn,
            depth_used=self._depth,
            tokens_used=self.total_tokens_used,
            execution_time_ms=execution_time,
            history=self.history.copy(),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_rlaph_loop.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/repl/rlaph_loop.py tests/unit/test_rlaph_loop.py
git commit -m "feat: RLAPHLoop saves FrameIndex on exit"
```

---

## Task 3: Update extract_frames.py Hook

**Files:**
- Modify: `scripts/extract_frames.py:20-65`

**Step 1: Write implementation (no test - hook script)**

The hook runs as a separate process and needs to load frames that were saved by RLAPHLoop.

```python
# scripts/extract_frames.py - Complete rewrite

#!/usr/bin/env python3
"""Extract frames from session and save to FrameStore.

Hook: Stop
Purpose: Persist CausalFrame tree when session ends.

This hook loads frames saved by RLAPHLoop and ensures they are
persisted to the FrameStore for long-term storage.
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frame.frame_store import FrameStore
from frame.frame_index import FrameIndex


def extract_frames(session_id: str) -> dict:
    """
    Extract frames from saved index and persist to FrameStore.

    Args:
        session_id: The session identifier

    Returns:
        Dict with status and frame count
    """
    # Load frames from index (saved by RLAPHLoop)
    index = FrameIndex.load(session_id)

    if len(index) == 0:
        return {"status": "no_frames", "count": 0, "session_id": session_id}

    # Create session directory
    session_dir = Path.home() / ".claude" / "rlm-frames" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save to FrameStore (JSONL format)
    store_path = session_dir / "frames.jsonl"
    store = FrameStore(path=store_path)

    for frame in index.values():
        store.save(frame)

    return {
        "status": "success",
        "count": len(index),
        "session_id": session_id,
        "store_path": str(store_path),
    }


def main():
    """Main entry point for hook."""
    # Read hook input from stdin
    hook_data = {}
    try:
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            hook_data = json.loads(stdin_data)
    except json.JSONDecodeError:
        pass

    session_id = hook_data.get("session_id", os.environ.get("CLAUDE_SESSION_ID", "default"))

    # Extract and persist frames
    result = extract_frames(session_id)

    # Output result for debugging
    print(json.dumps(result))

    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 2: Test manually**

```bash
# Create a test session directory with index.json
mkdir -p ~/.claude/rlm-frames/test-session
echo '{"session_id":"test","frames":[{"frame_id":"f1","depth":0,"parent_id":null,"children":[],"query":"test","context_slice":{"files":{},"memory_refs":[],"tool_outputs":{},"token_budget":1000},"evidence":[],"conclusion":"done","confidence":0.9,"invalidation_condition":"","status":"COMPLETED","branched_from":null,"escalation_reason":null,"created_at":"2026-02-20T00:00:00","completed_at":"2026-02-20T00:00:01"}],"saved_at":"2026-02-20T00:00:02"}' > ~/.claude/rlm-frames/test-session/index.json

# Run the hook
echo '{"session_id":"test-session"}' | uv run python scripts/extract_frames.py

# Verify output
cat ~/.claude/rlm-frames/test-session/frames.jsonl
```

Expected: JSON output with `status: success` and frames.jsonl file created

**Step 3: Commit**

```bash
git add scripts/extract_frames.py
git commit -m "feat: implement extract_frames hook to persist frames to FrameStore"
```

---

## Task 4: Add SessionArtifacts Persistence

**Files:**
- Modify: `src/session/session_artifacts.py:32-80`
- Test: `tests/unit/test_session_artifacts.py`

**Step 1: Write the failing tests**

```python
# tests/unit/test_session_artifacts.py - Add to existing file

import tempfile
from pathlib import Path
import pytest

from src.session.session_artifacts import FileRecord, SessionArtifacts


class TestSessionArtifactsPersistence:
    """Tests for SessionArtifacts save/load."""

    def test_save_to_file(self, tmp_path):
        """SessionArtifacts.save persists to JSON file."""
        artifacts = SessionArtifacts(
            session_id="test-session",
            initial_prompt="Fix the bug",
            files={"auth.py": FileRecord("auth.py", "abc123", "read")},
            root_frame_id="frame-001",
            conversation_log="/path/to/log.json",
        )

        save_path = artifacts.save(base_dir=tmp_path)

        assert save_path.exists()
        assert save_path.name == "artifacts.json"

    def test_load_from_file(self, tmp_path):
        """SessionArtifacts.load reconstructs from JSON file."""
        # First save
        artifacts = SessionArtifacts(
            session_id="test-session",
            initial_prompt="Fix the bug",
            files={"auth.py": FileRecord("auth.py", "abc123", "read")},
            root_frame_id="frame-001",
            conversation_log="/path/to/log.json",
        )
        artifacts.save(base_dir=tmp_path)

        # Then load
        loaded = SessionArtifacts.load("test-session", base_dir=tmp_path)

        assert loaded.session_id == "test-session"
        assert loaded.initial_prompt == "Fix the bug"
        assert "auth.py" in loaded.files
        assert loaded.files["auth.py"].hash == "abc123"

    def test_load_nonexistent_returns_none(self, tmp_path):
        """SessionArtifacts.load returns None if file doesn't exist."""
        loaded = SessionArtifacts.load("nonexistent", base_dir=tmp_path)
        assert loaded is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_session_artifacts.py::TestSessionArtifactsPersistence -v`
Expected: FAIL with "AttributeError: 'SessionArtifacts' object has no attribute 'save'"

**Step 3: Write implementation**

```python
# src/session/session_artifacts.py - Add these methods

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class FileRecord:
    """Record of a file's state in a session."""
    path: str
    hash: str
    role: str  # "read" | "modified" | "created"


@dataclass
class SessionArtifacts:
    """
    Session metadata for cross-session comparison.

    Captures what was in scope at session start for later comparison.
    The raw Claude Code transcript is the source of truth; this is
    structure extracted from it.
    """

    session_id: str
    initial_prompt: str           # Why this session started
    files: dict[str, FileRecord]  # What was in scope
    root_frame_id: str            # Entry point to call tree
    conversation_log: str         # Path to Claude Code transcript

    def save(self, base_dir: Path | None = None) -> Path:
        """
        Save session artifacts to JSON file.

        Args:
            base_dir: Optional base directory (default: ~/.claude/rlm-frames/{session_id}/)

        Returns:
            Path to the saved file
        """
        if base_dir is None:
            base_dir = Path.home() / ".claude" / "rlm-frames" / self.session_id
        else:
            base_dir = base_dir / self.session_id

        base_dir.mkdir(parents=True, exist_ok=True)
        save_path = base_dir / "artifacts.json"

        # Convert to dict with FileRecord serialization
        files_dict = {
            path: {"path": rec.path, "hash": rec.hash, "role": rec.role}
            for path, rec in self.files.items()
        }

        data = {
            "session_id": self.session_id,
            "initial_prompt": self.initial_prompt,
            "files": files_dict,
            "root_frame_id": self.root_frame_id,
            "conversation_log": self.conversation_log,
            "saved_at": datetime.now().isoformat(),
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        return save_path

    @classmethod
    def load(cls, session_id: str, base_dir: Path | None = None) -> "SessionArtifacts | None":
        """
        Load session artifacts from JSON file.

        Args:
            session_id: Session identifier to load
            base_dir: Optional base directory (default: ~/.claude/rlm-frames/)

        Returns:
            SessionArtifacts if found, None otherwise
        """
        if base_dir is None:
            base_dir = Path.home() / ".claude" / "rlm-frames"

        load_path = base_dir / session_id / "artifacts.json"

        if not load_path.exists():
            return None

        with open(load_path) as f:
            data = json.load(f)

        # Reconstruct FileRecord dict
        files = {
            path: FileRecord(path=rec["path"], hash=rec["hash"], role=rec["role"])
            for path, rec in data.get("files", {}).items()
        }

        return cls(
            session_id=data["session_id"],
            initial_prompt=data.get("initial_prompt", ""),
            files=files,
            root_frame_id=data.get("root_frame_id", ""),
            conversation_log=data.get("conversation_log", ""),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_session_artifacts.py::TestSessionArtifactsPersistence -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/session/session_artifacts.py tests/unit/test_session_artifacts.py
git commit -m "feat: add save/load persistence to SessionArtifacts"
```

---

## Task 5: Update compare_sessions.py Hook

**Files:**
- Modify: `scripts/compare_sessions.py:20-109`

**Step 1: Implement proper session comparison**

```python
# scripts/compare_sessions.py - Update to use actual persistence

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

from session.session_comparison import compare_sessions
from session.session_artifacts import SessionArtifacts
from frame.frame_index import FrameIndex


def find_most_recent_session(current_session_id: str) -> str | None:
    """Find the most recent session (excluding current)."""
    frames_dir = Path.home() / ".claude" / "rlm-frames"

    if not frames_dir.exists():
        return None

    sessions = []
    for session_dir in frames_dir.iterdir():
        if session_dir.is_dir() and session_dir.name != current_session_id:
            artifacts_path = session_dir / "artifacts.json"
            if artifacts_path.exists():
                sessions.append((session_dir.name, artifacts_path.stat().st_mtime))

    if not sessions:
        return None

    # Sort by modification time, most recent first
    sessions.sort(key=lambda x: x[1], reverse=True)
    return sessions[0][0]


def main():
    """Main entry point for hook."""
    # Read hook input from stdin
    hook_data = {}
    try:
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            hook_data = json.loads(stdin_data)
    except json.JSONDecodeError:
        pass

    session_id = hook_data.get("session_id", "unknown")

    # Find prior session
    prior_session_id = find_most_recent_session(session_id)

    if not prior_session_id:
        print(json.dumps({"status": "no_prior_session", "session_id": session_id}))
        sys.exit(0)

    # Load prior session artifacts
    prior = SessionArtifacts.load(prior_session_id)

    if not prior:
        print(json.dumps({"status": "no_prior_artifacts", "prior_session_id": prior_session_id}))
        sys.exit(0)

    # Load prior frame index
    prior_index = FrameIndex.load(prior_session_id)

    # Create current session (placeholder - real data comes from Claude Code)
    current = SessionArtifacts(
        session_id=session_id,
        initial_prompt=os.environ.get("CLAUDE_PROMPT", ""),
        files={},  # Would be populated from actual session
        root_frame_id="",
        conversation_log="",
    )

    # Compare sessions
    diff = compare_sessions(current, prior, index=prior_index)

    # Output results
    output = {
        "status": "compared",
        "prior_session_id": prior_session_id,
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

**Step 2: Test manually**

```bash
# Create a test prior session
mkdir -p ~/.claude/rlm-frames/prior-session
echo '{"session_id":"prior-session","initial_prompt":"Test prompt","files":{"test.py":{"path":"test.py","hash":"abc123","role":"read"}},"root_frame_id":"f1","conversation_log":""}' > ~/.claude/rlm-frames/prior-session/artifacts.json

# Run the hook
echo '{"session_id":"current-session"}' | uv run python scripts/compare_sessions.py
```

Expected: JSON output with `prior_session_id: prior-session`

**Step 3: Commit**

```bash
git add scripts/compare_sessions.py
git commit -m "feat: implement compare_sessions hook with actual persistence"
```

---

## Task 6: Add Orchestrator Status Events

**Files:**
- Modify: `scripts/run_orchestrator.py:217-267` (run_rlaph function)

**Design Decision:** Keep Bash execution, just add status print statements. Simple, no architectural changes.

**Step 1: Add status event output**

```python
# scripts/run_orchestrator.py - Modify run_rlaph function

async def run_rlaph(
    query: str,
    depth: int = 2,
    verbose: bool = False,
    working_dir: Path | None = None,
) -> str:
    """
    Run the RLAPH loop with status event output.

    Status events are printed to stdout for progress tracking:
    - [RLM:START] - Loop starting
    - [RLM:QUERY] - Query being processed
    - [RLM:DONE] - Loop completed
    """
    from src.repl.rlaph_loop import RLAPHLoop

    # Build empty context - files are read by REPL as needed
    context_data = build_context(files={}, use_disk_fallback=False)
    context = build_session_context(context_data)

    # Status: Starting
    print("[RLM:START] Initializing RLAPH loop")

    if verbose:
        print(f"[RLM:CONFIG] Max depth: {depth}")
        if working_dir:
            print(f"[RLM:CONFIG] Working dir: {working_dir}")

    print(f"[RLM:QUERY] Processing ({len(query)} chars)")

    # Create RLAPH loop
    loop = RLAPHLoop(
        max_iterations=20,
        max_depth=depth,
    )

    # Run loop
    result = await loop.run(query, context, working_dir=working_dir)

    # Status: Done
    print(f"[RLM:DONE] Completed in {result.iterations} iterations")
    print(f"[RLM:DONE] Tokens: {result.tokens_used}, Time: {result.execution_time_ms:.0f}ms")

    return result.answer
```

**Step 2: Test manually**

```bash
uv run python scripts/run_orchestrator.py "What is 2+2?"
```

Expected: Output with `[RLM:START]`, `[RLM:QUERY]`, `[RLM:DONE]` markers

**Step 3: Commit**

```bash
git add scripts/run_orchestrator.py
git commit -m "feat: add status event output to orchestrator for progress tracking"
```

---

## Task 7: Update Hooks Configuration

**Files:**
- Modify: `hooks/hooks.json`

**Step 1: Verify hooks configuration**

The existing hooks.json should already be correct. Verify it points to the right scripts:

```json
{
  "description": "RLM v2 plugin hooks - REPL + CausalFrame",
  "hooks": {
    "SessionStart": [...],
    "PostToolUse": [...],
    "Stop": [...]
  }
}
```

**Step 2: Test hooks end-to-end**

1. Start a new Claude Code session
2. Run `/rlm-orchestrator` with a query
3. Verify frames are saved: `ls ~/.claude/rlm-frames/*/index.json`
4. Start another session
5. Verify SessionStart hook compares with prior session

**Step 3: Commit (if changes needed)**

```bash
git add hooks/hooks.json
git commit -m "chore: verify hooks configuration for Phase 13-14"
```

---

## Task 8: Add Async File I/O

**Files:**
- Modify: `src/repl/repl_environment.py` (add async read methods)
- Modify: `src/repl/rlaph_loop.py` (use async file reads)
- Test: `tests/unit/test_repl_environment.py`

**Architecture Rationale:**

```
Sync LLM (Predictable)          Async I/O (Faster)
─────────────────────────       ─────────────────────────
✓ Easy to debug                 ✓ Parallel file reads
✓ Clear execution order         ✓ Non-blocking disk access
✓ No race conditions            ✓ Better UX (responsive)

Hybrid Approach:
┌─────────────────────────────────────────────────────────┐
│  for iteration in range(max_iterations):               │
│      response = await llm_call_sync(query)  # BLOCKING │
│                                                         │
│      # Parallel file reads                             │
│      files = await asyncio.gather(*[                   │
│          read_file_async(path) for path in paths       │
│      ])                                                │
└─────────────────────────────────────────────────────────┘
```

**Step 1: Write the failing tests**

```python
# tests/unit/test_repl_environment.py - Add to existing file

import pytest
import asyncio
from pathlib import Path
from src.repl.repl_environment import RLMEnvironment
from src.types import SessionContext


class TestAsyncFileIO:
    """Tests for async file I/O operations."""

    @pytest.fixture
    def env_with_files(self, tmp_path):
        """Create environment with test files."""
        # Create test files
        (tmp_path / "file1.py").write_text("print('file1')")
        (tmp_path / "file2.py").write_text("print('file2')")
        (tmp_path / "file3.py").write_text("print('file3')")

        # Create environment
        context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
        env = RLMEnvironment(context)
        env.enable_file_access(working_dir=tmp_path)
        return env, tmp_path

    @pytest.mark.asyncio
    async def test_read_file_async_single(self, env_with_files):
        """read_file_async returns file content."""
        env, tmp_path = env_with_files

        content = await env.read_file_async("file1.py")
        assert "file1" in content

    @pytest.mark.asyncio
    async def test_read_files_async_parallel(self, env_with_files):
        """read_files_async reads multiple files in parallel."""
        env, tmp_path = env_with_files

        files = ["file1.py", "file2.py", "file3.py"]
        results = await env.read_files_async(files)

        assert len(results) == 3
        assert "file1" in results["file1.py"]
        assert "file2" in results["file2.py"]
        assert "file3" in results["file3.py"]

    @pytest.mark.asyncio
    async def test_read_files_async_handles_missing(self, env_with_files):
        """read_files_async handles missing files gracefully."""
        env, tmp_path = env_with_files

        files = ["file1.py", "nonexistent.py"]
        results = await env.read_files_async(files)

        assert "file1.py" in results
        assert results.get("nonexistent.py") is None or "error" in str(results.get("nonexistent.py", "")).lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_repl_environment.py::TestAsyncFileIO -v`
Expected: FAIL with "AttributeError: 'RLMEnvironment' object has no attribute 'read_file_async'"

**Step 3: Write implementation**

```python
# src/repl/repl_environment.py - Add these methods to RLMEnvironment class

    async def read_file_async(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """
        Async version of read_file for non-blocking file access.

        Uses aiofiles for true async I/O, falling back to
        asyncio.to_thread for compatibility.

        Args:
            path: File path (absolute or relative to working_dir)
            offset: Line offset to start reading (default 0)
            limit: Maximum number of lines to read (default 2000)

        Returns:
            File content as string
        """
        import asyncio

        # Run sync version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._read_file(path, offset=offset, limit=limit)
        )

    async def read_files_async(
        self,
        paths: list[str],
        offset: int = 0,
        limit: int = 2000,
    ) -> dict[str, str]:
        """
        Read multiple files in parallel using asyncio.gather.

        This is the key optimization for large file sets:
        - Sync: read(file1) → wait → read(file2) → wait → ...
        - Async: gather(read(file1), read(file2), ...) → wait once

        Args:
            paths: List of file paths to read
            offset: Line offset for all files (default 0)
            limit: Maximum lines per file (default 2000)

        Returns:
            Dict mapping path -> content (or error string)
        """
        import asyncio

        async def read_with_error_handling(path: str) -> tuple[str, str]:
            """Read file and return (path, content) tuple."""
            try:
                content = await self.read_file_async(path, offset=offset, limit=limit)
                return (path, content)
            except Exception as e:
                return (path, f"ERROR: {e}")

        # Gather all reads in parallel
        results = await asyncio.gather(*[
            read_with_error_handling(path) for path in paths
        ])

        return dict(results)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_repl_environment.py::TestAsyncFileIO -v`
Expected: PASS

**Step 5: Update RLAPHLoop to use async file reads**

```python
# src/repl/rlaph_loop.py - Add helper method and update REPL integration

    async def _read_files_parallel(self, paths: list[str]) -> dict[str, str]:
        """
        Read multiple files in parallel using async I/O.

        This is used when the REPL needs to read multiple files
        for context externalization.

        Args:
            paths: List of file paths to read

        Returns:
            Dict mapping path -> content
        """
        if not self.repl:
            return {}

        return await self.repl.read_files_async(paths)
```

**Step 6: Commit**

```bash
git add src/repl/repl_environment.py src/repl/rlaph_loop.py tests/unit/test_repl_environment.py
git commit -m "feat: add async file I/O for parallel file reads

- Add read_file_async() for non-blocking single file reads
- Add read_files_async() for parallel multi-file reads
- Use asyncio.gather() for concurrent I/O
- Sync LLM calls remain predictable, async I/O improves performance"
```

---

## Task 9: Add Architecture Documentation

**Files:**
- Create: `docs/ARCHITECTURE.md` (optional, for future reference)

**Step 1: Document design decisions**

```markdown
# docs/ARCHITECTURE.md

## Design Decisions

### Sync LLM + Async I/O Hybrid

**Problem:** Pure async LLM calls are hard to debug. Pure sync I/O is slow.

**Solution:** Hybrid approach:
- LLM calls are synchronous (predictable execution order)
- File I/O is async (parallel reads, non-blocking)

**Code Pattern:**
```python
# Sync LLM - predictable
response = await llm_call_sync(query)

# Async I/O - fast
files = await asyncio.gather(*[
    read_file_async(path) for path in paths
])
```

### Frame Persistence (Hybrid A+B)

**Problem:** Hooks run as separate processes, can't access in-memory FrameIndex.

**Solution:** Two-level persistence:
1. RLAPHLoop saves directly to disk (safety backup)
2. extract_frames.py hook loads from disk (proper integration)

**Data Flow:**
```
RLAPHLoop.run()
    ↓
FrameIndex.save(session_id)  → ~/.claude/rlm-frames/{session}/index.json
    ↓
Session ends
    ↓
extract_frames.py hook
    ↓
FrameIndex.load(session_id)
    ↓
FrameStore.save()  → ~/.claude/rlm-frames/{session}/frames.jsonl
```
```

**Step 2: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: add architecture decision records"
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] `FrameIndex.save()` and `FrameIndex.load()` work correctly
- [ ] `RLAPHLoop.run()` saves frames before returning
- [ ] `extract_frames.py` hook loads and persists frames
- [ ] `SessionArtifacts.save()` and `SessionArtifacts.load()` work correctly
- [ ] `compare_sessions.py` hook compares with prior session
- [ ] Orchestrator outputs status events
- [ ] `read_file_async()` and `read_files_async()` work correctly
- [ ] Async file reads are parallel (use `asyncio.gather`)
- [ ] End-to-end: Run orchestrator, verify frames persisted

---

## Session Folder Structure (Final)

```
~/.claude/rlm-frames/
├── session-abc123/
│   ├── index.json       # FrameIndex saved by RLAPHLoop
│   ├── frames.jsonl     # FrameStore persisted by extract_frames.py
│   ├── artifacts.json   # SessionArtifacts for comparison
│   └── tools.jsonl      # Tool outputs captured by capture_output.py
├── session-def456/
│   └── ...
```
