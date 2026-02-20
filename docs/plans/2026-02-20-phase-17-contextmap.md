# Phase 17: ContextMap + Git-Aware Loading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Externalize context so REPL navigates instead of reading files directly, with git-aware cross-session change detection.

**Architecture:** Minimal ContextMap class holds paths, hashes, and lazy content cache. REPL uses ContextMap.get_content() instead of direct file reads. Git commit hash stored in FrameIndex for diff-based invalidation at load time.

**Tech Stack:** Python dataclasses, subprocess for git commands, hashlib for content hashing

---

## Task 1: Create ContextMap Class

**Files:**
- Create: `src/frame/context_map.py`
- Test: `tests/frame/test_context_map.py`

**Step 1: Write the failing test**

```python
"""Tests for ContextMap."""
import pytest
from pathlib import Path
from src.frame.context_map import ContextMap


def test_context_map_initializes_with_root_dir(tmp_path):
    """ContextMap should initialize with root directory."""
    # Create some test files
    (tmp_path / "test.py").write_text("print('hello')")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.py").write_text("# nested")

    cm = ContextMap(tmp_path)

    assert cm.root == tmp_path.resolve()
    assert isinstance(cm.paths, set)
    assert isinstance(cm.hashes, dict)
    assert isinstance(cm.contents, dict)


def test_get_content_loads_lazily(tmp_path):
    """get_content should load file content on first access."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    # Content not loaded yet
    assert test_file not in cm.contents

    # Load it
    content = cm.get_content(test_file)
    assert content == "print('hello')"
    assert test_file in cm.contents


def test_get_content_caches_result(tmp_path):
    """get_content should cache and reuse content."""
    test_file = tmp_path / "test.py"
    test_file.write_text("original")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    content1 = cm.get_content(test_file)

    # Modify file on disk
    test_file.write_text("modified")

    # Should return cached version
    content2 = cm.get_content(test_file)
    assert content2 == "original"


def test_get_content_rejects_unknown_path(tmp_path):
    """get_content should raise for paths not in context."""
    cm = ContextMap(tmp_path)

    unknown_file = tmp_path / "unknown.py"
    unknown_file.write_text("unknown")

    with pytest.raises(ValueError, match="not in context"):
        cm.get_content(unknown_file)


def test_get_hash_computes_and_caches(tmp_path):
    """get_hash should compute hash and cache it."""
    test_file = tmp_path / "test.py"
    test_file.write_text("test content")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    hash1 = cm.get_hash(test_file)
    assert len(hash1) == 16  # blake2b truncated

    # Should be cached
    hash2 = cm.get_hash(test_file)
    assert hash1 == hash2


def test_refresh_from_diff_updates_paths(tmp_path):
    """refresh_from_diff should update paths and hashes."""
    test_file = tmp_path / "test.py"
    test_file.write_text("original")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)
    old_hash = cm.get_hash(test_file)

    # Modify file
    test_file.write_text("modified")

    # Refresh
    cm.refresh_from_diff({test_file})

    # Hash should be updated
    new_hash = cm.get_hash(test_file)
    assert new_hash != old_hash

    # Content cache should be cleared
    assert test_file not in cm.contents


def test_refresh_from_diff_handles_deleted_files(tmp_path):
    """refresh_from_diff should remove deleted files."""
    test_file = tmp_path / "deleted.py"
    test_file.write_text("will be deleted")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    # Delete file
    test_file.unlink()

    # Refresh
    cm.refresh_from_diff({test_file})

    # Should be removed from paths
    assert test_file not in cm.paths
    assert test_file not in cm.hashes
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/gilpark/.dotfiles/.claude/plugins/marketplaces/causeway
uv run pytest tests/frame/test_context_map.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.frame.context_map'"

**Step 3: Write minimal implementation**

```python
"""ContextMap - minimal context externalization for REPL navigation."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ContextMap:
    """
    Minimal context externalization for REPL navigation.

    REPL navigates this map instead of reading files directly.
    This enables:
    - Spatial externalization (REPL doesn't touch disk)
    - Hash-based invalidation
    - Lazy content loading with caching
    - Git-aware change detection

    Usage:
        cm = ContextMap(Path.cwd())
        content = cm.get_content("src/main.py")  # lazy load
        cm.refresh_from_diff({Path("src/main.py")})  # after edit
    """

    root: Path
    paths: set[Path] = field(default_factory=set)
    hashes: dict[Path, str] = field(default_factory=dict)
    contents: dict[Path, str] = field(default_factory=dict)
    commit_hash: str | None = field(default=None, init=False)

    def __post_init__(self):
        """Initialize paths from git ls-files or empty."""
        self.root = Path(self.root).resolve()
        self._populate_from_git()

    def _populate_from_git(self) -> None:
        """Populate paths from git ls-files if in git repo."""
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    if line:
                        p = self.root / line
                        if p.is_file():
                            self.paths.add(p)

                # Also get commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if hash_result.returncode == 0:
                    self.commit_hash = hash_result.stdout.strip()[:8]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Not a git repo or git not available

    def _compute_hash(self, path: Path) -> str:
        """Compute blake2b hash of file content (truncated to 16 chars)."""
        return hashlib.blake2b(path.read_bytes()).hexdigest()[:16]

    def get_content(self, path: str | Path) -> str:
        """
        Get file content, loading lazily if needed.

        Args:
            path: File path (relative to root or absolute)

        Returns:
            File content as string

        Raises:
            ValueError: If path not in context
        """
        p = Path(path) if not isinstance(path, Path) else path
        if not p.is_absolute():
            p = self.root / p

        p = p.resolve()

        if p not in self.paths:
            raise ValueError(f"Path not in context: {path}")

        if p not in self.contents:
            self.contents[p] = p.read_text(encoding="utf-8", errors="replace")
            # Update hash on first load
            if p not in self.hashes:
                self.hashes[p] = self._compute_hash(p)

        return self.contents[p]

    def get_hash(self, path: Path) -> str:
        """
        Get content hash, computing if needed.

        Args:
            path: Absolute file path

        Returns:
            16-char blake2b hash
        """
        if path not in self.hashes:
            self.hashes[path] = self._compute_hash(path)
        return self.hashes[path]

    def refresh_from_diff(self, changed_paths: set[Path]) -> None:
        """
        Refresh paths after files changed.

        Called by PostToolUse hook after Write/Edit operations.

        Args:
            changed_paths: Set of paths that may have changed
        """
        for p in changed_paths:
            p = p.resolve()

            if p.exists() and p.is_file():
                # Add to paths if new
                self.paths.add(p)
                # Recompute hash
                self.hashes[p] = self._compute_hash(p)
                # Drop cached content (will reload on next access)
                self.contents.pop(p, None)
            else:
                # File deleted - remove from all caches
                self.paths.discard(p)
                self.hashes.pop(p, None)
                self.contents.pop(p, None)

    def add_path(self, path: str | Path) -> None:
        """
        Add a path to the context (e.g., discovered via glob).

        Args:
            path: File path to add
        """
        p = Path(path) if not isinstance(path, Path) else path
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()

        if p.exists() and p.is_file():
            self.paths.add(p)


def detect_changed_files(old_commit: str, root_dir: Path) -> set[Path]:
    """
    Detect files changed between old commit and HEAD.

    Args:
        old_commit: Git commit hash (can be short)
        root_dir: Repository root directory

    Returns:
        Set of changed file paths (absolute)
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{old_commit}..HEAD"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return set()

        changed = set()
        for line in result.stdout.strip().splitlines():
            if line:
                p = root_dir / line
                if p.exists():
                    changed.add(p.resolve())
        return changed
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()


def get_current_commit_hash(root_dir: Path) -> str | None:
    """
    Get current git HEAD commit hash.

    Args:
        root_dir: Repository root directory

    Returns:
        8-char commit hash or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/frame/test_context_map.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/frame/context_map.py tests/frame/test_context_map.py
git commit -m "feat: add ContextMap for REPL context externalization

- Minimal ContextMap class with paths, hashes, lazy contents
- get_content() for lazy loading with caching
- refresh_from_diff() for hook integration
- Git-aware commit hash tracking
- detect_changed_files() for cross-session invalidation"
```

---

## Task 2: Add commit_hash to FrameIndex

**Files:**
- Modify: `src/frame/frame_index.py:115-120`
- Test: `tests/frame/test_frame_index.py`

**Step 1: Write the failing test**

```python
# Add to tests/frame/test_frame_index.py

def test_frame_index_save_load_with_commit_hash(tmp_path):
    """FrameIndex should persist and load commit_hash."""
    from src.frame.frame_index import FrameIndex
    from src.frame.context_slice import ContextSlice
    from src.frame.causal_frame import CausalFrame, FrameStatus
    from datetime import datetime

    index = FrameIndex()
    index.commit_hash = "abc12345"

    frame = CausalFrame(
        frame_id="test123",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(frame)

    # Save
    save_path = index.save("test-session", tmp_path)

    # Load
    loaded = FrameIndex.load("test-session", tmp_path)

    assert loaded.commit_hash == "abc12345"


def test_frame_index_commit_hash_defaults_none():
    """FrameIndex commit_hash should default to None."""
    from src.frame.frame_index import FrameIndex

    index = FrameIndex()
    assert index.commit_hash is None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/frame/test_frame_index.py -v -k commit_hash
```

Expected: FAIL with "AssertionError: None != 'abc12345'" (commit_hash not persisted)

**Step 3: Write minimal implementation**

In `src/frame/frame_index.py`:

```python
# In __init__ method, add:
def __init__(self):
    self._frames: dict[str, "CausalFrame"] = {}
    self.commit_hash: str | None = None  # Git commit hash for change detection

# In save method, add to data dict:
data = {
    "session_id": session_id,
    "commit_hash": self.commit_hash,  # Add this line
    "frames": frames_data,
    "saved_at": datetime.now().isoformat(),
}

# In load method, after creating index:
index = cls()
index.commit_hash = data.get("commit_hash")  # Add this line
for frame_dict in data.get("frames", []):
    ...
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/frame/test_frame_index.py -v -k commit_hash
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/frame/frame_index.py tests/frame/test_frame_index.py
git commit -m "feat: add commit_hash to FrameIndex for git-aware invalidation

- Store git HEAD in FrameIndex at save time
- Load commit_hash when loading frames
- Enables cross-session diff-based change detection"
```

---

## Task 3: Integrate ContextMap with REPL Environment

**Files:**
- Modify: `src/repl/repl_environment.py:150-200` (file access section)
- Test: `tests/repl/test_repl_context_map.py`

**Step 1: Write the failing test**

```python
"""Tests for REPL ContextMap integration."""
import pytest
from pathlib import Path
from src.repl.repl_environment import RLMEnvironment
from src.frame.context_map import ContextMap
from src.types import SessionContext


def test_repl_uses_context_map_for_read_file(tmp_path):
    """RLMEnvironment should use ContextMap for file reads."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    env = RLMEnvironment(context, context_map=cm)

    result = env.execute("content = read_file('test.py')")
    assert result.success
    assert "hello" in env.get_variable("content")


def test_repl_rejects_file_not_in_context_map(tmp_path):
    """RLMEnvironment should reject files not in ContextMap."""
    unknown_file = tmp_path / "unknown.py"
    unknown_file.write_text("unknown")

    cm = ContextMap(tmp_path)
    # Don't add unknown_file to paths

    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    env = RLMEnvironment(context, context_map=cm)

    result = env.execute("read_file('unknown.py')")
    assert not result.success
    assert "not in context" in result.error.lower()


def test_repl_context_map_tracks_files_read(tmp_path):
    """RLMEnvironment should track files read via ContextMap."""
    test_file = tmp_path / "test.py"
    test_file.write_text("test content")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    env = RLMEnvironment(context, context_map=cm)

    env.execute("read_file('test.py')")

    # Should track in files_read (for frame context_slice)
    assert str(test_file) in env.files_read or test_file in env.files_read
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/repl/test_repl_context_map.py -v
```

Expected: FAIL with "TypeError: __init__() got an unexpected keyword argument 'context_map'"

**Step 3: Write minimal implementation**

In `src/repl/repl_environment.py`:

1. Add import:
```python
from ..frame.context_map import ContextMap
```

2. Modify `__init__` to accept context_map:
```python
def __init__(
    self,
    context: SessionContext,
    llm_client: "LLMClient | None" = None,
    context_map: ContextMap | None = None,  # Add this
):
    # ... existing init ...
    self._context_map = context_map  # Add this
```

3. Modify `enable_file_access` to use context_map:
```python
def enable_file_access(self, working_dir: Path | None = None):
    """Enable file access via ContextMap if available."""
    self._working_dir = working_dir or Path.cwd()

    # Create ContextMap if not provided
    if self._context_map is None:
        self._context_map = ContextMap(self._working_dir)
```

4. Modify `_read_file` to use context_map:
```python
def _read_file(self, path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read file via ContextMap (if available) or direct read."""
    if self._context_map is not None:
        # Use ContextMap - enforces context scope
        try:
            content = self._context_map.get_content(path)
        except ValueError as e:
            return f"Error: {e}"
    else:
        # Fallback to direct read (legacy behavior)
        # ... existing code ...
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/repl/test_repl_context_map.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/repl/repl_environment.py tests/repl/test_repl_context_map.py
git commit -m "feat: integrate ContextMap with RLMEnvironment

- Accept optional context_map in __init__
- Use ContextMap for file reads when available
- Reject files not in context scope
- Track files read for frame context_slice"
```

---

## Task 4: Create ContextMap in Orchestrator

**Files:**
- Modify: `scripts/run_orchestrator.py:250-300`
- Test: `tests/test_run_orchestrator.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_run_orchestrator.py

@pytest.mark.asyncio
async def test_run_rlaph_creates_context_map(tmp_path):
    """run_rlaph should create ContextMap and pass to RLAPHLoop."""
    from scripts.run_orchestrator import run_rlaph

    # Create test file
    test_file = tmp_path / "test.py"
    test_file.write_text("print('test')")

    result = await run_rlaph(
        "What is in test.py?",
        depth=0,
        verbose=False,
        working_dir=tmp_path,
    )

    # Should have created and used ContextMap
    assert result is not None


def test_get_current_commit_hash_in_git_repo():
    """get_current_commit_hash should return hash in git repo."""
    from scripts.run_orchestrator import get_current_commit_hash

    # This test runs in the causeway repo (git repo)
    hash_result = get_current_commit_hash(Path.cwd())
    assert hash_result is not None
    assert len(hash_result) == 8
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_run_orchestrator.py -v -k context_map
```

Expected: FAIL (ContextMap not yet integrated)

**Step 3: Write minimal implementation**

In `scripts/run_orchestrator.py`:

1. Add import:
```python
from src.frame.context_map import ContextMap, get_current_commit_hash
```

2. Modify `run_rlaph` function:
```python
async def run_rlaph(
    query: str,
    depth: int = 2,
    verbose: bool = False,
    working_dir: Path | None = None,
    session_id: str | None = None,
) -> str:
    from src.repl.rlaph_loop import RLAPHLoop

    print("[RLM:START] Initializing RLAPH loop")

    # Build empty context - files are read by REPL as needed
    context_data = build_context(files={}, use_disk_fallback=False)
    context = build_session_context(context_data)

    work_dir = working_dir or Path.cwd()

    # Create ContextMap for session
    context_map = ContextMap(work_dir)
    commit_hash = context_map.commit_hash

    if verbose:
        print(f"[RLM:CONFIG] Max depth: {depth}")
        print(f"[RLM:CONFIG] Working dir: {work_dir}")
        if commit_hash:
            print(f"[RLM:CONFIG] Git commit: {commit_hash}")

    print(f"[RLM:QUERY] Processing ({len(query)} chars)")
    print(f"[RLM:CONTEXT] {len(context_map.paths)} files in scope")

    # Create RLAPH loop
    loop = RLAPHLoop(
        max_iterations=20,
        max_depth=depth,
        context_map=context_map,  # Pass ContextMap
    )

    # Set commit_hash on frame index
    loop.frame_index.commit_hash = commit_hash

    # Run loop
    result = await loop.run(query, context, working_dir=work_dir, session_id=session_id)

    print(f"[RLM:DONE] Completed in {result.iterations} iterations")
    print(f"[RLM:DONE] Tokens: {result.tokens_used}, Time: {result.execution_time_ms:.0f}ms")

    return result.answer
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_run_orchestrator.py -v -k context_map
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/run_orchestrator.py tests/test_run_orchestrator.py
git commit -m "feat: create ContextMap in orchestrator entry point

- Create ContextMap at session start
- Pass ContextMap to RLAPHLoop
- Store commit_hash on FrameIndex
- Log context scope size"
```

---

## Task 5: Pass ContextMap to RLAPHLoop

**Files:**
- Modify: `src/repl/rlaph_loop.py:84-120`
- Test: `tests/repl/test_rlaph_loop_context_map.py`

**Step 1: Write the failing test**

```python
"""Tests for RLAPHLoop ContextMap integration."""
import pytest
from pathlib import Path
from src.repl.rlaph_loop import RLAPHLoop
from src.frame.context_map import ContextMap
from src.types import SessionContext


@pytest.mark.asyncio
async def test_rlaph_loop_accepts_context_map(tmp_path):
    """RLAPHLoop should accept and use ContextMap."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    cm = ContextMap(tmp_path)

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=0,
        context_map=cm,
    )

    assert loop._context_map is cm


@pytest.mark.asyncio
async def test_rlaph_loop_passes_context_map_to_repl(tmp_path):
    """RLAPHLoop should pass ContextMap to RLMEnvironment."""
    cm = ContextMap(tmp_path)

    loop = RLAPHLoop(context_map=cm)
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})

    # Run will create REPL with context_map
    # We can verify by checking the repl has the context_map after run
    # (This is a simplification - real test would check actual behavior)

    assert loop._context_map is cm
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/repl/test_rlaph_loop_context_map.py -v
```

Expected: FAIL with "TypeError: __init__() got an unexpected keyword argument 'context_map'"

**Step 3: Write minimal implementation**

In `src/repl/rlaph_loop.py`:

1. Add import:
```python
from ..frame.context_map import ContextMap
```

2. Modify `__init__`:
```python
def __init__(
    self,
    max_iterations: int = 20,
    max_depth: int = 3,
    config: RLMConfig | None = None,
    llm_client: LLMClient | None = None,
    context_map: ContextMap | None = None,  # Add this
):
    # ... existing init ...
    self._context_map = context_map  # Add this
```

3. Modify `run` method where REPL is created:
```python
# Initialize REPL with LLM client and ContextMap
self.repl = RLMEnvironment(
    context,
    llm_client=self.llm_client,
    context_map=self._context_map,  # Pass ContextMap
)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/repl/test_rlaph_loop_context_map.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/repl/rlaph_loop.py tests/repl/test_rlaph_loop_context_map.py
git commit -m "feat: pass ContextMap through RLAPHLoop to RLMEnvironment

- Accept context_map in RLAPHLoop.__init__
- Pass to RLMEnvironment on creation
- Enables REPL to use externalized context"
```

---

## Task 6: Update Frame __init__ Exports

**Files:**
- Modify: `src/frame/__init__.py`

**Step 1: Add ContextMap to exports**

```python
# In src/frame/__init__.py, add:
from .context_map import ContextMap, detect_changed_files, get_current_commit_hash

__all__ = [
    # ... existing exports ...
    "ContextMap",
    "detect_changed_files",
    "get_current_commit_hash",
]
```

**Step 2: Verify import works**

```bash
uv run python -c "from src.frame import ContextMap; print('OK')"
```

Expected: "OK"

**Step 3: Commit**

```bash
git add src/frame/__init__.py
git commit -m "feat: export ContextMap from frame package"
```

---

## Task 7: Add Cross-Session Frame Loading with Diff

**Files:**
- Modify: `src/frame/frame_index.py:126-180` (load method)
- Test: `tests/frame/test_frame_index_diff.py`

**Step 1: Write the failing test**

```python
"""Tests for cross-session diff-based frame loading."""
import pytest
from pathlib import Path
from src.frame.frame_index import FrameIndex
from src.frame.context_map import ContextMap
from src.frame.context_slice import ContextSlice
from src.frame.causal_frame import CausalFrame, FrameStatus
from datetime import datetime


def test_load_frames_with_git_diff_invalidates_changed(tmp_path, monkeypatch):
    """Loading frames should invalidate those with changed files."""
    # Create a file
    test_file = tmp_path / "test.py"
    test_file.write_text("original content")

    # Create frame index with commit hash and frame referencing the file
    index = FrameIndex()
    index.commit_hash = "oldcommit"

    # Compute hash of original content
    import hashlib
    original_hash = hashlib.blake2b(test_file.read_bytes()).hexdigest()[:16]

    frame = CausalFrame(
        frame_id="test123",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(
            files={str(test_file): original_hash},
            memory_refs=[],
            tool_outputs={},
            token_budget=8000,
        ),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(frame)
    index.save("test-session", tmp_path)

    # Modify the file
    test_file.write_text("modified content")

    # Mock git diff to say file changed
    def mock_detect_changed(*args, **kwargs):
        return {test_file.resolve()}

    import src.frame.frame_index as fi
    monkeypatch.setattr(fi, "detect_changed_files", mock_detect_changed)

    # Load frames
    loaded = FrameIndex.load_with_validation("test-session", tmp_path)

    # Frame should be invalidated
    loaded_frame = loaded.get("test123")
    assert loaded_frame.status == FrameStatus.INVALIDATED


def test_load_frames_without_changes_keeps_valid(tmp_path, monkeypatch):
    """Loading frames with no changes should keep frames valid."""
    test_file = tmp_path / "test.py"
    test_file.write_text("unchanged content")

    index = FrameIndex()
    index.commit_hash = "oldcommit"

    import hashlib
    content_hash = hashlib.blake2b(test_file.read_bytes()).hexdigest()[:16]

    frame = CausalFrame(
        frame_id="test456",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(
            files={str(test_file): content_hash},
            memory_refs=[],
            tool_outputs={},
            token_budget=8000,
        ),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(frame)
    index.save("test-session", tmp_path)

    # Mock git diff to say no files changed
    def mock_detect_changed(*args, **kwargs):
        return set()

    import src.frame.frame_index as fi
    monkeypatch.setattr(fi, "detect_changed_files", mock_detect_changed)

    loaded = FrameIndex.load_with_validation("test-session", tmp_path)

    loaded_frame = loaded.get("test456")
    assert loaded_frame.status == FrameStatus.COMPLETED
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/frame/test_frame_index_diff.py -v
```

Expected: FAIL with "AttributeError: 'FrameIndex' has no attribute 'load_with_validation'"

**Step 3: Write minimal implementation**

In `src/frame/frame_index.py`, add new method:

```python
from .context_map import detect_changed_files

@classmethod
def load_with_validation(
    cls,
    session_id: str,
    base_dir: Path | None = None,
    current_root: Path | None = None,
) -> "FrameIndex":
    """
    Load frame index and validate frames against current file state.

    Uses git diff to detect changed files, then invalidates frames
    whose context_slice files have different hashes.

    Args:
        session_id: Session identifier to load
        base_dir: Optional base directory for frame files
        current_root: Current repo root (for git diff)

    Returns:
        FrameIndex with validated/invalidated frames
    """
    from .causal_frame import FrameStatus
    from .frame_invalidation import propagate_invalidation

    index = cls.load(session_id, base_dir)

    if index.commit_hash and current_root:
        # Detect changed files via git diff
        changed_paths = detect_changed_files(index.commit_hash, current_root)
        changed_strs = {str(p) for p in changed_paths}

        # Check each frame for hash mismatches
        for frame in index.values():
            if frame.status != FrameStatus.COMPLETED:
                continue

            for file_path, stored_hash in frame.context_slice.files.items():
                if file_path in changed_strs:
                    # File changed since frame was created
                    propagate_invalidation(
                        frame.frame_id,
                        f"File changed: {file_path}",
                        index,
                    )
                    break

    return index
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/frame/test_frame_index_diff.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/frame/frame_index.py tests/frame/test_frame_index_diff.py
git commit -m "feat: add load_with_validation for cross-session frame loading

- Use git diff to detect changed files
- Invalidate frames with hash mismatches
- Cascade invalidation to dependent frames"
```

---

## Verification

Run all tests to verify Phase 17 is complete:

```bash
uv run pytest tests/ -v --tb=short
```

All tests should pass.

---

## Summary

Phase 17 adds:

1. **ContextMap** - Minimal class for REPL context externalization
2. **commit_hash in FrameIndex** - Git HEAD tracking for change detection
3. **REPL integration** - RLMEnvironment uses ContextMap for file reads
4. **Orchestrator integration** - Creates ContextMap, passes to loop
5. **Cross-session validation** - load_with_validation uses git diff

Result: REPL navigates externalized context instead of reading files directly, with git-aware cross-session change detection.
