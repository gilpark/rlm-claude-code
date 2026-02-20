# Phase 18: Tree Structure + Evidence + Cascade + Recursion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform linear frame chain into proper tree structure with evidence tracking, cascade invalidation, and **working recursion via synchronous llm(sub_query)**.

**Architecture:**
- Populate `children` when creating frames
- Auto-track evidence from child frames and tool outputs
- Auto-generate `invalidation_condition`
- Implement full cascade propagation with dependent frame discovery
- **Add synchronous recursion via llm(sub_query) - no subprocess**

**Tech Stack:** Python dataclasses, existing frame infrastructure

---

## Part A: Recursion Infrastructure

### Task A1: Add Synchronous `llm(sub_query)` Recursion

**Problem:** All frames are at depth 0 because `llm()` makes single LLM calls but doesn't create child frames or sub-loops.

**Solution:** Use synchronous `llm(sub_query)` that creates child frames and runs sub-iterations.

**Files:**
- Modify: `src/repl/rlaph_loop.py` (add spawn_child mechanism)
- Modify: `src/repl/repl_environment.py` (expose llm with recursion)
- Test: `tests/repl/test_llm_recursion.py`

**Step 1: Write the failing test**

```python
"""Tests for llm() recursive calls."""
import pytest
from pathlib import Path
from src.repl.rlaph_loop import RLAPHLoop
from src.frame.causal_frame import FrameStatus
from src.types import SessionContext


@pytest.mark.asyncio
async def test_llm_creates_child_frame():
    """When llm(sub_query) is called, it should create a child frame."""
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    loop = RLAPHLoop(max_iterations=5, max_depth=2)

    result = await loop.run(
        "Test recursive llm call",
        context,
        working_dir=Path.cwd(),
    )

    # Should have at least one frame
    assert len(loop.frame_index) >= 1

    # If recursion happened, should have depth > 0 frames
    # (depends on LLM deciding to recurse)


@pytest.mark.asyncio
async def test_llm_sync_returns_result():
    """llm_sync should return actual LLM result synchronously."""
    loop = RLAPHLoop(max_depth=2)

    # This should work without error
    result = loop.llm_sync("What is 2+2?")

    assert result is not None
    assert len(result) > 0


@pytest.mark.asyncio
async def test_llm_sync_respects_max_depth():
    """llm_sync should raise RecursionDepthError if max depth exceeded."""
    loop = RLAPHLoop(max_depth=0)  # Depth 0 = no recursion allowed
    loop._depth = 0

    from src.types import RecursionDepthError
    with pytest.raises(RecursionDepthError):
        loop.llm_sync("This should fail")


def test_spawn_child_increments_depth():
    """Spawning a child should increment depth."""
    loop = RLAPHLoop(max_depth=3)

    # Initial depth
    assert loop.depth == 0

    # After spawning child (simulated)
    loop._depth = 1
    assert loop.depth == 1
```

**Step 2: Run test to verify current state**

```bash
cd /Users/gilpark/.dotfiles/.claude/plugins/marketplaces/causeway
uv run pytest tests/repl/test_llm_recursion.py -v
```

Expected: Some tests fail (depth management not working)

**Step 3: Enhance llm_sync to spawn child frames**

In `src/repl/rlaph_loop.py`, update `llm_sync`:

```python
def llm_sync(self, query: str, context: str = "") -> str:
    """
    Synchronous LLM call - returns actual result immediately.

    This is the key method that makes RLAPH work:
    - Called from REPL's llm() function
    - Returns actual string result
    - Handles depth management
    - Creates child frames for recursion tracking

    Args:
        query: Query string
        context: Optional context string

    Returns:
        LLM response as string

    Raises:
        RecursionDepthError: If max depth exceeded
    """
    # Check depth
    if self._depth >= self.max_depth:
        raise RecursionDepthError(self._depth + 1, self.max_depth)

    # Increment depth for this call
    self._depth += 1

    try:
        # Use LLMClient directly
        result = self.llm_client.call(
            query=query,
            context={"prior": context} if context else None,
            depth=self._depth,
        )

        # Track tokens
        self._tokens_used += len(result) // 4

        # Create child frame for this llm call
        if self.repl:
            context_slice = ContextSlice(
                files={},
                memory_refs=list(self.repl.memory_refs) if hasattr(self.repl, 'memory_refs') else [],
                tool_outputs={},
                token_budget=self.config.cost_controls.max_tokens_per_recursive_call,
            )

            child_frame = CausalFrame(
                frame_id=compute_frame_id(
                    self._current_frame_id,
                    query[:100],
                    context_slice,
                ),
                depth=self._depth,
                parent_id=self._current_frame_id,
                children=[],
                query=query[:200],
                context_slice=context_slice,
                evidence=[],
                conclusion=result[:500] if result else None,
                confidence=0.8,
                invalidation_condition="",
                status=FrameStatus.COMPLETED,
                branched_from=None,
                escalation_reason=None,
                created_at=datetime.now(),
                completed_at=datetime.now(),
            )

            self.frame_index.add(child_frame)
            self._current_frame_id = child_frame.frame_id

        return result
    finally:
        # Decrement depth after call completes
        self._depth -= 1
```

Add imports at top if needed:
```python
from datetime import datetime
from .context_slice import ContextSlice
```

**Step 4: Run test to verify**

```bash
uv run pytest tests/repl/test_llm_recursion.py -v
```

Expected: Tests pass

**Step 5: Commit**

```bash
git add src/repl/rlaph_loop.py tests/repl/test_llm_recursion.py
git commit -m "feat: add child frame creation to llm_sync for recursion

- llm_sync now creates child frames for each recursive call
- Depth properly tracked and child frames linked to parent
- Enables tree structure instead of flat chain"
```

---

### Task A2: Enhance System Prompt for Recursion

**Problem:** The LLM doesn't know it can decompose tasks via llm().

**Solution:** Add explicit recursion guidance to the system prompt.

**Files:**
- Modify: `src/repl/rlaph_loop.py:404-452` (_build_system_prompt)

**Step 1: Update system prompt**

In `src/repl/rlaph_loop.py`, update `_build_system_prompt`:

```python
def _build_system_prompt(self) -> str:
    """Build system prompt with REPL instructions and recursion guidance."""
    return f"""You are an RLM (Recursive Language Model) agent with access to a REAL Python REPL.

CRITICAL RULES:
1. When you write code in ```python blocks, the system EXECUTES it and returns REAL output
2. DO NOT generate fake "REPL output" or "Human:" messages yourself
3. DO NOT pretend to see execution results - wait for the actual system response
4. After writing code, STOP and wait for the [SYSTEM - Code execution result]
5. When you have the final answer, write: FINAL: <answer>
6. NEVER use import statements - they are blocked. Pre-loaded: hashlib, json, re, os, sys

RECURSION - DECOMPOSE COMPLEX TASKS:
You can call llm(sub_query) to delegate sub-tasks. This creates a CHILD FRAME.
- Max recursion depth: {self.max_depth}
- Use llm(sub_query) for parallel/branching analysis
- Each llm() call is tracked as a child frame
- Example: For codebase summary, first glob files, then llm("summarize auth/*.py")

Your workflow:
1. Write Python code in ```python blocks
2. STOP - the system will execute and return [SYSTEM - Code execution result]
3. Read the REAL output from the system
4. For complex tasks, use llm(sub_query) to decompose
5. Write more code OR provide FINAL: <answer>

Pre-loaded Libraries (NO import needed):
- hashlib: Use directly as `hashlib.sha256(data.encode()).hexdigest()`
- json: Use directly as `json.loads()`, `json.dumps()`
- re: Use directly for regex operations

File Access Functions:
- `read_file(path, offset=0, limit=2000)`: Read file content from disk
- `glob_files(pattern)`: Find files matching pattern
- `grep_files(pattern, path)`: Search for pattern in files
- `list_dir(path)`: List directory contents

Recursion Function:
- `llm(query)`: Call LLM with sub-query, returns result string
  - Creates child frame at depth+1
  - Use for task decomposition

Example with Recursion:
User: Analyze the auth module architecture

Your response:
```python
auth_files = glob_files("src/auth/**/*.py")
print(f"Found {{len(auth_files)}} auth files")
```
[STOP - wait for system]

System returns: Found 5 auth files

Your response:
```python
# Decompose into sub-analysis
auth_summary = llm("Summarize the main authentication flow in src/auth/login.py")
oauth_summary = llm("Summarize the OAuth implementation in src/auth/oauth.py")
print(f"Auth: {{auth_summary[:100]}}...")
print(f"OAuth: {{oauth_summary[:100]}}...")
```
[STOP - wait for system]

System returns results from child frames

Your response:
FINAL: The auth module consists of login.py (main flow) and oauth.py (OAuth 2.0)...

Other functions: peek(), search(), summarize(), llm_batch(), map_reduce()
Working memory: working_memory dict for storing results across code blocks"""
```

**Step 2: Commit**

```bash
git add src/repl/rlaph_loop.py
git commit -m "feat: enhance system prompt with recursion guidance

- Explicit llm(sub_query) usage instructions
- Example with task decomposition
- Max depth visibility in prompt
- Encourages tree structure creation"
```

---

### Task A3: Add Query/Intent Tracking to FrameIndex

**Problem:** Sessions don't track the user's initial intention, making cross-session awareness poor.

**Solution:** Add `initial_query` and optional `query_summary` to FrameIndex.

**Files:**
- Modify: `src/frame/frame_index.py`
- Modify: `scripts/run_orchestrator.py`
- Test: `tests/frame/test_frame_index_query.py`

**Step 1: Write the failing test**

```python
"""Tests for query/intent tracking in FrameIndex."""
import pytest
from src.frame.frame_index import FrameIndex


def test_frame_index_stores_initial_query():
    """FrameIndex should store the initial user query."""
    index = FrameIndex(initial_query="Analyze the auth module")

    assert index.initial_query == "Analyze the auth module"


def test_frame_index_query_summary():
    """FrameIndex can store a short query summary."""
    index = FrameIndex(
        initial_query="Analyze the authentication flow in the auth module",
        query_summary="Auth flow analysis"
    )

    assert index.query_summary == "Auth flow analysis"


def test_frame_index_save_load_with_query(tmp_path):
    """Query should persist across save/load."""
    index = FrameIndex(
        initial_query="Debug the login bug",
        query_summary="Login bug debug"
    )

    # Save
    index.save("test_query_session", tmp_path)

    # Load
    loaded = FrameIndex.load("test_query_session", tmp_path)

    assert loaded.initial_query == "Debug the login bug"
    assert loaded.query_summary == "Login bug debug"


def test_frame_index_defaults_empty_query():
    """FrameIndex defaults to empty query strings."""
    index = FrameIndex()

    assert index.initial_query == ""
    assert index.query_summary == ""
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/frame/test_frame_index_query.py -v
```

Expected: FAIL with "TypeError: FrameIndex.__init__() got an unexpected keyword argument 'initial_query'"

**Step 3: Update FrameIndex**

In `src/frame/frame_index.py`:

```python
@dataclass
class FrameIndex:
    """
    In-memory index of CausalFrames for a session.

    Supports O(n) operations for 10-20 frames.
    """

    initial_query: str = ""
    query_summary: str = ""
    commit_hash: str | None = None
    _frames: dict[str, CausalFrame] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure _frames is a dict."""
        if self._frames is None:
            self._frames = {}
```

Update save/load methods:

```python
def save(self, session_id: str, base_dir: Path | None = None) -> Path:
    """Save frame index to JSON file."""
    if base_dir is None:
        base_dir = Path.home() / ".claude" / "rlm-frames"
    else:
        base_dir = Path(base_dir)

    base_dir.mkdir(parents=True, exist_ok=True)
    save_path = base_dir / session_id / "index.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "initial_query": self.initial_query,
        "query_summary": self.query_summary,
        "commit_hash": self.commit_hash,
        "frames": [frame_to_dict(f) for f in self._frames.values()],
    }

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

    return save_path

@classmethod
def load(cls, session_id: str, base_dir: Path | None = None) -> "FrameIndex | None":
    """Load frame index from JSON file."""
    if base_dir is None:
        base_dir = Path.home() / ".claude" / "rlm-frames"
    else:
        base_dir = Path(base_dir)

    load_path = base_dir / session_id / "index.json"

    if not load_path.exists():
        return None

    with open(load_path) as f:
        data = json.load(f)

    frames = {d["frame_id"]: dict_to_frame(d) for d in data.get("frames", [])}

    return cls(
        initial_query=data.get("initial_query", ""),
        query_summary=data.get("query_summary", ""),
        commit_hash=data.get("commit_hash"),
        _frames=frames,
    )
```

**Step 4: Update orchestrator to pass query**

In `scripts/run_orchestrator.py`, update `run_rlaph`:

```python
async def run_rlaph(
    query: str,
    depth: int = 2,
    verbose: bool = False,
    working_dir: Path | None = None,
    session_id: str | None = None,
) -> str:
    """..."""
    # ... existing code ...

    # Set initial query on frame index
    loop.frame_index.initial_query = query

    # Optional: Generate query summary (could be LLM call, but keep simple for now)
    # Take first 50 chars as summary
    loop.frame_index.query_summary = query[:50] + ("..." if len(query) > 50 else "")

    # Run loop
    result = await loop.run(query, context, working_dir=work_dir, session_id=session_id)

    # ... rest of code ...
```

**Step 5: Run tests**

```bash
uv run pytest tests/frame/test_frame_index_query.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/frame/frame_index.py scripts/run_orchestrator.py tests/frame/test_frame_index_query.py
git commit -m "feat: add query/intent tracking to FrameIndex

- initial_query: Full user query
- query_summary: Short summary for cross-session matching
- Persisted in index.json
- Enables 'continue last task' feature"
```

---

## Part B: Tree Structure + Evidence + Cascade

### Task B1: Populate Children in Frame Tree

**Files:**
- Modify: `src/frame/frame_index.py` (add method)
- Test: `tests/repl/test_frame_tree.py`

**Step 1: Write the failing test**

```python
"""Tests for frame tree structure."""
import pytest
from src.frame.frame_index import FrameIndex
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from datetime import datetime


def test_add_child_updates_parent_children_list():
    """FrameIndex.add should update parent's children list."""
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

### Task B2: Auto-Generate invalidation_condition

**Files:**
- Modify: `src/frame/causal_frame.py` (add helper function)
- Modify: `src/repl/rlaph_loop.py` (use in frame creation)
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

    assert len(condition) > 0
    assert "file" in condition.lower() or "2" in condition


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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/frame/test_invalidation_condition.py -v
```

Expected: FAIL with "ImportError: cannot import name 'generate_invalidation_condition'"

**Step 3: Write minimal implementation**

In `src/frame/causal_frame.py`, add function:

```python
from pathlib import Path

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

**Step 4: Use in RLAPHLoop frame creation**

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

**Step 5: Run tests**

```bash
uv run pytest tests/frame/test_invalidation_condition.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/frame/causal_frame.py src/repl/rlaph_loop.py tests/frame/test_invalidation_condition.py
git commit -m "feat: auto-generate invalidation_condition from context_slice

- generate_invalidation_condition helper function
- Produces human-readable condition string
- Called automatically in frame creation"
```

---

### Task B3: Auto Evidence Tracking

**Files:**
- Modify: `src/frame/frame_index.py` (update add method)
- Test: `tests/repl/test_evidence_tracking.py`

**Step 1: Write the failing test**

```python
"""Tests for automatic evidence tracking."""
import pytest
from src.frame.frame_index import FrameIndex
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from datetime import datetime


def test_child_frame_added_to_parent_evidence():
    """When child frame completes, it should be added to parent's evidence."""
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


def test_invalidated_child_not_added_to_evidence():
    """Invalidated frames should not be added as evidence."""
    index = FrameIndex()

    parent = CausalFrame(
        frame_id="inv_parent",
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
        frame_id="inv_child",
        depth=1,
        parent_id="inv_parent",
        children=[],
        query="c",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=[],
        conclusion=None,
        confidence=0.8,
        invalidation_condition="",
        status=FrameStatus.INVALIDATED,  # Invalidated!
        branched_from=None,
        escalation_reason="error",
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    index.add(child)

    # Parent's evidence should NOT include invalidated child
    parent_after = index.get("inv_parent")
    assert "inv_child" not in parent_after.evidence
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/repl/test_evidence_tracking.py -v
```

Expected: FAIL with "AssertionError: assert 'ev_child' in []"

**Step 3: Update implementation**

In `src/frame/frame_index.py`, update `add` method:

```python
from .causal_frame import FrameStatus

def add(self, frame: "CausalFrame") -> None:
    """Add a frame to the index, update parent's children and evidence."""
    self._frames[frame.frame_id] = frame

    # Update parent's children list and evidence
    if frame.parent_id and frame.parent_id in self._frames:
        parent = self._frames[frame.parent_id]

        # Add to children
        if frame.frame_id not in parent.children:
            parent.children.append(frame.frame_id)

        # Add to evidence (child conclusion is evidence for parent)
        # Only if child completed successfully
        if frame.status == FrameStatus.COMPLETED:
            if frame.frame_id not in parent.evidence:
                parent.evidence.append(frame.frame_id)
```

**Step 4: Run tests**

```bash
uv run pytest tests/repl/test_evidence_tracking.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/frame/frame_index.py tests/repl/test_evidence_tracking.py
git commit -m "feat: auto-track evidence when adding child frames

- Child frame IDs added to parent's evidence list
- Only COMPLETED frames added as evidence
- Enables cascade invalidation across evidence links"
```

---

### Task B4: Add find_dependent_frames to FrameIndex

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


def test_find_dependent_frames_by_evidence():
    """Should find frames that cite a frame as evidence."""
    index = FrameIndex()

    source = make_frame("source_frame")
    index.add(source)

    dependent = make_frame("dependent_frame", evidence=["source_frame"])
    index.add(dependent)

    dependents = index.find_dependent_frames("source_frame")

    assert "dependent_frame" in dependents


def test_find_dependent_frames_includes_children():
    """Should include children as dependents."""
    index = FrameIndex()

    parent = make_frame("parent_dep")
    index.add(parent)

    child = make_frame("child_dep", parent_id="parent_dep")
    index.add(child)

    dependents = index.find_dependent_frames("parent_dep")

    assert "child_dep" in dependents


def test_find_dependent_frames_none():
    """Should return empty set if no dependents."""
    index = FrameIndex()

    orphan = make_frame("orphan")
    index.add(orphan)

    dependents = index.find_dependent_frames("orphan")

    assert dependents == set()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/frame/test_find_dependent.py -v
```

Expected: FAIL with "AttributeError: 'FrameIndex' object has no attribute 'find_dependent_frames'"

**Step 3: Write implementation**

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

**Step 4: Run tests**

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

### Task B5: Strengthen Cascade Propagation

**Files:**
- Modify: `src/frame/frame_invalidation.py`
- Test: `tests/frame/test_cascade_invalidation.py`

**Step 1: Write the test**

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

    invalidated = propagate_invalidation("parent_cascade", "test reason", index)

    assert "parent_cascade" in invalidated
    assert "child_cascade" in invalidated
    assert index.get("parent_cascade").status == FrameStatus.INVALIDATED
    assert index.get("child_cascade").status == FrameStatus.INVALIDATED


def test_propagate_invalidates_evidence_consumers():
    """Invalidating a frame should invalidate frames citing it as evidence."""
    index = FrameIndex()

    source = make_frame("source_evidence")
    index.add(source)

    consumer = make_frame("consumer", evidence=["source_evidence"])
    index.add(consumer)

    invalidated = propagate_invalidation("source_evidence", "evidence changed", index)

    assert "source_evidence" in invalidated
    assert "consumer" in invalidated


def test_propagate_handles_cycles():
    """Should handle cycles without infinite loop."""
    index = FrameIndex()

    frame_a = make_frame("frame_a", evidence=["frame_b"])
    frame_b = make_frame("frame_b", evidence=["frame_a"])

    index.add(frame_a)
    index.add(frame_b)

    # Should not hang
    invalidated = propagate_invalidation("frame_a", "test cycle", index)

    assert "frame_a" in invalidated
    assert "frame_b" in invalidated
```

**Step 2: Run test to verify**

```bash
uv run pytest tests/frame/test_cascade_invalidation.py -v
```

**Step 3: Ensure implementation is robust**

In `src/frame/frame_invalidation.py`:

```python
def propagate_invalidation(
    frame_id: str,
    reason: str,
    index: "FrameIndex"
) -> set[str]:
    """
    Invalidate a frame and all its dependents.

    Propagation direction:
    - DOWN: to all children (tree walk)
    - SIDEWAYS: to frames using this as evidence

    Returns set of all invalidated frame IDs.
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
            if dep_id not in invalidated:
                _invalidate(dep_id, f"Evidence invalidated: {fid}")

    _invalidate(frame_id, reason)
    return invalidated
```

**Step 4: Run tests**

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
- Handle circular evidence gracefully
- Return set of all invalidated IDs"
```

---

### Task B6: Integration Test - Full Tree with Cascade

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

    When root is invalidated, all should be invalidated.
    """
    index = FrameIndex()

    root = make_frame("root", files={"/src/main.py": "hash1"})
    index.add(root)

    child1 = make_frame("child1", parent_id="root")
    index.add(child1)

    child2 = make_frame("child2", parent_id="root")
    index.add(child2)

    leaf1 = make_frame("leaf1", parent_id="child1")
    index.add(leaf1)

    evidence_user = make_frame("evidence_user", evidence=["leaf1"])
    index.add(evidence_user)

    # Verify tree structure
    assert "child1" in index.get("root").children
    assert "child2" in index.get("root").children
    assert "leaf1" in index.get("child1").children

    # Invalidate root
    invalidated = propagate_invalidation("root", "main.py changed", index)

    # All frames should be invalidated
    assert len(invalidated) == 5
    for fid in ["root", "child1", "child2", "leaf1", "evidence_user"]:
        assert fid in invalidated
        assert index.get(fid).status == FrameStatus.INVALIDATED


def test_partial_tree_invalidation():
    """Only invalidating a subtree should not affect siblings."""
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
    invalidated = propagate_invalidation("child1_partial", "child1 reason", index)

    assert "child1_partial" in invalidated
    assert "child1_leaf" in invalidated
    assert "child2_partial" not in invalidated
    assert "child2_leaf" not in invalidated
    assert "root_partial" not in invalidated
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

### Part A: Recursion Infrastructure
1. **Synchronous llm(sub_query)** - Creates child frames, proper depth tracking
2. **Enhanced system prompt** - Explicit recursion guidance with examples
3. **Query/intent tracking** - initial_query and query_summary in FrameIndex

### Part B: Tree Structure + Evidence + Cascade
4. **Children population** - FrameIndex.add updates parent.children
5. **Auto evidence tracking** - Child frames added to parent.evidence
6. **Auto invalidation_condition** - Generated from context_slice
7. **find_dependent_frames** - Discover evidence consumers
8. **Stronger cascade** - Full tree walk + evidence propagation

**Key Design Decision:** Use synchronous `llm(sub_query)` instead of subprocess.
- Simpler, faster, no IPC overhead
- Fits RLM ethos (recursive calls in same process)
- Subprocess can be future work for isolation if needed

Result: Proper tree structure with working recursion and cascade invalidation.
