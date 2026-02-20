# Phase 18: Tree Structure + Evidence + Cascade + Recursion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Summary:** Enable true recursive decomposition, tree-structured frames, evidence linking, and full cascade invalidation — turning flat chains into evolving causal graphs.

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

**Solution:** Use synchronous `llm(sub_query)` that creates child frames with **explicit depth parameter** (safer for future async/parallel).

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


@pytest.mark.asyncio
async def test_llm_sync_returns_result():
    """llm_sync should return actual LLM result synchronously."""
    loop = RLAPHLoop(max_depth=2)

    # This should work without error
    result = loop.llm_sync("What is 2+2?")

    assert result is not None
    assert len(result) > 0


def test_llm_sync_with_explicit_depth():
    """llm_sync should accept explicit depth parameter."""
    loop = RLAPHLoop(max_depth=3)

    # Call with explicit depth
    result = loop.llm_sync("Test query", depth=1)

    assert result is not None

    # Check that child frame was created at depth 1
    frames_at_depth_1 = [f for f in loop.frame_index._frames.values() if f.depth == 1]
    assert len(frames_at_depth_1) >= 1


def test_llm_sync_respects_max_depth():
    """llm_sync should raise RecursionDepthError if max depth exceeded."""
    loop = RLAPHLoop(max_depth=2)

    from src.types import RecursionDepthError
    with pytest.raises(RecursionDepthError):
        loop.llm_sync("This should fail", depth=3)  # Exceeds max_depth=2


def test_llm_sync_depth_default_increments():
    """Without explicit depth, llm_sync should use current_depth + 1."""
    loop = RLAPHLoop(max_depth=3)
    loop._current_frame_id = None  # Root level

    # First call at implicit depth 1
    result1 = loop.llm_sync("Query 1")
    frames_after_1 = len(loop.frame_index)

    # Second call should also work
    result2 = loop.llm_sync("Query 2")
    frames_after_2 = len(loop.frame_index)

    assert frames_after_2 > frames_after_1
```

**Step 2: Run test to verify current state**

```bash
cd /Users/gilpark/.dotfiles/.claude/plugins/marketplaces/causeway
uv run pytest tests/repl/test_llm_recursion.py -v
```

Expected: Tests fail (llm_sync doesn't create frames yet)

**Step 3: Implement llm_sync with explicit depth parameter**

In `src/repl/rlaph_loop.py`, update `llm_sync`:

```python
def llm_sync(self, query: str, context: str = "", depth: int | None = None) -> str:
    """
    Synchronous LLM call - returns actual result immediately.

    This is the key method that makes RLAPH work:
    - Called from REPL's llm() function
    - Returns actual string result
    - Creates child frames for recursion tracking
    - Uses explicit depth parameter (safer for future async)

    Args:
        query: Query string
        context: Optional context string
        depth: Explicit depth for this call (default: self._depth + 1)

    Returns:
        LLM response as string

    Raises:
        RecursionDepthError: If max depth exceeded
    """
    # Calculate depth explicitly (don't mutate self._depth)
    current_depth = depth if depth is not None else self._depth + 1

    # Check depth limit
    if current_depth > self.max_depth:
        raise RecursionDepthError(current_depth, self.max_depth)

    # Verbose logging for recursion decisions
    if self._verbose:
        print(f"[RLM] Recursion at depth {current_depth}: {query[:80]}...")

    # Use LLMClient directly
    result = self.llm_client.call(
        query=query,
        context={"prior": context} if context else None,
        depth=current_depth,
    )

    # Track tokens
    self._tokens_used += len(result) // 4

    # Create child frame for this llm call
    context_slice = ContextSlice(
        files={},
        memory_refs=list(self.repl.memory_refs) if self.repl else [],
        tool_outputs={},
        token_budget=self.config.cost_controls.max_tokens_per_recursive_call,
    )

    child_frame = CausalFrame(
        frame_id=compute_frame_id(
            self._current_frame_id,
            query[:100],
            context_slice,
        ),
        depth=current_depth,
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

    # Optionally track current frame (for nested calls)
    # Don't update self._current_frame_id to avoid affecting sibling calls
    # self._current_frame_id = child_frame.frame_id

    return result
```

Add `verbose` parameter to `__init__`:

```python
def __init__(
    self,
    max_iterations: int = 20,
    max_depth: int = 3,
    config: RLMConfig | None = None,
    llm_client: LLMClient | None = None,
    context_map: ContextMap | None = None,
    verbose: bool = False,  # NEW
):
    # ... existing init ...
    self._verbose = verbose
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
- Explicit depth parameter (safer for future async)
- Verbose logging for recursion decisions
- Enables tree structure instead of flat chain"
```

---

### Task A2: Enhance System Prompt for Recursion

**Problem:** The LLM doesn't know it can decompose tasks via llm(), or might over-use recursion.

**Solution:** Add explicit recursion guidance with guardrails against depth explosion.

**Files:**
- Modify: `src/repl/rlaph_loop.py` (_build_system_prompt)

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

IMPORTANT RECURSION RULES:
- Only call llm(sub_query) when the sub-task is meaningfully independent or parallelizable
- Do NOT call llm() for tiny steps — that wastes depth budget
- Always prefer small, focused sub-queries (1–3 sentences)
- If you're unsure, try simple code first, then recurse if needed

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
  - Keep sub-queries focused (1-3 sentences)

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
# Decompose into parallel sub-analyses (each is independent)
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
git commit -m "feat: enhance system prompt with recursion guidance and guardrails

- Explicit llm(sub_query) usage instructions
- Guardrails against depth explosion (no tiny steps)
- Example with task decomposition
- Max depth visibility in prompt"
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
from pathlib import Path
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
    _dependent_cache: dict[str, set[str]] = field(default_factory=dict)  # For B4

    def __post_init__(self):
        """Ensure _frames is a dict."""
        if self._frames is None:
            self._frames = {}
        if self._dependent_cache is None:
            self._dependent_cache = {}
```

Update save/load methods to include query fields:

```python
def save(self, session_id: str, base_dir: Path | None = None) -> Path:
    """Save frame index to JSON file."""
    # ... existing path logic ...

    data = {
        "initial_query": self.initial_query,
        "query_summary": self.query_summary,
        "commit_hash": self.commit_hash,
        "frames": [frame_to_dict(f) for f in self._frames.values()],
    }

    # ... rest of save ...

@classmethod
def load(cls, session_id: str, base_dir: Path | None = None) -> "FrameIndex | None":
    """Load frame index from JSON file."""
    # ... existing load logic ...

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
    # ... existing setup ...

    # Set initial query on frame index
    loop.frame_index.initial_query = query

    # Generate simple query summary (first 50 chars)
    loop.frame_index.query_summary = query[:50] + ("..." if len(query) > 50 else "")

    # Run loop
    result = await loop.run(query, context, working_dir=work_dir, session_id=session_id)

    # ... rest of function ...
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
        invalidation_condition="",
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/repl/test_frame_tree.py -v
```

Expected: FAIL with "AssertionError: assert 'child111' in []"

**Step 3: Write implementation with defensive checks**

In `src/frame/frame_index.py`, modify `add` method:

```python
def add(self, frame: "CausalFrame") -> None:
    """Add a frame to the index and update parent's children list."""
    self._frames[frame.frame_id] = frame

    # Invalidate dependent cache when frames change
    self._dependent_cache.clear()

    # Update parent's children list (defensive)
    if frame.parent_id and frame.parent_id in self._frames:
        parent = self._frames[frame.parent_id]
        if frame.frame_id not in parent.children:
            parent.children.append(frame.frame_id)
        # else: already added (idempotent, no-op)
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
- Defensive check for duplicates (idempotent)
- Clear dependent cache on frame changes
- Enables proper tree structure traversal"
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
    assert "file" in condition.lower()


def test_generate_invalidation_condition_single_file():
    """Should show filename for single file."""
    context_slice = ContextSlice(
        files={"/src/main.py": "abc123"},
        memory_refs=[],
        tool_outputs={},
        token_budget=8000,
    )

    condition = generate_invalidation_condition(context_slice)

    assert "main.py" in condition
    assert "deleted" in condition or "change" in condition


def test_generate_invalidation_condition_empty():
    """Should return default message for empty context_slice."""
    context_slice = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={},
        token_budget=8000,
    )

    condition = generate_invalidation_condition(context_slice)

    # Should have some default message
    assert isinstance(condition, str)


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

**Step 3: Write implementation with precise conditions**

In `src/frame/causal_frame.py`, add function:

```python
from pathlib import Path

def generate_invalidation_condition(context_slice: "ContextSlice") -> str:
    """
    Generate default invalidation condition from context_slice.

    The condition describes what would make this frame's conclusion
    no longer valid. More precise = easier debugging.

    Args:
        context_slice: The frame's context slice

    Returns:
        Human-readable invalidation condition string
    """
    parts = []

    if context_slice.files:
        # Show actual filenames for clarity
        filenames = [Path(p).name for p in context_slice.files.keys()]
        if len(filenames) == 1:
            parts.append(f"{filenames[0]} changes or is deleted")
        else:
            # Show first 3 files, indicate if more
            shown = filenames[:3]
            more = f" (+{len(filenames) - 3} more)" if len(filenames) > 3 else ""
            parts.append(f"any of {len(filenames)} files ({', '.join(shown)}{more}) change")

    if context_slice.tool_outputs:
        tool_names = list(context_slice.tool_outputs.keys())
        parts.append(f"tool results from {', '.join(tool_names)} change")

    if context_slice.memory_refs:
        parts.append(f"memory entries change")

    if not parts:
        return "No automatic invalidation condition"

    return "; or ".join(parts)
```

**Step 4: Use in RLAPHLoop frame creation**

In `src/repl/rlaph_loop.py`, update frame creation (both in `run()` and `llm_sync()`):

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
git commit -m "feat: auto-generate precise invalidation_condition from context_slice

- generate_invalidation_condition helper function
- Shows actual filenames for clarity
- Handles tool outputs and memory refs
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


def make_frame(frame_id: str, parent_id: str = None, status: FrameStatus = FrameStatus.COMPLETED) -> CausalFrame:
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
        invalidation_condition="",
        status=status,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )


def test_child_frame_added_to_parent_evidence():
    """When child frame completes, it should be added to parent's evidence."""
    index = FrameIndex()

    parent = make_frame("ev_parent")
    index.add(parent)

    child = make_frame("ev_child", parent_id="ev_parent")
    index.add(child)

    # Parent's evidence should include child
    parent_after = index.get("ev_parent")
    assert "ev_child" in parent_after.evidence


def test_invalidated_child_not_added_to_evidence():
    """Invalidated frames should not be added as evidence."""
    index = FrameIndex()

    parent = make_frame("inv_parent")
    index.add(parent)

    child = make_frame("inv_child", parent_id="inv_parent", status=FrameStatus.INVALIDATED)
    index.add(child)

    # Parent's evidence should NOT include invalidated child
    parent_after = index.get("inv_parent")
    assert "inv_child" not in parent_after.evidence


def test_evidence_tracking_idempotent():
    """Adding child twice shouldn't duplicate evidence."""
    index = FrameIndex()

    parent = make_frame("ev_dup_parent")
    index.add(parent)

    child = make_frame("ev_dup_child", parent_id="ev_dup_parent")
    index.add(child)
    index.add(child)  # Add again

    parent_after = index.get("ev_dup_parent")
    assert parent_after.evidence.count("ev_dup_child") == 1
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

    # Invalidate dependent cache
    self._dependent_cache.clear()

    # Update parent's children list and evidence
    if frame.parent_id and frame.parent_id in self._frames:
        parent = self._frames[frame.parent_id]

        # Add to children (defensive)
        if frame.frame_id not in parent.children:
            parent.children.append(frame.frame_id)

        # Add to evidence only if child completed successfully
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
- Defensive duplicate check
- Enables cascade invalidation across evidence links"
```

---

### Task B4: Add find_dependent_frames with Caching

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


def test_find_dependent_frames_caching():
    """Should cache results and return copy."""
    index = FrameIndex()

    source = make_frame("cached_source")
    index.add(source)

    dependent = make_frame("cached_dep", evidence=["cached_source"])
    index.add(dependent)

    # First call
    dependents1 = index.find_dependent_frames("cached_source")
    assert "cached_dep" in dependents1

    # Second call should use cache
    dependents2 = index.find_dependent_frames("cached_source")
    assert dependents2 == dependents1

    # Modifying returned set shouldn't affect cache
    dependents1.add("fake")
    dependents3 = index.find_dependent_frames("cached_source")
    assert "fake" not in dependents3
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/frame/test_find_dependent.py -v
```

Expected: FAIL with "AttributeError: 'FrameIndex' object has no attribute 'find_dependent_frames'"

**Step 3: Write implementation with caching**

In `src/frame/frame_index.py`, add method:

```python
def find_dependent_frames(self, frame_id: str) -> set[str]:
    """
    Find all frames that depend on a given frame.

    Dependents include:
    - Children (frames with this frame as parent_id)
    - Evidence consumers (frames citing this frame in evidence list)

    Results are cached for O(1) lookup until frames change.

    Args:
        frame_id: Frame to find dependents for

    Returns:
        Set of frame IDs that depend on this frame (copy)
    """
    # Check cache
    if frame_id in self._dependent_cache:
        return self._dependent_cache[frame_id].copy()

    dependents = set()

    for fid, frame in self._frames.items():
        # Check if child
        if frame.parent_id == frame_id:
            dependents.add(fid)

        # Check if cites as evidence
        if frame_id in frame.evidence:
            dependents.add(fid)

    # Cache result
    self._dependent_cache[frame_id] = dependents

    return dependents.copy()
```

**Step 4: Run tests**

```bash
uv run pytest tests/frame/test_find_dependent.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/frame/frame_index.py tests/frame/test_find_dependent.py
git commit -m "feat: add find_dependent_frames with caching for cascade invalidation

- Find frames that cite a frame as evidence
- Include children as dependents
- O(n) scan with caching for repeated lookups
- Returns copy to prevent cache corruption"
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


def test_propagate_records_escalation_reason():
    """Invalidated frames should have escalation_reason set."""
    index = FrameIndex()

    parent = make_frame("parent_reason")
    index.add(parent)

    child = make_frame("child_reason", parent_id="parent_reason")
    index.add(child)

    propagate_invalidation("parent_reason", "original reason", index)

    assert "original reason" in index.get("parent_reason").escalation_reason
    assert "original reason" in index.get("child_reason").escalation_reason
```

**Step 2: Run test to verify**

```bash
uv run pytest tests/frame/test_cascade_invalidation.py -v
```

**Step 3: Ensure implementation is robust**

In `src/frame/frame_invalidation.py`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frame_index import FrameIndex


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
        # Cycle detection
        if fid in invalidated:
            return
        invalidated.add(fid)

        frame = index.get(fid)
        if frame is None:
            return

        # Update status and reason
        frame.status = FrameStatus.INVALIDATED
        frame.escalation_reason = current_reason

        # CASCADE DOWN to children
        for child_id in frame.children:
            _invalidate(child_id, f"Parent invalidated: {reason}")

        # CASCADE SIDEWAYS to evidence consumers
        dependents = index.find_dependent_frames(fid)
        for dep_id in dependents:
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

### Task B6: End-to-End Integration Test

**Files:**
- Create: `tests/integration/test_tree_cascade_integration.py`

**Step 1: Write integration test**

```python
"""End-to-end integration test for tree + evidence + cascade + recursion."""
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
    files: dict = None,
    depth: int = 0
) -> CausalFrame:
    """Helper to create test frames."""
    return CausalFrame(
        frame_id=frame_id,
        depth=depth,
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
    E2E: Full tree with children and evidence links.

    Tree structure:
              root (depth 0)
             /    \
          child1  child2 (depth 1)
            |       |
          leaf1  evidence_user (cites leaf1)

    When root is invalidated, all should be invalidated.
    """
    index = FrameIndex()

    # Build 3-level tree
    root = make_frame("root", files={"/src/main.py": "hash1"}, depth=0)
    index.add(root)

    child1 = make_frame("child1", parent_id="root", depth=1)
    index.add(child1)

    child2 = make_frame("child2", parent_id="root", depth=1)
    index.add(child2)

    leaf1 = make_frame("leaf1", parent_id="child1", depth=2)
    index.add(leaf1)

    evidence_user = make_frame("evidence_user", evidence=["leaf1"], depth=1)
    index.add(evidence_user)

    # Verify tree structure was built
    assert "child1" in index.get("root").children
    assert "child2" in index.get("root").children
    assert "leaf1" in index.get("child1").children
    assert "leaf1" in index.get("child1").evidence

    # Invalidate root
    invalidated = propagate_invalidation("root", "main.py changed", index)

    # All frames should be invalidated
    assert len(invalidated) == 5
    for fid in ["root", "child1", "child2", "leaf1", "evidence_user"]:
        assert fid in invalidated
        assert index.get(fid).status == FrameStatus.INVALIDATED

    # Verify escalation reasons propagate
    assert "main.py changed" in index.get("root").escalation_reason
    assert "leaf1" in index.get("evidence_user").escalation_reason


def test_partial_tree_invalidation():
    """E2E: Only invalidating a subtree should not affect siblings."""
    index = FrameIndex()

    root = make_frame("root_partial", depth=0)
    index.add(root)

    child1 = make_frame("child1_partial", parent_id="root_partial", depth=1)
    index.add(child1)

    child1_leaf = make_frame("child1_leaf", parent_id="child1_partial", depth=2)
    index.add(child1_leaf)

    child2 = make_frame("child2_partial", parent_id="root_partial", depth=1)
    index.add(child2)

    child2_leaf = make_frame("child2_leaf", parent_id="child2_partial", depth=2)
    index.add(child2_leaf)

    # Invalidate only child1
    invalidated = propagate_invalidation("child1_partial", "child1 reason", index)

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


def test_query_tracking_persists():
    """E2E: Query tracking should persist through save/load."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create index with query
        index = FrameIndex(
            initial_query="Analyze the auth module deeply",
            query_summary="Auth analysis"
        )

        # Add some frames
        root = make_frame("qroot", depth=0)
        index.add(root)

        # Save
        index.save("test_query", tmp_path)

        # Load
        loaded = FrameIndex.load("test_query", tmp_path)

        assert loaded.initial_query == "Analyze the auth module deeply"
        assert loaded.query_summary == "Auth analysis"
        assert "qroot" in loaded._frames
```

**Step 2: Run integration test**

```bash
uv run pytest tests/integration/test_tree_cascade_integration.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_tree_cascade_integration.py
git commit -m "test: add E2E integration test for tree + evidence + cascade

- Full tree cascade scenario (3 levels)
- Partial subtree invalidation
- Evidence propagation across branches
- Query tracking persistence"
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
1. **Synchronous llm(sub_query)** - Creates child frames with explicit depth parameter
2. **Enhanced system prompt** - Recursion guidance with guardrails against depth explosion
3. **Query/intent tracking** - initial_query and query_summary in FrameIndex

### Part B: Tree Structure + Evidence + Cascade
4. **Children population** - FrameIndex.add updates parent.children (defensive)
5. **Auto invalidation_condition** - Precise conditions with filenames
6. **Auto evidence tracking** - Only COMPLETED frames as evidence
7. **find_dependent_frames** - With caching for O(1) lookups
8. **Stronger cascade** - Full tree walk + evidence propagation
9. **E2E integration test** - Validates everything connects

### Key Design Decisions
- **Explicit depth parameter** in llm_sync (safer for future async)
- **Defensive checks** in add() (idempotent, no duplicates)
- **Caching** in find_dependent_frames
- **Verbose logging** flag for debugging recursion decisions

Result: Proper tree structure with working recursion and cascade invalidation.
