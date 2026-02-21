# Phase 18: Tree Structure + Evidence + Cascade + Recursion + Intent Normalization

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Summary:** Enable true recursive decomposition, tree-structured frames, evidence linking, full cascade invalidation, **and intent-based frame deduplication** — turning flat chains into an evolving, reusable causal graph.

**Goal:**
- Fix recursion (depth > 0, real branching)
- Shift frame identity from raw query → canonical task (prevents duplication)
- Make invalidation_condition structured & executable
- Auto-track evidence + children correctly
- Strengthen cascade with caching & defensiveness
- Add verbose logging for debugging recursion decisions

**Architecture Decisions (from conversation):**
- Frame identity = hash(canonical_task + context_slice.hash) — **not** raw query
- Evidence only from COMPLETED children (no invalidated)
- invalidation_condition becomes dict (structured, not just string)
- No weighted evidence yet — prefer symbol-level granularity later
- Synchronous llm(sub_query) — no subprocess
- Verbose logging for recursion decisions

**Tech Stack:** Python dataclasses, existing frame infrastructure, json, hashlib

---

## Part A: Recursion Infrastructure + Intent Normalization

### Task A1: Add Synchronous `llm(sub_query)` Recursion with Explicit Depth

**Files:**
- Modify: `src/repl/rlaph_loop.py`
- Modify: `src/repl/repl_environment.py`
- Test: `tests/repl/test_llm_recursion.py`

**Step 1: Write failing tests**

```python
"""Tests for llm() recursive calls."""
import pytest
from pathlib import Path
from src.repl.rlaph_loop import RLAPHLoop
from src.frame.causal_frame import FrameStatus
from src.types import SessionContext


def test_llm_sync_returns_result():
    """llm_sync should return actual LLM result synchronously."""
    loop = RLAPHLoop(max_depth=2)
    result = loop.llm_sync("What is 2+2?")
    assert result is not None
    assert len(result) > 0


def test_llm_sync_with_explicit_depth():
    """llm_sync should accept explicit depth parameter."""
    loop = RLAPHLoop(max_depth=3)
    result = loop.llm_sync("Test query", depth=1)
    assert result is not None
    frames_at_depth_1 = [f for f in loop.frame_index._frames.values() if f.depth == 1]
    assert len(frames_at_depth_1) >= 1


def test_llm_sync_respects_max_depth():
    """llm_sync should raise RecursionDepthError if max depth exceeded."""
    loop = RLAPHLoop(max_depth=2)
    from src.types import RecursionDepthError
    with pytest.raises(RecursionDepthError):
        loop.llm_sync("This should fail", depth=3)


def test_llm_sync_creates_child_frame():
    """llm_sync should create a child frame for each call."""
    loop = RLAPHLoop(max_depth=3)
    initial_count = len(loop.frame_index)
    loop.llm_sync("Test query", depth=1)
    assert len(loop.frame_index) > initial_count


def test_llm_sync_verbose_logging(capsys):
    """llm_sync should log when verbose=True."""
    loop = RLAPHLoop(max_depth=3, verbose=True)
    loop.llm_sync("Test query", depth=1)
    captured = capsys.readouterr()
    assert "[RLM]" in captured.out
    assert "depth 1" in captured.out
```

**Step 2: Implement llm_sync with explicit depth**

In `src/repl/rlaph_loop.py`:

```python
def llm_sync(self, query: str, context: str = "", depth: int | None = None) -> str:
    """
    Synchronous LLM call - returns actual result immediately.

    Uses explicit depth parameter (safer for future async).
    Creates child frames for recursion tracking.
    """
    current_depth = depth if depth is not None else self._depth + 1

    if current_depth > self.max_depth:
        raise RecursionDepthError(current_depth, self.max_depth)

    if self._verbose:
        print(f"[RLM] Recursion at depth {current_depth}: {query[:80]}...")

    result = self.llm_client.call(
        query=query,
        context={"prior": context} if context else None,
        depth=current_depth,
    )

    self._tokens_used += len(result) // 4

    # Create child frame
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
        invalidation_condition={},
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )

    self.frame_index.add(child_frame)
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
    verbose: bool = False,
):
    # ... existing init ...
    self._verbose = verbose
```

**Step 3: Run tests**

```bash
uv run pytest tests/repl/test_llm_recursion.py -v
```

**Step 4: Commit**

```bash
git add src/repl/rlaph_loop.py tests/repl/test_llm_recursion.py
git commit -m "feat: add synchronous llm_sync with explicit depth parameter

- Creates child frames for each recursive call
- Verbose logging for debugging recursion decisions
- Explicit depth (safer for future async)"
```

---

### Task A2: Add Intent Normalization & Canonical Frame ID

**Files:**
- New: `src/frame/canonical_task.py`
- New: `src/frame/intent_extractor.py`
- Modify: `src/frame/causal_frame.py` (add canonical_task field)
- Modify: `src/repl/rlaph_loop.py` (use in frame creation)
- Modify: `src/frame/frame_index.py` (save/load canonical_task)
- Test: `tests/frame/test_canonical_task.py`

**Problem:** Frame identity uses raw query → similar queries create duplicate frames → entropy explosion.

**Solution:** Extract canonical task from query, use hash(canonical_task + context_slice) as frame identity.

**Step 1: Create CanonicalTask dataclass**

In `src/frame/canonical_task.py`:

```python
"""Canonical task representation for frame deduplication."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.llm_client import LLMClient


@dataclass(frozen=True)
class CanonicalTask:
    """
    Normalized representation of a task intent.

    Used for frame identity to prevent duplicate frames from similar queries.
    """
    task_type: str  # "analyze", "debug", "summarize", "implement", "verify"
    target: str | list[str]  # "auth.py" or ["src/auth/*"]
    analysis_scope: str | None = None  # "correctness", "architecture", "security"
    params: dict = field(default_factory=dict)

    def to_hash(self) -> str:
        """Generate stable hash for this canonical task."""
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.blake2b(data.encode(), digest_size=8).hexdigest()

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalTask":
        """Deserialize from dict."""
        return cls(
            task_type=data.get("task_type", "unknown"),
            target=data.get("target", "unknown"),
            analysis_scope=data.get("analysis_scope"),
            params=data.get("params", {}),
        )
```

**Step 2: Create intent extractor (LANGUAGE/FRAMEWORK AGNOSTIC)**

In `src/frame/intent_extractor.py`:

> **Design Principle:** Intent-first, not project-first. The rules work for any language (.py, .ts, .go, .rs) and any framework (FastAPI, NestJS, etc.)

```python
"""Extract canonical task from user query - language/framework agnostic."""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Callable

from .canonical_task import CanonicalTask

if TYPE_CHECKING:
    from ..llm.llm_client import LLMClient


# ──────────────────────────────────────────────────────────────
# Common intent verbs and their canonical task_type + scope
# Language-agnostic: works for Python, TypeScript, Go, Rust, etc.
# ──────────────────────────────────────────────────────────────
INTENT_VERBS: dict[str, tuple[str, str]] = {
    # Analysis / understanding
    r"\b(analyze|analyse|review|check|inspect|examine|look at|study)\b": ("analyze", "overview"),
    r"\b(explain|describe|how does|what is|tell me about)\b": ("explain", "overview"),
    r"\b(summarize|summary|overview|tl;?dr)\b": ("summarize", "overview"),

    # Debugging / correctness
    r"\b(debug|fix|solve|troubleshoot|error|bug|issue|wrong|broken|not working)\b": ("debug", "correctness"),
    r"\b(test|verify|validate|check if|make sure)\b": ("verify", "correctness"),

    # Implementation / change
    r"\b(implement|add|create|build|write|make)\b": ("implement", "functionality"),
    r"\b(refactor|improve|optimize|clean up)\b": ("refactor", "structure"),
    r"\b(document|comment|doc|readme)\b": ("document", "documentation"),

    # Security / performance (override default scope)
    r"\b(security|vulnerability|exploit|safe|attack)\b": ("analyze", "security"),
    r"\b(performance|slow|optimize|scale)\b": ("analyze", "performance"),
    r"\b(architecture|design|structure)\b": ("analyze", "architecture"),
}

# ──────────────────────────────────────────────────────────────
# Target extraction patterns (file, dir, module, function, etc.)
# Extracts ACTUAL target from query, not hardcoded names
# ──────────────────────────────────────────────────────────────
TARGET_PATTERNS: list[tuple[str, Callable]] = [
    # Direct file mention: "auth.py", "server.ts", "main.rs"
    (r"\b([a-zA-Z0-9_-]+\.(py|ts|js|jsx|tsx|go|rs|java|cpp|c|cs|rb|scala|kt))\b",
     lambda m: [m.group(1)]),

    # Directory or module: "src/auth", "controllers/user", "lib/mdns"
    (r"\b(src|lib|app|controllers|services|utils|pkg|internal|cmd)/([a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*)\b",
     lambda m: [f"{m.group(1)}/{m.group(2)}"]),

    # Generic "the X file/module": "the auth file", "the user service"
    (r"\b(the|this|my|our)\s+([a-zA-Z0-9_-]+)\s+(file|module|service|controller|component|endpoints?|api)\b",
     lambda m: [f"**/*{m.group(2)}*"]),

    # Function/method/class: "login function", "UserService class"
    (r"\b([A-Za-z0-9_]+)\s+(function|method|func|class|struct|interface|handler)\b",
     lambda m: [f"**/*{m.group(1)}*"]),

    # "in X" pattern: "in auth", "in the api"
    (r"\b(in|inside)\s+(?:the\s+)?([a-zA-Z0-9_-]+)\b",
     lambda m: [f"**/*{m.group(2)}*"]),
]


def extract_canonical_task(
    query: str,
    llm_client: "LLMClient | None" = None,
    use_llm_fallback: bool = True,
) -> CanonicalTask:
    """
    Extract canonical task from user query.

    Uses fast deterministic rules first (70-80% hit rate), then LLM fallback.

    Design:
    - Intent-first: focuses on VERB + TARGET patterns
    - Language-agnostic: works for .py, .ts, .go, .rs, etc.
    - Framework-agnostic: no hardcoded module names

    Args:
        query: User query string
        llm_client: Optional LLM client for fallback
        use_llm_fallback: Whether to use LLM if rules don't match

    Returns:
        CanonicalTask with normalized intent
    """
    q_lower = query.lower()

    # Step 1: Determine task_type and scope from intent verbs
    task_type = "analyze"  # default fallback
    analysis_scope = "overview"

    for pattern, (tt, scope) in INTENT_VERBS.items():
        if re.search(pattern, q_lower, re.IGNORECASE):
            task_type = tt
            analysis_scope = scope
            break

    # Step 2: Extract target dynamically
    target: str | list[str] | None = None

    for pattern, extractor in TARGET_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            target = extractor(match)
            break

    # Step 3: If we got both task and target → confident rule-based result
    if target is not None:
        return CanonicalTask(
            task_type=task_type,
            target=target,
            analysis_scope=analysis_scope,
            params={"original_query": query[:200]},
        )

    # Step 4: LLM fallback for everything else
    if llm_client and use_llm_fallback:
        try:
            canonical = _extract_with_llm(query, llm_client)
            if canonical:
                return canonical
        except Exception:
            pass  # Fall through to ultimate fallback

    # Ultimate fallback: analyze whole codebase
    return CanonicalTask(
        task_type="analyze",
        target=["**/*"],
        analysis_scope="overview",
        params={"original_query": query[:200]},
    )


def _extract_with_llm(query: str, llm_client: "LLMClient") -> CanonicalTask | None:
    """Use LLM to extract canonical task (fallback for complex queries)."""
    prompt = f'''Extract canonical task from user query: "{query}"

Output JSON only:
{{
  "task_type": "analyze|debug|summarize|implement|refactor|document|verify|explain",
  "target": ["file pattern or list of files"],
  "analysis_scope": "overview|correctness|architecture|performance|security|documentation",
  "params": {{}}
}}

Be precise. Use glob patterns like "src/auth/*.ts" if appropriate. Target should be a list.'''

    result = llm_client.call(prompt)

    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', result, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group(1) if json_match.lastindex else json_match.group())
            # Ensure target is a list
            target = data.get("target", ["**/*"])
            if isinstance(target, str):
                target = [target]

            return CanonicalTask(
                task_type=data.get("task_type", "analyze"),
                target=target,
                analysis_scope=data.get("analysis_scope", "overview"),
                params=data.get("params", {}),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    return None
```

**Step 3: Update CausalFrame to include canonical_task**

In `src/frame/causal_frame.py`:

```python
from .canonical_task import CanonicalTask

@dataclass
class CausalFrame:
    # ... existing fields ...
    canonical_task: CanonicalTask | None = None
```

Update `frame_to_dict` and `dict_to_frame` to handle canonical_task.

**Step 4: Update frame creation in RLAPHLoop**

In `src/repl/rlaph_loop.py`:

```python
from ..frame.canonical_task import CanonicalTask
from ..frame.intent_extractor import extract_canonical_task

# In llm_sync and frame creation:
canonical = extract_canonical_task(query, self.llm_client)
frame_id = canonical.to_hash() + "_" + context_slice_hash

child_frame = CausalFrame(
    frame_id=frame_id,
    canonical_task=canonical,
    # ... rest of fields ...
)
```

**Step 5: Write tests (language-agnostic)**

In `tests/frame/test_canonical_task.py`:

```python
"""Tests for canonical task extraction - language/framework agnostic."""
import pytest
from src.frame.canonical_task import CanonicalTask
from src.frame.intent_extractor import extract_canonical_task


def test_canonical_task_hash_stable():
    """Same canonical task should produce same hash."""
    task1 = CanonicalTask(task_type="analyze", target=["auth.py"], analysis_scope="correctness")
    task2 = CanonicalTask(task_type="analyze", target=["auth.py"], analysis_scope="correctness")
    assert task1.to_hash() == task2.to_hash()


def test_canonical_task_hash_different():
    """Different tasks should produce different hashes."""
    task1 = CanonicalTask(task_type="analyze", target=["auth.py"])
    task2 = CanonicalTask(task_type="debug", target=["auth.py"])
    assert task1.to_hash() != task2.to_hash()


# ──────────────────────────────────────────────────────────────
# Language-agnostic tests (work for any stack)
# ──────────────────────────────────────────────────────────────

def test_extract_python_file():
    """Should extract Python file from query."""
    task = extract_canonical_task("Analyze auth.py for security issues")
    assert task.task_type == "analyze"
    assert "auth.py" in task.target


def test_extract_typescript_file():
    """Should extract TypeScript file from query."""
    task = extract_canonical_task("Debug the error in user.service.ts")
    assert task.task_type == "debug"
    assert "user.service.ts" in task.target


def test_extract_go_file():
    """Should extract Go file from query."""
    task = extract_canonical_task("Explain how main.go works")
    assert task.task_type == "explain"
    assert "main.go" in task.target


def test_extract_rust_file():
    """Should extract Rust file from query."""
    task = extract_canonical_task("Review lib.rs for performance")
    assert task.task_type == "analyze"
    assert "lib.rs" in task.target
    assert task.analysis_scope == "performance"


def test_extract_directory_target():
    """Should extract directory path from query."""
    task = extract_canonical_task("Summarize src/auth module")
    assert task.task_type == "summarize"
    assert any("auth" in t for t in task.target)


def test_extract_generic_module():
    """Should extract module name with glob pattern."""
    task = extract_canonical_task("Implement the cache service")
    assert task.task_type == "implement"
    assert any("cache" in t.lower() for t in task.target)


def test_security_scope_override():
    """Should detect security scope."""
    task = extract_canonical_task("Check for security vulnerabilities")
    assert task.task_type == "analyze"
    assert task.analysis_scope == "security"


def test_ultimate_fallback():
    """Should fallback to whole codebase when no target found."""
    task = extract_canonical_task("What do you think?")
    assert task.task_type == "analyze"
    assert task.target == ["**/*"]


def test_extract_canonical_task_debug():
    """Should extract debug task from debug query."""
    task = extract_canonical_task("Debug the login error")
    assert task.task_type == "debug"
    assert task.target == "auth.py"


def test_canonical_task_serialization():
    """Should serialize/deserialize correctly."""
    task = CanonicalTask(
        task_type="analyze",
        target=["auth.py"],
        analysis_scope="security",
        params={"key": "value"},
    )

    data = task.to_dict()
    restored = CanonicalTask.from_dict(data)

    assert restored.task_type == task.task_type
    assert restored.target == task.target
    assert restored.analysis_scope == task.analysis_scope
    assert restored.params == task.params
```

**Step 6: Run tests**

```bash
uv run pytest tests/frame/test_canonical_task.py -v
```

**Step 7: Commit**

```bash
git add src/frame/canonical_task.py src/frame/intent_extractor.py src/frame/causal_frame.py src/repl/rlaph_loop.py tests/frame/test_canonical_task.py
git commit -m "feat: add intent normalization + canonical_task for frame deduplication

- CanonicalTask dataclass with stable hash
- Language-agnostic intent extractor (works for .py, .ts, .go, .rs)
- Intent-first design: verb + target patterns, not hardcoded modules
- Dynamic target extraction from query (not project-specific)
- Hybrid rule (70-80% hit rate) + LLM fallback
- Frame identity = canonical_task.hash() + context_slice.hash()"
```

---

### Task A3: Structured invalidation_condition (dict)

**Files:**
- Modify: `src/frame/causal_frame.py`
- Modify: `src/repl/rlaph_loop.py`

**Step 1: Update generate_invalidation_condition**

In `src/frame/causal_frame.py`:

```python
from dataclasses import field
from pathlib import Path

def generate_invalidation_condition(context_slice: "ContextSlice") -> dict:
    """
    Generate structured invalidation condition from context_slice.

    Returns a dict that can be programmatically checked, not just a string.
    """
    return {
        "files": list(context_slice.files.keys()) if context_slice.files else [],
        "tools": list(context_slice.tool_outputs.keys()) if context_slice.tool_outputs else [],
        "memory_refs": list(context_slice.memory_refs) if context_slice.memory_refs else [],
        "description": _generate_description(context_slice),
    }


def _generate_description(context_slice: "ContextSlice") -> str:
    """Generate human-readable description for debugging."""
    parts = []

    if context_slice.files:
        filenames = [Path(p).name for p in context_slice.files.keys()]
        if len(filenames) == 1:
            parts.append(f"{filenames[0]} changes or is deleted")
        else:
            shown = filenames[:3]
            more = f" (+{len(filenames) - 3} more)" if len(filenames) > 3 else ""
            parts.append(f"any of {len(filenames)} files ({', '.join(shown)}{more}) change")

    if context_slice.tool_outputs:
        tool_names = list(context_slice.tool_outputs.keys())
        parts.append(f"tool results from {', '.join(tool_names)} change")

    if context_slice.memory_refs:
        parts.append("memory entries change")

    if not parts:
        return "No automatic invalidation condition"

    return "; or ".join(parts)
```

**Step 2: Update CausalFrame field**

```python
@dataclass
class CausalFrame:
    # ... existing fields ...
    invalidation_condition: dict = field(default_factory=dict)  # Changed from str
```

**Step 3: Update tests**

```python
def test_generate_invalidation_condition_structured():
    """Should return dict with structured data."""
    context_slice = ContextSlice(
        files={"/src/main.py": "hash123"},
        memory_refs=[],
        tool_outputs={"Read": "hash456"},
        token_budget=8000,
    )

    condition = generate_invalidation_condition(context_slice)

    assert isinstance(condition, dict)
    assert "files" in condition
    assert "tools" in condition
    assert "description" in condition
    assert "/src/main.py" in condition["files"]
    assert "Read" in condition["tools"]
```

**Step 4: Commit**

```bash
git add src/frame/causal_frame.py src/repl/rlaph_loop.py tests/frame/test_invalidation_condition.py
git commit -m "feat: make invalidation_condition structured dict

- Now {\"files\": [...], \"tools\": [...], \"description\": \"...\"}
- Easier to automate/execute invalidation checks
- Backward compatible with human-readable description"
```

---

### Task A4: Enhance System Prompt for Recursion

**Files:**
- Modify: `src/repl/rlaph_loop.py` (_build_system_prompt)

**Step 1: Update system prompt**

```python
def _build_system_prompt(self) -> str:
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
- The system tracks frame identity by task intent, not exact query wording

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
git commit -m "feat: enhance system prompt with recursion guardrails

- Explicit rules against tiny-step recursion
- Frame identity explanation
- Better examples for decomposition"
```

---

### Task A5: Add Query/Intent Tracking to FrameIndex

**Files:**
- Modify: `src/frame/frame_index.py`
- Modify: `scripts/run_orchestrator.py`
- Test: `tests/frame/test_frame_index_query.py`

**Step 1: Add fields to FrameIndex**

```python
@dataclass
class FrameIndex:
    initial_query: str = ""
    query_summary: str = ""
    commit_hash: str | None = None
    _frames: dict[str, CausalFrame] = field(default_factory=dict)
    _dependent_cache: dict[str, set[str]] = field(default_factory=dict)
```

**Step 2: Update save/load**

```python
def save(self, session_id: str, base_dir: Path | None = None) -> Path:
    data = {
        "initial_query": self.initial_query,
        "query_summary": self.query_summary,
        "commit_hash": self.commit_hash,
        "frames": [frame_to_dict(f) for f in self._frames.values()],
    }
    # ... save to file ...

@classmethod
def load(cls, session_id: str, base_dir: Path | None = None) -> "FrameIndex | None":
    # ... load from file ...
    return cls(
        initial_query=data.get("initial_query", ""),
        query_summary=data.get("query_summary", ""),
        commit_hash=data.get("commit_hash"),
        _frames=frames,
    )
```

**Step 3: Commit**

```bash
git add src/frame/frame_index.py scripts/run_orchestrator.py tests/frame/test_frame_index_query.py
git commit -m "feat: add query/intent tracking to FrameIndex

- initial_query: Full user query
- query_summary: Short summary for cross-session matching
- Persisted in index.json"
```

---

## Part B: Tree Structure + Evidence + Cascade

### Task B1: Populate Children in Frame Tree

**Files:**
- Modify: `src/frame/frame_index.py`
- Test: `tests/repl/test_frame_tree.py`

**Step 1: Update add() method**

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
```

**Step 2: Commit**

```bash
git add src/frame/frame_index.py tests/repl/test_frame_tree.py
git commit -m "feat: populate children list when adding frames to index

- FrameIndex.add updates parent's children list
- Defensive check for duplicates
- Clear dependent cache on changes"
```

---

### Task B2: Auto Evidence Tracking (COMPLETED only)

**Files:**
- Modify: `src/frame/frame_index.py`
- Test: `tests/repl/test_evidence_tracking.py`

**Step 1: Update add() method**

```python
def add(self, frame: "CausalFrame") -> None:
    """Add a frame, update parent's children and evidence."""
    self._frames[frame.frame_id] = frame
    self._dependent_cache.clear()

    if frame.parent_id and frame.parent_id in self._frames:
        parent = self._frames[frame.parent_id]

        # Add to children
        if frame.frame_id not in parent.children:
            parent.children.append(frame.frame_id)

        # Add to evidence ONLY if COMPLETED
        if frame.status == FrameStatus.COMPLETED:
            if frame.frame_id not in parent.evidence:
                parent.evidence.append(frame.frame_id)
```

**Step 2: Commit**

```bash
git add src/frame/frame_index.py tests/repl/test_evidence_tracking.py
git commit -m "feat: auto-track evidence for COMPLETED children only

- Prevents invalidated frames from polluting evidence
- Defensive duplicate check"
```

---

### Task B3: find_dependent_frames with Caching

**Files:**
- Modify: `src/frame/frame_index.py`
- Test: `tests/frame/test_find_dependent.py`

**Step 1: Add method with caching**

```python
def find_dependent_frames(self, frame_id: str) -> set[str]:
    """
    Find all frames that depend on a given frame.

    Dependents include:
    - Children (frames with this frame as parent_id)
    - Evidence consumers (frames citing this frame in evidence)

    Results are cached until frames change.
    """
    if frame_id in self._dependent_cache:
        return self._dependent_cache[frame_id].copy()

    dependents = set()

    for fid, frame in self._frames.items():
        if frame.parent_id == frame_id:
            dependents.add(fid)
        if frame_id in frame.evidence:
            dependents.add(fid)

    self._dependent_cache[frame_id] = dependents
    return dependents.copy()
```

**Step 2: Commit**

```bash
git add src/frame/frame_index.py tests/frame/test_find_dependent.py
git commit -m "feat: add find_dependent_frames with caching

- O(n) scan with cache for repeated lookups
- Returns copy to prevent cache corruption"
```

---

### Task B4: Strengthen Cascade Propagation

**Files:**
- Modify: `src/frame/frame_invalidation.py`
- Test: `tests/frame/test_cascade_invalidation.py`

**Step 1: Implement robust cascade**

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
    """
    from .causal_frame import FrameStatus

    invalidated = set()

    def _invalidate(fid: str, current_reason: str):
        if fid in invalidated:
            return  # Cycle detection
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
            _invalidate(dep_id, f"Evidence invalidated: {fid}")

    _invalidate(frame_id, reason)
    return invalidated
```

**Step 2: Commit**

```bash
git add src/frame/frame_invalidation.py tests/frame/test_cascade_invalidation.py
git commit -m "feat: strengthen cascade with find_dependent_frames

- Use cached dependent discovery
- Handle circular evidence gracefully
- Clear escalation_reason chain"
```

---

### Task B5: E2E Integration Test

**Files:**
- Create: `tests/integration/test_tree_cascade_integration.py`

**Step 1: Write comprehensive integration test**

```python
"""E2E integration test for tree + evidence + cascade + canonical_task."""
import pytest
from src.frame.frame_index import FrameIndex
from src.frame.frame_invalidation import propagate_invalidation
from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.frame.canonical_task import CanonicalTask
from datetime import datetime


def make_frame(
    frame_id: str,
    parent_id: str = None,
    evidence: list = None,
    canonical_task: CanonicalTask = None,
    depth: int = 0
) -> CausalFrame:
    return CausalFrame(
        frame_id=frame_id,
        depth=depth,
        parent_id=parent_id,
        children=[],
        query=f"query_{frame_id}",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=8000),
        evidence=evidence or [],
        conclusion=f"conclusion_{frame_id}",
        confidence=0.8,
        invalidation_condition={},
        status=FrameStatus.COMPLETED,
        canonical_task=canonical_task,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )


def test_full_tree_with_canonical_task():
    """E2E: Tree with canonical_task prevents duplicates."""
    index = FrameIndex()

    # Same canonical task should not create duplicate frames
    task1 = CanonicalTask(task_type="analyze", target="auth.py", analysis_scope="correctness")
    task2 = CanonicalTask(task_type="analyze", target="auth.py", analysis_scope="correctness")

    # Same hash
    assert task1.to_hash() == task2.to_hash()

    # Create frames with canonical task
    root = make_frame("root", canonical_task=task1, depth=0)
    index.add(root)

    # Verify canonical_task is stored
    assert index.get("root").canonical_task is not None
    assert index.get("root").canonical_task.task_type == "analyze"


def test_cascade_with_evidence_and_children():
    """E2E: Cascade invalidates children + evidence consumers."""
    index = FrameIndex()

    root = make_frame("root", depth=0)
    index.add(root)

    child1 = make_frame("child1", parent_id="root", depth=1)
    index.add(child1)

    leaf1 = make_frame("leaf1", parent_id="child1", depth=2)
    index.add(leaf1)

    # Evidence user (cites leaf1)
    evidence_user = make_frame("ev_user", evidence=["leaf1"], depth=1)
    index.add(evidence_user)

    # Verify structure
    assert "child1" in index.get("root").children
    assert "leaf1" in index.get("child1").children
    assert "leaf1" in index.get("child1").evidence

    # Invalidate root
    invalidated = propagate_invalidation("root", "test cascade", index)

    # All should be invalidated
    assert len(invalidated) == 4
    for fid in ["root", "child1", "leaf1", "ev_user"]:
        assert fid in invalidated
        assert index.get(fid).status == FrameStatus.INVALIDATED


def test_partial_invalidation():
    """E2E: Partial subtree invalidation."""
    index = FrameIndex()

    root = make_frame("root_p", depth=0)
    index.add(root)

    child1 = make_frame("child1_p", parent_id="root_p", depth=1)
    index.add(child1)

    child2 = make_frame("child2_p", parent_id="root_p", depth=1)
    index.add(child2)

    # Invalidate only child1
    invalidated = propagate_invalidation("child1_p", "child1 reason", index)

    assert "child1_p" in invalidated
    assert "root_p" not in invalidated
    assert "child2_p" not in invalidated


def test_query_tracking_persistence():
    """E2E: Query tracking persists through save/load."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        index = FrameIndex(
            initial_query="Analyze auth module",
            query_summary="Auth analysis"
        )

        root = make_frame("qroot", depth=0)
        index.add(root)

        index.save("test_query", tmp_path)

        loaded = FrameIndex.load("test_query", tmp_path)

        assert loaded.initial_query == "Analyze auth module"
        assert loaded.query_summary == "Auth analysis"
```

**Step 2: Run integration test**

```bash
uv run pytest tests/integration/test_tree_cascade_integration.py -v
```

**Step 3: Commit**

```bash
git add tests/integration/test_tree_cascade_integration.py
git commit -m "test: add E2E integration test for tree + evidence + cascade + canonical_task

- Full tree cascade with evidence links
- Partial subtree invalidation
- Canonical task deduplication
- Query tracking persistence"
```

---

## Verification

Run all tests:

```bash
uv run pytest tests/ -v --tb=short
```

---

## Summary

### Part A: Recursion + Intent Normalization
1. **llm_sync with explicit depth** - Creates child frames, verbose logging
2. **CanonicalTask + intent extractor** - Prevents duplicate frames
   - Language-agnostic: works for .py, .ts, .go, .rs
   - Intent-first design: verb + target patterns
   - Dynamic target extraction (not project-specific)
   - 70-80% rule hit rate + LLM fallback
3. **Structured invalidation_condition** - Dict instead of string
4. **Enhanced system prompt** - Guardrails against depth explosion
5. **Query tracking** - initial_query and query_summary

### Part B: Tree + Evidence + Cascade
6. **Children population** - Auto-update parent.children
7. **Evidence tracking** - Only COMPLETED frames
8. **find_dependent_frames** - Cached O(n) lookup
9. **Cascade propagation** - Uses dependent discovery
10. **E2E integration test** - Validates everything

### Key Design Decisions
- Frame identity = canonical_task.hash() + context_slice.hash()
- Intent extractor is **language/framework agnostic**
- Evidence only from COMPLETED children
- invalidation_condition is dict (structured)
- Synchronous llm(sub_query), no subprocess
- Verbose logging for debugging

### After Phase 18
Run 2-session demo:
1. Session 1: "auth 분석해줘" → canonical_task("analyze", ["**/*auth*"])
2. Change auth.py
3. Session 2: "auth 코드 어떻게 생각해?" → same canonical_task → detects change → targeted invalidation
