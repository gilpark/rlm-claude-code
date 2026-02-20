# RLM-Claude-Code: Design Document

*February 2026 — v2.1*

---

## Current Reality vs. Vision (Feb 20, 2026)

The core data model (CausalFrame + tree + invalidation) and spatial externalization (ContextMap + REPL) are implemented and working.
Recursion now branches correctly (depth > 0, children populated).

**What remains** is exercising the system over multiple sessions:
- Change a file → see targeted invalidation
- Ask to resume a suspended branch → see it pick up where it left off
- Watch living documentation auto-update only affected sections

These are not theoretical — the scaffolding is in place. The next unlock is multi-session workflows where prior knowledge survives code changes.

---

## One Problem, Two Mechanisms

### The Problem

> **Can an AI's work product continuously evolve with its environment?**

Context rot — Claude degrading as context grows — is a real friction, but not the fundamental problem. Context windows will keep growing, as RAM did in the 1990s. The engineering constraint will dissolve.

The problem that doesn't dissolve: every session starts cold. The AI re-discovers what it already knew, re-makes decisions it already made, and has no way to detect when something it previously understood has changed. It cannot say "I analyzed this before, but auth.py changed since then, so my prior analysis may be invalid."

This is reasoning amnesia. And it is an architecture problem, not a hardware problem.

### Two Mechanisms

**Mechanism 1: REPL — Active Navigation**

Zhang et al. (2025) showed that externalizing context into a REPL the model can actively navigate eliminates context rot. But the deeper insight is not about token limits — it is about the shift from passive to active:

```
Passive: load everything → model processes it all
Active:  model peeks → model searches → model loads only what this sub-task needs
```

The model navigates its environment. It doesn't drown in it. This remains the right approach even when context windows are large — not because of token limits, but because focused context produces better reasoning.

**Mechanism 2: CausalFrame — Temporal Persistence**

The question is what to persist across sessions. Three options:

```
I/O (logs):       captures what was said, not why
                  → can't verify, can't detect staleness

REPL state:       captures what was computed, not the reasoning
                  → no provenance, model can't judge whether to trust it

Causation:        captures why the conclusion was reached
                  → verifiable, invalidatable, navigable
```

Causation is the only unit that supports:
- *Verification*: re-trace the chain, confirm it still holds
- *Invalidation*: when a premise changes, find all dependent conclusions
- *Navigation*: given a new question, find relevant prior reasoning

**Insight: Core enforces structure. LM decides content and policy.**

| Core enforces | LM decides |
|---------------|------------|
| token_budget | what files to include in context |
| max depth | whether to recurse or not |
| CausalFrame structure | confidence interpretation |
| invalidation cascade | escalation policy |

When in doubt: structural constraint → Core. Judgment call → LM.

---

## Architecture

Neither mechanism works without the other.

**REPL alone:** the model navigates context well within a session — but when the session ends, everything is lost. No externalization across time.

**CausalFrame alone:** reasoning history is stored — but the model still drowns in context within a session. And critically, CausalFrame creation depends on `llm()`, which depends on REPL. The mechanisms are coupled.

**Together:** the model externalizes spatially (REPL navigates current context) and temporally (CausalFrame persists the causal history of that navigation). Externalization is complete only when both dimensions are covered.

```
Claude Code
    │
    └── rlm-orchestrator (entry point)
        │
        ├── ContextMap              ← session-scoped, lazy file navigator (git ls-files or dynamic)
        │   └── repl_environment.py
        │
        ├── RLAPHLoop               ← synchronous recursion via llm(sub_query)
        │   └── creates child frames + evidence links
        │
        ├── Causal Layer            ← persistent tree (JSONL)
        │   ├── CausalFrame         ← now auto-generates invalidation_condition
        │   ├── FrameIndex          ← children + evidence populated
        │   ├── context_slice.py
        │   ├── frame_invalidation.py ← cascade (down + sideways via find_dependent_frames)
        │   ├── frame_store.py
        │   ├── session_artifacts.py
        │   └── session_comparison.py
        │
        ├── Plugin Layer            ← extensibility (future work)
        │   ├── plugin_interface.py
        │   └── plugins/
        │
        └── hooks/                  ← connect REPL to Causal Layer
            ├── SessionStart        → load prior frames + git diff invalidation
            ├── PostToolUse         → capture tool outputs + refresh ContextMap
            └── Stop                → save tree + initial_query
```

The hooks are the coupling point: `PostToolUse` captures what the REPL saw into the active CausalFrame's context_slice. `Stop` persists the frame tree built during REPL execution. Without hooks, REPL and Causal Layer are independent modules. With hooks, they are one system.

---

## Part 1: REPL Layer

### RLAPH Loop — Immediate Execution

**Status: Fully implemented with synchronous recursion.** The `llm(sub_query)` call now creates child frames and returns results synchronously.

The original approach stacks deferred/async LLM calls into a queue and batch-executes. This is fundamentally broken:

```
# Deferred (wrong)
llm(query_1) → queue
llm(query_2) → queue   # can't use result_1 yet
llm(query_3) → queue
→ batch execute → error somewhere → which one? how to recover?
```

RLAPH (Recursive LLM Agent with Python REPL) executes immediately:

```python
# Immediate (correct)
result_1 = llm(query_1)            # executes now, error handled here
result_2 = llm(query_2, result_1)  # result_1 available immediately
result_3 = llm(query_3, result_2)  # builds naturally
```

**Why this matters for CausalFrame:**
With immediate execution, the frame lifecycle is unambiguous:
```
llm() called → CausalFrame created (status: RUNNING)
llm() returns → conclusion stored  (status: COMPLETED)
llm() errors  → error recorded     (status: INVALIDATED)
```

With deferred execution, when does the frame get created? Before the batch? After? What if batch partially fails? The deferred model makes frame lifecycle ambiguous.

**Error handling is local:**
```python
try:
    result = llm("analyze auth.py", context={"files": ["auth.py"]})
except LLMError as e:
    # handle exactly here, frame marked INVALIDATED
    # parent frame can decide: retry, skip, or suspend
```

No global error recovery. No partial batch unwind. Each `llm()` call is self-contained.

### LLM Client — Provider Agnostic

`llm_client.py` is kept as a swappable provider layer. The LM in REPL calls `llm()`, not the API directly:

```python
class LLMClient:
    def call(
        self,
        query: str,
        context: dict,
        model: str | None = None   # None = use default for this depth
    ) -> str: ...
```

Provider swap happens in one place. Default model cascade: root uses larger model, sub-calls use smaller. LM can override per-call if needed.

### REPL Functions

**Status: Implemented.** `llm()` now supports synchronous recursion, creating child frames automatically.

```python
peek(var, start, end)                    # inspect context before committing
search(var, pattern)                     # narrow focus, find relevant parts
llm(query, context)                      # immediate nested LLM call (depth+1)
                                         # auto-creates CausalFrame
llm_batch([(q1,c1), (q2,c2)])           # parallel immediate calls
map_reduce(content, map_fn, reduce_fn)  # partition + aggregate
```

`llm()` internally:
1. Creates child CausalFrame (status: RUNNING)
2. Calls LLMClient with context_slice
3. Stores conclusion in frame (status: COMPLETED)
4. Returns conclusion to caller

The LM never sees frames directly — it just calls `llm()` and gets a result. Frame management is invisible.

---

## Part 2: Causal Layer

### CausalFrame

**Status: Fully implemented.** `invalidation_condition` is now auto-generated from `context_slice.files` and `tool_outputs`.

```python
@dataclass
class CausalFrame:
    # Identity — DETERMINISTIC
    frame_id: str          # hash(parent_id + query + context_slice.hash())
    depth: int             # 0 = root
    parent_id: str | None
    children: list[str]

    # Reasoning
    query: str
    context_slice: ContextSlice
    evidence: list[str]           # frame IDs + raw observations
    conclusion: str | None
    confidence: float
    invalidation_condition: str   # what would make this wrong

    # Branch management
    status: FrameStatus
    branched_from: str | None     # pivot: which frame this branched from

    created_at: datetime
    completed_at: datetime | None

class FrameStatus(Enum):
    RUNNING     = "running"
    COMPLETED   = "completed"
    SUSPENDED   = "suspended"     # pivot — preserved, not deleted
    INVALIDATED = "invalidated"
    PROMOTED    = "promoted"      # persisted as long-term knowledge
```

A PROMOTED frame is a persisted fact. No separate facts store needed:

```python
# This IS the fact
CausalFrame(
    conclusion="This project uses FastAPI",
    evidence=["package.json frame", "imports scan"],
    invalidation_condition="package.json modified",
    status=FrameStatus.PROMOTED
)
```

### ContextSlice

**Status: Fully implemented with lazy loading and auto-discovery.** Files inside the working directory are discovered dynamically via git ls-files or filesystem scan.

```python
@dataclass
class ContextSlice:
    files: dict[str, str]        # file_path → content_hash
    memory_refs: list[str]
    tool_outputs: dict[str, str]
    token_budget: int            # Core enforces this

    def hash(self) -> str: ...
```

Core enforces `token_budget`. LM decides what files to include:

```python
# LM in REPL
result = llm(
    "analyze compute_hash()",
    context={"files": ["utils/hash.py"]}  # LM chose this
)
# Core computes budget: parent_budget // expected_children * 0.8
```

### Invalidation

**Status: Partially implemented.** Downward cascade (children) works. Sideways cascade via evidence links uses `find_dependent_frames` scan. Evidence linking adds child frame IDs to parent.evidence only on COMPLETED status.

```python
def invalidate(frame_id: str, reason: str, index: dict) -> list[str]:
    invalidated = {frame_id}
    frame = index[frame_id]

    # Down: children
    for child_id in frame.children:
        invalidated.update(invalidate(child_id, reason, index))

    # Sideways: frames that used this as evidence
    for other_id, other in index.items():
        if frame_id in other.evidence and other_id not in invalidated:
            invalidated.update(invalidate(other_id, reason, index))

    return list(invalidated)
```

O(n) at n=10-20. No DAG. Correct at this scale.

Suspended frames are invalidated but preserved — recoverable if the invalidation condition no longer applies.

### Branch Management

**Status: Data model implemented, usage coming soon.** The fields exist but real suspension/resumption logic is not yet exercised.

Users pivot and backtrack. Two fields handle all cases:

```
frame_root        (status: COMPLETED)
├── frame_auth    (status: SUSPENDED, branched_from: None)
└── frame_db      (status: RUNNING,   branched_from: "frame_auth")
```

```python
# Active work
active    = [f for f in index.values() if f.status == FrameStatus.RUNNING]
# Pivots
suspended = [f for f in index.values() if f.status == FrameStatus.SUSPENDED]
# Where pivots happened
pivots    = [f for f in index.values() if f.branched_from]
```

### FrameStore

**Status: Implemented and working.** SQLite is unnecessary — 10-20 frames fit comfortably in JSONL, and it remains human-readable.

**Sub-call stitching:** When `llm()` creates a sub-call, Claude Code creates a new session:

```
root session: session_abc
  └── llm() → sub session: session_xyz
        └── llm() → sub session: session_def
```

The CausalFrame tree gets split across multiple transcripts. The `PostToolUse` hook stitches them together:

```
PostToolUse (after each sub-call completes)
  → capture sub-call session_id + result
  → track root_session_id
  → append to root_session_id.jsonl
```

Result: `~/.claude/rlm-frames/{root_session_id}.jsonl` contains the entire call tree:

```jsonl
{"frame_id": "f1", "depth": 0, "session_id": "session_abc", "parent_id": null}
{"frame_id": "f2", "depth": 1, "session_id": "session_xyz", "parent_id": "f1"}
{"frame_id": "f3", "depth": 2, "session_id": "session_def", "parent_id": "f2"}
```

`session_id` tracks which Claude Code session executed the frame. `parent_id` encodes the tree structure.

```python
class FrameStore:
    path: Path  # ~/.claude/rlm-frames/{root_session_id}.jsonl

    def save(frame: CausalFrame) -> None         # json.dumps + append
    def load(frame_id: str) -> CausalFrame       # readlines + match
    def list() -> list[CausalFrame]              # readlines + parse
    def find_by_status(status) -> list           # list + filter
```

Zero dependencies. One file. Human-readable.

### SessionArtifacts

**Status: Implemented.** `initial_query` is now tracked in FrameIndex for session continuity.

```python
@dataclass
class SessionArtifacts:
    session_id: str
    initial_prompt: str           # why this session started
    files: dict[str, FileRecord]  # what was in scope
    root_frame_id: str
    conversation_log: str         # Claude Code transcript path (source of truth)

@dataclass
class FileRecord:
    path: str
    hash: str                     # change detection
    role: str                     # "read" | "modified" | "created"
```

On SessionStart:
```python
diff = compare_sessions(current, prior, index)
# → same task?
# → which files changed → which frames invalidated?
# → which suspended branches worth resuming?
```

### Current Implementation Status (Feb 20, 2026)

| Feature                        | Implemented? | Notes / Next |
|--------------------------------|--------------|--------------|
| Synchronous llm() recursion    | Yes          | llm(sub_query) creates child frames; depth tracking works |
| Tree structure (children)      | Yes          | Populated in FrameIndex.add() |
| Evidence linking               | Partial      | Added to parent.evidence on child completion (only COMPLETED) |
| Auto invalidation_condition    | Yes          | Generated from context_slice.files & tool_outputs |
| Cascade invalidation           | Partial      | Down to children + sideways via evidence; O(n) scan |
| Git-aware change detection     | Yes          | commit_hash + git diff on load |
| Query/intent tracking          | Yes          | initial_query saved in FrameIndex |
| ContextMap lazy loading        | Yes          | git ls-files + dynamic discovery inside working dir |
| Multi-session resumption       | No           | Data model ready; needs UI/skill to resume branches |
| Living docs / surgical updates | No           | Hooks ready; needs plugin to re-run invalidated frames |
| Plugin interface               | No           | Future work |

The system now supports **real tree-structured reasoning** within a session.
The next unlock is **multi-session workflows** where prior knowledge survives code changes.

---

## Part 3: Plugin Layer

**Status: Future work.** The plugin interface is designed but not yet implemented.

### Why Plugins

Some use cases need to intercept `llm()` I/O without owning the call tree:

- **Living Docs**: git diff → updated documentation
- **Code Review**: PR diff → structured feedback that learns across PRs
- **Test Generation**: changed signatures → test specs (rationale, not code)

The call tree structure is invariant. What varies is: what does the model see, what do we keep, where does it go.

### Interface

```python
class RLMPlugin(Protocol):

    def transform_input(
        self,
        raw_input: Any,
        ctx: CoreContext
    ) -> dict:
        """What does the model see? Returns context_slice."""
        ...

    def parse_output(
        self,
        raw_output: str,
        frame: CausalFrame
    ) -> Any:
        """What do we keep? Must preserve invalidation_condition."""
        ...

    def store(
        self,
        parsed: Any,
        frame: CausalFrame
    ) -> None:
        """Where does it go? Side effect only."""
        ...


class CoreContext(TypedDict):
    current_frame: CausalFrame
    index: dict[str, CausalFrame]
    artifacts: SessionArtifacts
    changed_files: list[str]
    invalidated_frames: list[str]
    suspended_frames: list[str]
```

### Plugin Rules

**Complexity lives in `transform_input`.** What does this frame need to see? That's the domain-specific judgment.

**`parse_output` must preserve `invalidation_condition`.** If it doesn't, output is a cache, not living knowledge.

**`store` is a side effect, not a decision.** Decisions belong in `parse_output`.

**Plugins cannot modify the call tree.** They read `CoreContext`, write to their own storage. Core owns causality.

**LLM provider flows through Core.** Plugins do not call LLMClient directly. `llm()` is the only entry point.

### Cognitive Loop

```
INPUT arrives (git diff, PR, query)
    ↓
Core resolves context (changed files, invalidated frames, suspended branches)
    ↓
transform_input() → context_slice for this frame
    ↓
llm() executes immediately (RLAPH) → CausalFrame created
    ↓
parse_output() → structured result with invalidation_condition
    ↓
store() → plugin storage
    ↓
Core: invalidation check → cascade if conclusion changed
```

Steps 1, 3, 5 differ per plugin. Everything else is Core.

---

## What We Rejected and Why

| Rejected | Reason |
|----------|--------|
| Deferred/async LLM stack | Error recovery is unclear, frame lifecycle is ambiguous |
| Subprocess for recursion | Synchronous llm(sub_query) is simpler, faster, and sufficient for 10–20 frames |
| DAG structure | n=10-20, O(n) scan is correct |
| BranchPoint dataclass | status + branched_from is sufficient |
| facts/experiences store | PROMOTED CausalFrame is the fact |
| trajectory.py (event log) | FrameStore is clearer and sufficient |
| Core-enforced confidence threshold | LM decides escalation policy |
| Core-enforced context slicing | LM decides what it needs |
| Majority voting | 3x cost at 2-3M tokens per frame |
| SessionLink | undefined concept, removed |
| Vector/embedding search | 10-20 frames, flat scan is instant |
| SQLite / MemoryBackend abstraction | JSONL per session, zero dependencies |
| Complexity classifier / routing | LM navigates in REPL |

---

## File List

```
src/
├── rlaph_loop.py            # immediate execution loop with recursion
├── repl_environment.py      # peek, search, llm, llm_batch, map_reduce
├── llm_client.py            # provider-agnostic LLM calls
├── tool_bridge.py           # controlled tool access
├── causal_frame.py          # CausalFrame, FrameStatus, compute_frame_id
├── context_slice.py         # ContextSlice, hash, budget, lazy loading
├── frame_index.py           # dict[str, CausalFrame], branch queries, initial_query
├── frame_invalidation.py    # propagate_invalidation, find_dependent_frames
├── frame_store.py           # JSONL: save, load, list, find_by_status
├── session_artifacts.py     # SessionArtifacts, FileRecord
├── session_comparison.py    # compare_sessions, SessionDiff
└── plugin_interface.py      # CoreContext, RLMPlugin protocol (future)

scripts/
├── extract_frames.py        # Stop hook: extract + save
└── compare_sessions.py      # SessionStart hook: diff + surface

hooks/
└── hooks.json               # SessionStart, PostToolUse, Stop
```

**12 src files. 2 scripts. 1 hooks file.**

---

*Based on: Zhang et al., "Recursive Language Models" (2025)*
*One problem. Two mechanisms. One principle: externalize what the model cannot hold, and let it navigate actively.*