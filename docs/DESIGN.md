# RLM-Claude-Code: Design Document

*February 2026 — v2*

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
    └── rlm-claude-code (plugin)
        │
        ├── REPL Layer              ← spatial externalization (within session)
        │   ├── repl_environment.py
        │   ├── rlaph_loop.py
        │   ├── llm_client.py
        │   └── tool_bridge.py
        │
        ├── Causal Layer            ← temporal externalization (across sessions)
        │   ├── causal_frame.py
        │   ├── context_slice.py
        │   ├── frame_index.py
        │   ├── frame_invalidation.py
        │   ├── frame_store.py
        │   ├── session_artifacts.py
        │   └── session_comparison.py
        │
        ├── Plugin Layer            ← extensibility
        │   ├── plugin_interface.py
        │   └── plugins/
        │
        └── hooks/                  ← where the two mechanisms connect
            ├── SessionStart        → stitch sub-call frames, compare prior session
            ├── PostToolUse         → capture tool outputs into active CausalFrame
            └── Stop                → extract frame tree, save SessionArtifacts
```

The hooks are the coupling point: `PostToolUse` captures what the REPL saw into the active CausalFrame's context_slice. `Stop` persists the frame tree built during REPL execution. Without hooks, REPL and Causal Layer are independent modules. With hooks, they are one system.

---

## Part 1: REPL Layer

### RLAPH Loop — Immediate Execution

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

SQLite는 불필요하다. 10-20 frames는 JSONL이면 충분하고, 사람이 직접 읽을 수 있다.

**Sub-call 문제:** `llm()`이 sub-call을 만들면 Claude Code는 새 세션을 생성한다:

```
root session: session_abc
  └── llm() → sub session: session_xyz
        └── llm() → sub session: session_def
```

CausalFrame tree가 여러 transcript에 쪼개진다. `PostToolUse` hook이 하나로 stitch한다:

```
PostToolUse (sub-call 끝날 때마다)
  → sub-call session_id + result 캡처
  → root_session_id 추적
  → root_session_id.jsonl에 append
```

결과: `~/.claude/rlm-frames/{root_session_id}.jsonl` 하나에 전체 call tree:

```jsonl
{"frame_id": "f1", "depth": 0, "session_id": "session_abc", "parent_id": null}
{"frame_id": "f2", "depth": 1, "session_id": "session_xyz", "parent_id": "f1"}
{"frame_id": "f3", "depth": 2, "session_id": "session_def", "parent_id": "f2"}
```

`session_id`는 어느 Claude Code 세션에서 실행됐는지 추적. `parent_id`가 tree 구조를 담는다.

```python
class FrameStore:
    path: Path  # ~/.claude/rlm-frames/{root_session_id}.jsonl

    def save(frame: CausalFrame) -> None         # json.dumps + append
    def load(frame_id: str) -> CausalFrame       # readlines + match
    def list() -> list[CausalFrame]              # readlines + parse
    def find_by_status(status) -> list           # list + filter
```

의존성 제로. 파일 하나. 사람이 읽을 수 있다.

### SessionArtifacts

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

---

## Part 3: Plugin Layer

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
├── rlaph_loop.py            # immediate execution loop
├── repl_environment.py      # peek, search, llm, llm_batch, map_reduce
├── llm_client.py            # provider-agnostic LLM calls
├── tool_bridge.py           # controlled tool access
├── causal_frame.py          # CausalFrame, FrameStatus, compute_frame_id
├── context_slice.py         # ContextSlice, hash, budget
├── frame_index.py           # dict[str, CausalFrame], branch queries
├── frame_invalidation.py    # propagate_invalidation
├── frame_store.py           # JSONL: save, load, list, find_by_status
├── session_artifacts.py     # SessionArtifacts, FileRecord
├── session_comparison.py    # compare_sessions, SessionDiff
└── plugin_interface.py      # CoreContext, RLMPlugin protocol

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