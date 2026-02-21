# RLM-Claude-Code: Design Document

*February 2026 — v2.3*

---

## Current Reality vs. Vision (Feb 20, 2026 — post-Phase 18)

The core data model (CausalFrame + tree + invalidation) and spatial externalization (ContextMap + REPL) are implemented and working.

**Recursion now branches correctly** (depth > 0, children populated, evidence linking from completed children).

**Intent normalization** is added: frame identity is based on canonical_task (task_type + target + scope), not raw query string — preventing duplication and enabling reuse.

**What remains** is exercising the system over multiple sessions:
- Change a file → see targeted invalidation
- Ask to resume a suspended branch → see it pick up where it left off
- Watch living documentation auto-update only affected sections

These are no longer theoretical — the scaffolding is complete. The next unlock is multi-session workflows where prior knowledge survives code changes and task reuse becomes visible.

---

## One Problem, Two Mechanisms

### The Problem

> **Can an AI's work product continuously evolve with its environment?**

Context rot is real friction, but not the fundamental problem. Context windows will keep growing. The engineering constraint will dissolve.

The enduring problem: every session starts cold. The AI re-discovers what it already knew, re-makes decisions it already made, and has no way to detect when something it previously understood has changed. It cannot say "I analyzed this before, but auth.py changed since then, so my prior analysis may be invalid."

This is reasoning amnesia. And it is an architecture problem, not a hardware problem.

### Two Mechanisms

**Mechanism 1: REPL — Active Navigation**

Zhang et al. (2025) showed that externalizing context into a REPL the model can actively navigate eliminates context rot. The deeper insight is the shift from passive to active:

```
Passive: load everything → model processes it all
Active:  model peeks → model searches → model loads only what this sub-task needs
```

The model navigates its environment. It doesn't drown in it. This remains the right approach even when context windows are large — not because of token limits, but because focused context produces better reasoning.

**Mechanism 2: CausalFrame — Temporal Persistence + Task Reuse**

The question is what to persist across sessions. Three options:

```
I/O (logs):       captures what was said, not why
                  → can't verify, can't detect staleness

REPL state:       captures what was computed, not the reasoning
                  → no provenance, model can't judge whether to trust it

Causation + Task: captures why the conclusion was reached, normalized by intent
                  → verifiable, invalidatable, navigable, reusable
```

Causation + task normalization is the only unit that supports:
- *Verification*: re-trace the chain, confirm it still holds
- *Invalidation*: when a premise changes, find all dependent conclusions
- *Navigation*: given a new question, find relevant prior reasoning
- *Reuse*: same intent → same frame (prevents duplication)

**Insight: Core enforces structure. LM decides content and policy.**

| Core enforces | LM decides |
|---------------|------------|
| token_budget | what files to include in context |
| max depth | whether to recurse or not |
| CanonicalTask + frame identity | confidence interpretation |
| invalidation cascade | escalation policy |

When in doubt: structural constraint → Core. Judgment call → LM.

---

## Architecture

Neither mechanism works without the other.

**REPL alone:** the model navigates context well within a session — but when the session ends, everything is lost. No externalization across time.

**CausalFrame alone:** reasoning history is stored — but the model still drowns in context within a session. And critically, CausalFrame creation depends on `llm()`, which depends on REPL. The mechanisms are coupled.

**Together:** the model externalizes spatially (REPL navigates current context) and temporally (CausalFrame persists the causal history of that navigation, normalized by task intent). Externalization is complete only when both dimensions are covered.

```
Claude Code
    │
    └── rlm-orchestrator (entry point)
        │
        ├── ContextMap              ← session-scoped, lazy file navigator (git ls-files + dynamic discovery)
        │   └── repl_environment.py
        │
        ├── RLAPHLoop               ← synchronous recursion via llm(sub_query)
        │   └── creates child frames + evidence links
        │
        ├── Intent Normalization    ← query → canonical_task (prevents duplication)
        │   └── frame identity = canonical_task.hash() + context_slice.hash()
        │
        ├── Causal Layer            ← persistent tree (JSONL)
        │   ├── CausalFrame         ← auto-generates structured invalidation_condition
        │   ├── FrameIndex          ← children + evidence populated, query tracking
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

### RLAPH Loop — Immediate Execution with Recursion

**Status: Fully implemented with synchronous recursion.** The `llm(sub_query)` call now creates child frames with explicit depth and returns results immediately.

```python
result_1 = llm(query_1)            # executes now, child frame created
result_2 = llm(query_2, result_1)  # result_1 available immediately
result_3 = llm(query_3, result_2)  # builds naturally
```

Recursion is explicit and guarded:
- Depth tracked per call (explicit parameter)
- Max depth enforced
- Verbose logging shows decisions

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
llm(query)                               # immediate nested LLM call (depth+1)
                                         # auto-creates CausalFrame with canonical_task
```

The LM never sees frames directly — it just calls `llm()` and gets a result. Frame management is invisible.

---

## Part 2: Causal Layer

### Intent Normalization

**Status: Implemented.** Raw query is not used for frame identity — prevents duplication from wording differences.

```python
CanonicalTask(
    task_type="analyze_module",
    target="auth.py",
    analysis_scope="correctness",
    params={}
)
```

Frame identity = hash(canonical_task + context_slice.hash)

Extraction: hybrid (rule-based fast path + LLM fallback)

This makes the system a **task-oriented reasoning cache** — same intent reuses prior frames.

### CausalFrame

**Status: Fully implemented.** `invalidation_condition` is now a structured dict.

```python
@dataclass
class CausalFrame:
    frame_id: str          # hash(canonical_task + context_slice.hash())
    depth: int
    parent_id: str | None
    children: list[str]

    query: str             # original natural language (for reference)
    canonical_task: CanonicalTask | None
    context_slice: ContextSlice
    evidence: list[str]           # only COMPLETED child frames
    conclusion: str | None
    confidence: float
    invalidation_condition: dict  # {"files": [...], "tools": [...], "description": "..."}
    status: FrameStatus
    branched_from: str | None
    escalation_reason: str | None

    created_at: datetime
    completed_at: datetime | None
```

### ContextSlice

**Status: Fully implemented with lazy loading and auto-discovery.**

### Invalidation

**Status: Fully implemented.** Downward cascade (children) works. Sideways cascade via evidence uses `find_dependent_frames` (with caching).

Structured condition enables future automation.

### Branch Management

**Status: Data model implemented.** Real resumption logic coming soon.

### FrameStore & SessionArtifacts

**Status: Implemented.** `initial_query` tracked for session continuity.

### Current Implementation Status (Feb 20, 2026 — post-Phase 18)

| Feature                        | Implemented? | Notes / Next |
|--------------------------------|--------------|--------------|
| Synchronous llm() recursion    | Yes          | llm(sub_query) creates child frames; explicit depth param |
| Tree structure (children)      | Yes          | Populated in FrameIndex.add() |
| Evidence linking               | Yes          | Only COMPLETED children added to parent.evidence |
| Intent normalization           | Yes          | canonical_task + hash-based identity (prevents duplication) |
| Structured invalidation_condition | Yes       | dict with files/tools/memory_refs |
| Cascade invalidation           | Yes          | Down + sideways via cached find_dependent_frames |
| Git-aware change detection     | Yes          | commit_hash + git diff on load |
| Query/intent tracking          | Yes          | initial_query saved in FrameIndex |
| ContextMap lazy loading        | Yes          | git ls-files + dynamic discovery inside working dir |
| Multi-session resumption       | No           | Data model ready; needs UI/skill to resume branches |
| Living docs / surgical updates | No           | Hooks ready; needs plugin to re-run invalidated frames |
| Plugin interface               | No           | Future work |

---

## Part 3: Plugin Layer (Future)

**Status: Designed but not implemented.**

### Cognitive Loop

```
INPUT arrives (git diff, PR, query)
    ↓
Intent Normalization → canonical_task
    ↓
Core resolves context (changed files, invalidated frames, suspended branches)
    ↓
transform_input() → context_slice
    ↓
llm() executes immediately → CausalFrame created
    ↓
parse_output() → structured result
    ↓
store() → plugin storage
    ↓
Core: invalidation check → cascade
```

---

## What We Rejected and Why

| Rejected | Reason |
|----------|--------|
| Deferred/async LLM stack | Error recovery unclear, frame lifecycle ambiguous |
| Subprocess for recursion | Synchronous llm(sub_query) simpler, faster, sufficient |
| DAG structure | n=10-20, O(n) scan correct |
| Weighted evidence | Prefer symbol-level granularity over tuning hell |
| Full reasoning trace | Verbose mode sufficient; adds noise |
| SQLite / MemoryBackend | JSONL per session, zero dependencies |
| Query string in frame identity | Leads to duplication/entropy; use canonical_task + context hash instead |

---

## File List

```
src/
├── rlaph_loop.py            # immediate execution loop with recursion
├── repl_environment.py      # peek, search, llm, llm_batch, map_reduce
├── llm_client.py            # provider-agnostic LLM calls
├── tool_bridge.py           # controlled tool access
├── causal_frame.py          # CausalFrame, FrameStatus, compute_frame_id
├── canonical_task.py        # Intent normalization
├── intent_extractor.py      # query → canonical_task
├── context_slice.py         # ContextSlice, hash, budget
├── frame_index.py           # dict[str, CausalFrame], branch queries, initial_query
├── frame_invalidation.py    # propagate_invalidation, find_dependent_frames
├── frame_store.py           # JSONL: save, load, list
├── session_artifacts.py     # SessionArtifacts, FileRecord
├── session_comparison.py    # compare_sessions, SessionDiff
└── plugin_interface.py      # CoreContext, RLMPlugin protocol (future)

scripts/
├── extract_frames.py        # Stop hook: extract + save
└── compare_sessions.py      # SessionStart hook: diff + surface

hooks/
└── hooks.json               # SessionStart, PostToolUse, Stop
```

**14 src files. 2 scripts. 1 hooks file.**

---

*Based on: Zhang et al., "Recursive Language Models" (2025)*
*One problem. Two mechanisms. One principle: externalize what the model cannot hold, normalize by task intent, and let it evolve with its environment.*
