# Causal Cognition Externalization: A Framework for Persistent AI Reasoning

*Draft v3 — February 2026*

---

## The Problem

Language models suffer from two forms of amnesia:

- **Within-session**: context rot — performance degrades as context grows
- **Across-session**: full reset — no continuity of understanding between conversations

Existing solutions (RAG, summarization, memory injection) treat symptoms. They store *what* the model knew, not *how* it came to know it. This produces brittle, unverifiable state.

And even within a session, users don't think linearly — they refine, pivot, backtrack. A storage model that assumes linear conversation loses the structure of actual reasoning.

---

## The Core Insight

> **What persists should not be state. It should be causality.**

There is a fundamental difference between:

```
State:    "auth module does not handle JWT expiry"

Causality: "prod logs showed 401 patterns → traced to line 47 →
            no expiry check found → reproduced in dev →
            conclusion confidence: 0.9
            invalidated if: refresh logic exists elsewhere"
```

State is a snapshot. It cannot be verified, replayed, or selectively invalidated.
Causality is a chain. It can be re-traced, challenged, and updated at the point of change.

---

## The RLM Foundation

Zhang et al. (2025) demonstrated that externalizing context — storing it outside the model's context window and letting the model *actively navigate* it — eliminates context rot and allows smaller models to outperform larger ones.

The key move: context shifts from **passive input** to **active environment**.

We extend this principle across the time axis:

| Dimension | RLM (Zhang) | This Framework |
|-----------|-------------|----------------|
| What is externalized | Large context (spatial) | Reasoning history (temporal) |
| Model's role | Navigate context actively | Navigate past cognition actively |
| Benefit | No context rot | No reasoning amnesia |

---

## Structural Insight: Execution is a Call Tree

RLM execution is recursive. This is not an implementation detail — it determines the correct storage structure.

```
frame_root (depth=0): "find auth problem"
├── frame_1 (depth=1): "analyze prod logs"
│   → evidence: "401 patterns found"
├── frame_2 (depth=1): "trace code"
│   → evidence: "line 47, no expiry check"
└── aggregates children → conclusion
```

**Consequence:** Causal storage must match this structure. A flat list of traces is wrong — it loses parent-child relationships, makes replay ambiguous, and requires a separate propagation mechanism that the tree already provides for free.

---

## The Call Frame

The unit of storage is a `CausalFrame` — one node in the call tree:

```python
@dataclass
class CausalFrame:
    # Identity
    frame_id: str
    depth: int                    # 0 = root
    parent_id: str | None         # None for root
    children: list[str]           # IDs of spawned frames

    # Reasoning
    query: str                    # What was asked of this frame
    context_slice: dict           # Context owned by THIS frame only
    evidence: list[str]           # Frame IDs + raw observations
    conclusion: str | None
    confidence: float
    invalidation_condition: str   # What would make this wrong

    # Branch management
    status: str                   # "active" | "suspended" | "completed"
    branched_from: str | None     # If pivot: which frame triggered this branch
```

The last two fields handle the full range of user behavior — linear refinement, pivots, and backtracking — without a separate data structure.

---

## Branch Management: No Separate Structure Needed

Users don't think linearly. Three patterns occur in practice:

```
Pattern A: Refinement (same direction, narrowing)
  "find auth problem" → "focus on JWT" → "just refresh tokens"
  → single frame chain, status: active throughout

Pattern B: Pivot (new direction, prior work preserved)
  "find auth problem" → ... → "wait, could be DB"
  → new frame with branched_from = prior frame
  → prior frame: status suspended (not deleted)

Pattern C: Backtrack
  "find auth problem" → ... → "start over"
  → prior frame: status suspended
  → new frame from root
```

A BranchPoint object is unnecessary. The tree already encodes this:

```
frame_root         (status: completed)
├── frame_auth     (status: suspended, branched_from: None)
└── frame_db       (status: active,    branched_from: "frame_auth")
```

Queries become trivial:

```python
# What is active now?
active    = [f for f in index.values() if f.status == "active"]

# Where did we pivot?
pivots    = [f for f in index.values() if f.branched_from]

# What can we resume?
suspended = [f for f in index.values() if f.status == "suspended"]
```

---

## Session Artifacts

What must be stored to compare sessions and detect change:

```python
@dataclass
class SessionArtifacts:
    session_id: str
    initial_prompt: str              # Why this session started
    files: dict[str, FileRecord]     # What was in scope
    root_frame_id: str               # Entry point to call tree
    conversation_log: str            # Path to raw transcript (Claude Code)

@dataclass
class FileRecord:
    path: str
    hash: str                        # For change detection
    role: str                        # "read" | "modified" | "created"
```

The raw Claude Code transcript is preserved as-is — full conversation history, tool calls, timestamps. It is the source of truth. `CausalFrame` tree is the *structure extracted from it*, not a replacement.

### Cross-Session Comparison

```
Next session starts:
  1. Compare initial_prompt    → same task or new?
  2. Compare file hashes       → what changed?
  3. Find frames referencing   → which frames are invalidated?
     changed files
  4. Check suspended frames    → any prior branches worth resuming?
```

Same files, same prompt, different branch explored → different session. The frame tree captures this distinction where a flat transcript cannot.

---

## Design Decision: No DAG

**Rejected:** Separate DAG structure for frame dependencies.

**Reason:** Scale does not justify it.

```
Actual constraints:
  Max depth:  2-3
  Max frames: 10-20 (handful of REPL environments)
  Context:    2-3M tokens (large, but distributed across frames)
```

**What replaces it:**

```python
index: dict[str, CausalFrame]   # flat lookup, always in memory
evidence: list[str]             # frame IDs + raw observations mixed
invalidate(reason, index)       # linear scan, O(n) — fine at this scale
```

**Revisit condition:** If frame count approaches 100+, introduce reverse index.
Until then, index scan is a conscious tradeoff, not an oversight.

---

## Verification and Invalidation

Because the structure is a tree, both operations emerge naturally:

```
verify()     → post-order walk   (leaves first, root last)
invalidate() → top-down cascade  (parent dies, children follow)
```

Invalidation respects branch status — suspended frames are invalidated but not deleted. They can be resumed if the invalidation condition no longer applies.

---

## Architecture

### Invariant Core

```
Core manages:
  CausalFrame tree (including status, branched_from)
  SessionArtifacts (prompt, files, hashes)
  Invalidation cascade
  Verification walk

Core does NOT manage:
  What the model sees (plugin)
  How output is parsed (plugin)
  Where output is stored (plugin)
```

### Variable Plugin Layer

```
Core (invariant)              Plugin (variable)
─────────────────             ──────────────────────────────
call tree storage        +    input transformer
session artifacts        +    output parser / storage format
branch management        +    retrieval strategy
invalidation cascade     +    domain-specific storage
```

---

## Design Principles

**1. Externalize causality, not state**
Store why the model concluded something, not just what it concluded.

**2. Storage structure matches execution structure**
RLM executes as a call tree. Storage is a call tree. Mismatch creates translation layers that obscure reasoning.

**3. The model navigates, not receives**
Past cognition is an environment the model queries, not data injected passively.

**4. Plugins own I/O, Core owns causality**
Input format and output storage change per use case. The call frame structure does not.

**5. Invalidation is first-class**
Every conclusion carries the condition under which it becomes invalid. Without this, living documents are caches, not knowledge.

**6. Branches are preserved, not deleted**
A pivot suspends a frame. Suspension is recoverable. Deletion is not.

**7. Simple at actual scale**
10-20 frames. Linear scan. No DAG. Complexity lives in context_slice distribution, not frame management.

---

## Summary

> *AI engineering is the discipline of structuring what the model perceives.*
> *For persistent AI systems, that means structuring not just current context,*
> *but the causal history of how past context was understood —*
> *and preserving the call tree through which that understanding was built,*
> *including every branch the user chose not to follow.*

State tells the model what was true.
Causality tells the model how to know if it is still true.
The call tree tells the model how it came to know it.
Branch status tells the model what was considered but set aside.

---

*Based on: Zhang et al., "Recursive Language Models" (2025)*
*Developed through: Session State Consolidation Plan, RLM Orchestrator (2026)*
*v2: Call tree insight, DAG rejection, scale constraint analysis*
*v3: Branch management via CausalFrame fields, SessionArtifacts, transcript integration*