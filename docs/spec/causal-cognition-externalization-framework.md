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

----
# Plugin Architecture for Causal Cognition Externalization
## A Vision for Extensible RLM Reasoning

*Draft v2 — February 2026*

---

## The Separation

The whitepaper established one invariant:

> **The call tree must be preserved. Causality must be stored.**

Everything else is variable. *How* inputs arrive, *how* outputs are stored, *how* knowledge is rendered — these change per use case. This is the plugin boundary.

```
┌─────────────────────────────────────────────┐
│                  CORE                        │
│                                              │
│  CausalFrame tree (status, branched_from)   │
│  SessionArtifacts (prompt, files, hashes)   │
│  Invalidation cascade                        │
│  Verification (post-order walk)             │
└──────────────────┬──────────────────────────┘
                   │ stable interface
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌───────────────┐
│   Plugin A    │    │   Plugin B    │
│  Living Docs  │    │ Code Review   │
└───────────────┘    └───────────────┘
```

Core owns the call tree, branch state, and session artifacts.
Plugins own everything that touches the outside world.

---

## What Core Provides to Plugins

Before a plugin runs, Core has already resolved:

```python
# What Core gives every plugin
class CoreContext:
    current_frame: CausalFrame      # Frame being executed now
    index: dict[str, CausalFrame]   # All frames this session
    artifacts: SessionArtifacts     # Prompt, files, hashes
    changed_files: list[str]        # Files changed since last session
    invalidated_frames: list[str]   # Frames already marked invalid
    suspended_frames: list[str]     # Branches set aside, resumable
```

Plugin does not compute these. Core does. Plugin only decides what to do with them.

---

## The Plugin Interface

Three methods. No more.

```python
class RLMPlugin:

    def transform_input(
        self,
        raw_input: Any,
        ctx: CoreContext
    ) -> dict:
        """
        What does the model see?

        raw_input: whatever came from outside (git diff, PR, file, text)
        ctx:       Core-resolved context (frame, index, artifacts)

        Returns: context_slice for this frame
        """
        ...

    def parse_output(
        self,
        raw_output: str,
        frame: CausalFrame
    ) -> Any:
        """
        What do we keep from the model's response?

        Returns: structured data — must preserve invalidation_condition
        """
        ...

    def store(
        self,
        parsed: Any,
        frame: CausalFrame
    ) -> None:
        """
        Where does it go?

        Side effect only. No decisions here — those belong in parse_output.
        """
        ...
```

**Rule:** Plugins cannot modify the call tree. They read `CoreContext` but write only to their own storage. Core owns causality. Plugins observe it.

---

## Plugin Examples

### Plugin 1: Living Docs

**Problem:** Documentation goes stale. Code changes, docs don't.

**Flow:**

```
transform_input:
  raw_input = git diff
  ctx.changed_files → find affected frames via context_slice
  ctx.invalidated_frames → know what needs re-evaluation
  → return: diff + prior frame conclusions as context

parse_output:
  → parse into YAML knowledge node
  → MUST preserve invalidation_condition for next cycle

store:
  → update knowledge tree node
  → unchanged nodes untouched
  → trigger render pass → updated docs
```

**Knowledge node:**

```yaml
module: auth
purpose: "JWT-based user authentication"
dependencies: [db, cache]
public_api:
  - login(email, password) → token
  - verify(token) → user_id
invalidation_condition: "auth.py modified OR dependency interface changes"
last_verified: "2026-02-18"
frame_id: "frame_auth_root"
```

**Key property:** Knowledge and presentation decoupled. YAML is truth. Markdown/Confluence are render passes.

---

### Plugin 2: Code Review

**Problem:** Review patterns not learned across PRs. Every review starts cold.

**Flow:**

```
transform_input:
  raw_input = PR diff
  ctx.index → find past review frames for same files
  ctx.suspended_frames → any prior review branches worth resuming?
  → return: diff + "last time we reviewed auth.py, we flagged X"

parse_output:
  → structured feedback JSON
  → tagged by: severity, category, file, line range
  → invalidation_condition: "finding resolved in follow-up PR"

store:
  → write to review store (per-PR)
  → if same finding appears 3+ times → promote to memory_store as principle
```

**Key property:** Repeated findings become principles. Plugin learns without model retraining.

---

### Plugin 3: Test Generation

**Problem:** Tests written once, never updated when implementation changes.

**Flow:**

```
transform_input:
  raw_input = changed function signatures
  ctx.changed_files → which test frames are invalidated?
  ctx.invalidated_frames → don't re-run what's still valid
  → return: new signature + prior test rationale

parse_output:
  → structured test spec (rationale + cases)
  → NOT raw code — rationale is stored, code is derived

store:
  → write test spec
  → generate test code from spec (deterministic, no LLM)
  → mark old tests invalidated
```

**Key insight:** Store the *rationale* for tests, not just tests. Rationale guides regeneration when implementation changes.

---

### Plugin 4: RLM Default (Baseline)

**Problem:** Each session starts cold.

**Flow:**

```
transform_input:
  raw_input = user query
  ctx.artifacts.initial_prompt → compare to prior sessions
  ctx.suspended_frames → any prior branches relevant to this query?
  → return: query + relevant past frame conclusions

parse_output:
  → extract conclusion, evidence, reasoning path
  → wrap in CausalFrame structure

store:
  → write frame to session tree (Core handles)
  → if confidence > threshold AND verified → promote to memory_store
```

---

## How Branch State Affects Plugins

When a user pivots mid-session, Core updates frame status. Plugins see this through `CoreContext`:

```
User: "wait, could be a DB problem"
  → Core: frame_auth.status = "suspended"
  → Core: creates frame_db, branched_from = "frame_auth"
  → Plugin receives ctx.suspended_frames = ["frame_auth"]

Plugin can now:
  → ignore suspended frames (continue fresh)
  → pull suspended frame conclusions as prior context
  → surface to user: "we have a suspended auth analysis, resume?"
```

The plugin decides what to do with branch state. Core just tracks it.

---

## The Cognitive Loop

All plugins share the same underlying loop:

```
1. INPUT arrives (code change, PR, query, commit)
        ↓
2. Core resolves context
   → changed files, invalidated frames, suspended branches
        ↓
3. transform_input()
   → plugin decides what the model sees
   → context_slice for this frame
        ↓
4. MODEL executes
   → navigates context_slice actively
   → may spawn child frames (depth+1)
        ↓
5. parse_output()
   → extract structured knowledge
   → preserve invalidation_condition
        ↓
6. store()
   → write to plugin storage
        ↓
7. Core: invalidation check
   → did this frame's conclusion change?
   → cascade to dependent frames
   → update branch states if needed
```

Steps 3, 5, 6 differ per plugin. Everything else is Core.

---

## What Plugins Cannot Do

**Cannot modify the call tree.** Frame parent-child relationships, status, and branched_from are managed by Core only.

**Cannot bypass invalidation.** If Core marks a frame invalidated, the plugin's stored output is stale. Must re-run.

**Cannot access other sessions** without explicit SessionLink.

**Cannot own causality.** Plugin stores its output. Core stores why that output was produced. These are separate.

**Cannot put decisions in store().** If store() contains logic, it belongs in parse_output().

---

## Design Principles

**1. Plugin complexity lives in transform_input**
The hardest problem per plugin: what does this frame need to see? Getting context_slice right is where each plugin earns its value.

**2. parse_output must preserve invalidation_condition**
If parse_output discards invalidation_condition, the knowledge produced is a cache, not a living node. This is the most common way to get plugins wrong.

**3. store is a side effect, not a decision**
Decision of what to store is made in parse_output. store() is mechanical.

**4. Plugins are stateless across calls**
State lives in the call tree (Core) and plugin's storage backend. Plugin itself is pure transformation.

**5. Branch state is Core's responsibility**
Plugins observe branch state via CoreContext. They do not manage it.

---

## Summary

The plugin system answers one question per use case:

> *Given what Core knows (call tree, branch state, session artifacts)*
> *and what just happened (raw input),*
> *what should the model see, what should we keep, and where should it go?*

Three methods. Clear boundary. Core owns the reasoning structure including branches. Plugins own the domain.

The invariant — causal frames as a call tree with branch management — means every plugin gets verifiable, invalidation-aware, branch-aware, replayable reasoning for free. The plugin only has to answer what's domain-specific.

---

*Part of: Causal Cognition Externalization framework*
*Companion to: RLM Whitepaper v3 (February 2026)*
*v2: CoreContext interface, branch state integration, session artifact access*