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