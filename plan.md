# Refactoring Plan: RLM-Claude-Code v2

*Generated: 2026-02-19*

---

## Executive Summary

The v2 design is **dramatically simpler**: REPL environment + CausalFrame persistence. The current codebase has ~90 files; the target is **12 src files**.

**Key insight:** Remove all orchestrators, routers, classifiers, vector search, async stacks, and complex telemetry. Focus on LM-driven navigation and frame-based persistence.

| Metric | Count |
|--------|-------|
| Files to keep | 15 |
| Files to refactor | 7 |
| Files to remove | ~82 |
| Files to create | 5 |

---

## Part 1: Files to KEEP (15)

These files align with the v2 design and need minimal changes:

```
src/
├── __init__.py           # Package init
├── config.py             # Configuration
├── prompts.py            # Prompt templates
├── types.py              # Type definitions
├── repl_plugin.py        # REPL plugin integration
├── rlaph_loop.py         # Immediate execution loop ✓
├── repl_environment.py   # peek, search, llm, llm_batch ✓
├── tool_bridge.py        # Controlled tool access ✓
├── causal_frame.py       # CausalFrame, FrameStatus ✓
├── context_slice.py      # ContextSlice, hash, budget ✓
├── frame_index.py        # dict[str, CausalFrame] ✓
├── frame_invalidation.py # propagate_invalidation ✓
├── session_artifacts.py  # SessionArtifacts, FileRecord ✓
├── session_comparison.py # compare_sessions, SessionDiff ✓
└── plugin_interface.py   # CoreContext, RLMPlugin protocol ✓
```

---

## Part 2: Files to REFACTOR (7)

These files exist but need updates to match the design spec:

### src/causal_frame.py
- **Reason:** Core data structure - must match design spec exactly
- **Changes:**
  - Verify CausalFrame has: `frame_id`, `depth`, `parent_id`, `children`
  - Add: `query`, `context_slice`, `evidence`, `conclusion`, `confidence`
  - Add: `invalidation_condition`, `status`, `branched_from`
  - Add: `created_at`, `completed_at`

### src/frame_index.py
- **Reason:** Should be simple in-memory index, not vector search
- **Changes:**
  - Implement flat dict scan of 10-20 frames
  - Remove any embedding/vector logic
  - Add branch queries: `active`, `suspended`, `pivots`

### src/frame_invalidation.py
- **Reason:** Core invalidation logic for file change detection
- **Changes:**
  - Implement cascade invalidation (children + sideways evidence)
  - Keep O(n) at n=10-20

### src/context_slice.py
- **Reason:** LM decides context, not core enforcement
- **Changes:**
  - Structure: `files`, `memory_refs`, `tool_outputs`, `token_budget`
  - Core enforces `token_budget`; LM decides what files to include

### src/session_artifacts.py
- **Reason:** Session artifact management
- **Changes:**
  - Structure: `session_id`, `initial_prompt`, `files`, `root_frame_id`, `conversation_log`
  - FileRecord: `path`, `hash`, `role`

### src/session_comparison.py
- **Reason:** Session comparison utility
- **Changes:**
  - Implement `compare_sessions(current, prior, index)`
  - Output: same task?, changed files → invalidated frames, suspended branches

### src/plugin_interface.py
- **Reason:** Plugin system interface
- **Changes:**
  - Simplify to: `transform_input`, `parse_output`, `store`
  - CoreContext: `current_frame`, `index`, `artifacts`, `changed_files`, `invalidated_frames`, `suspended_frames`

---

## Part 3: Files to REMOVE (~82)

### Category: Deferred/Async LLM Stack (REJECTED by design)

```
src/api_client.py
src/async_executor.py
src/async_handler.py
src/recursive_handler.py
```

**Reason:** Design mandates immediate execution. Deferred/async makes frame lifecycle ambiguous.

### Category: DAG Structure (REJECTED - n=10-20, O(n) scan is correct)

```
src/cell_manager.py
src/circuit_breaker.py
src/compute_allocation.py
src/frame_lifecycle.py
src/frame_serialization.py
src/proactive_computation.py
```

**Reason:** At 10-20 frames, linear scan is instant. DAG adds complexity without benefit.

### Category: Complexity Classifier/Routing (REJECTED - LM navigates in REPL)

```
src/complexity_classifier.py
src/intelligent_orchestrator.py
src/lats_orchestration.py
src/learned_routing.py
src/router_integration.py
src/setfit_classifier.py
src/smart_router.py
```

**Reason:** The model navigates in REPL. No routing needed.

### Category: Vector/Embedding Search (REJECTED - 10-20 frames, flat scan is instant)

```
src/context_index.py
src/embedding_retrieval.py
```

**Reason:** Vector search is overkill at this scale.

### Category: Facts/Experiences Store (REJECTED - PROMOTED CausalFrame is the fact)

```
src/cross_session_promotion.py
src/memory_evolution.py
src/memory_store.py
```

**Reason:** A PROMOTED frame IS the fact. No separate store needed.

### Category: SQLite/MemoryBackend Abstraction (REJECTED - JSONL per session)

```
src/memory_backend.py
```

**Reason:** JSONL per session requires zero dependencies and is human-readable.

### Category: Trajectory/Event Log (REJECTED - FrameStore is clearer)

```
src/trajectory.py
src/trajectory_analysis.py
src/progressive_trajectory.py
```

**Reason:** FrameStore provides cleaner persistence.

### Category: SessionLink (REJECTED - undefined concept)

```
src/session_manager.py
src/session_schema.py
```

**Reason:** Concept removed from v2 design.

### Category: Core-Enforced Logic (REJECTED - LM decides)

```
src/confidence_synthesis.py      # Majority voting
src/context_compression.py       # Context slicing
src/context_enrichment.py        # Context slicing
src/execution_guarantees.py      # Guarantees
```

**Reason:** Structural constraints → Core. Judgment calls → LM.

### Category: Orchestrator Components (REJECTED - not in v2 design)

```
src/local_orchestrator.py
src/orchestration_logger.py
src/orchestration_schema.py
src/orchestration_telemetry.py
src/orchestrator/               # entire directory
```

**Reason:** v2 uses immediate RLAPH loop, not complex orchestration.

### Category: Epistemic Verification (REJECTED - not in v2 design)

```
src/epistemic/                  # entire directory
```

**Reason:** Simplified for v2.

### Category: Complex Plugin System (REJECTED - simplify to basic hooks)

```
src/plugin_registry.py
src/plugins/                    # entire directory
```

**Reason:** Basic hooks are sufficient.

### Category: Not in Target File List

```
src/auto_activation.py
src/cache.py
src/context_manager.py
src/continuous_learning.py
src/cost_tracker.py
src/enhanced_budget.py
src/formal_verification.py
src/gliner_extractor.py
src/learning.py
src/progress.py
src/prompt_caching.py
src/prompt_optimizer.py
src/reasoning_traces.py
src/response_parser.py
src/rich_output.py
src/smart_pipeline.py
src/state_persistence.py
src/strategy_cache.py
src/tokenization.py
src/transcript_parser.py
src/tree_of_thoughts.py
src/user_corrections.py
src/user_preferences.py
src/visualization.py
```

**Reason:** Not required for v2 core functionality.

---

## Part 4: Files to CREATE (5)

### src/llm_client.py
```python
class LLMClient:
    def call(
        self,
        query: str,
        context: dict,
        model: str | None = None   # None = use default for this depth
    ) -> str: ...
```
- Provider-agnostic LLM calls
- Default model cascade: root uses larger model, sub-calls use smaller

### src/frame_store.py
```python
class FrameStore:
    path: Path  # ~/.claude/rlm-frames/{root_session_id}.jsonl

    def save(frame: CausalFrame) -> None         # json.dumps + append
    def load(frame_id: str) -> CausalFrame       # readlines + match
    def list() -> list[CausalFrame]              # readlines + parse
    def find_by_status(status) -> list           # list + filter
```
- JSONL per session
- Zero dependencies
- Human-readable

### scripts/extract_frames.py
- Hook: Stop
- Extract frame tree from session
- Save to FrameStore

### scripts/compare_sessions.py
- Hook: SessionStart
- Compare current vs prior session
- Surface: changed files → invalidated frames → suspended branches

### hooks/hooks.json
```json
{
  "SessionStart": "...",
  "PostToolUse": "...",
  "Stop": "..."
}
```
- Where REPL and Causal Layer connect

---

## Part 5: Target Architecture

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

## Execution Order

### Phase 1: Create Missing Core Files
1. `src/llm_client.py` - Simple synchronous client
2. `src/frame_store.py` - JSONL persistence

### Phase 2: Refactor Existing Core Files
1. `src/causal_frame.py` - Match design spec exactly
2. `src/frame_index.py` - Remove vector logic, add branch queries
3. `src/context_slice.py` - Simplify structure
4. `src/frame_invalidation.py` - Implement cascade
5. `src/session_artifacts.py` - Match design spec
6. `src/session_comparison.py` - Implement diff logic
7. `src/plugin_interface.py` - Simplify to hooks

### Phase 3: Create Hooks
1. `hooks/hooks.json` - Configure hooks
2. `scripts/extract_frames.py` - Stop hook
3. `scripts/compare_sessions.py` - SessionStart hook

### Phase 4: Remove Obsolete Files
- Delete all files in Part 3
- Update imports in remaining files
- Verify no broken dependencies

### Phase 5: Integration Testing
- Verify REPL functions work
- Verify frame creation/persistence
- Verify invalidation cascade
- Verify session comparison

---

## Notes

- **Scale:** 10-20 frames at depth 2-3. Linear scan is instant.
- **Principle:** Externalize what the model cannot hold, let it navigate actively.
- **Core vs LM:** Core enforces structure (budget, depth). LM decides content and policy.

---

## Part 6: Feasibility Analysis

### Overall Assessment

| Metric | Value |
|--------|-------|
| **Overall Feasibility** | HIGH |
| **Critical Blockers** | None identified |
| **Risk Level** | Low-Medium |

### Section-by-Section Analysis

#### Part 1: Files to KEEP
- **Feasibility:** HIGH
- **Effort:** LOW
- **Strategy:** These 15 files already align with v2 design. Minimal changes needed - mostly imports and minor API adjustments.
- **Risks:**
  - Some files may have hidden dependencies on removed modules
  - Import statements may need updating after deletions

#### Part 2: Files to REFACTOR
- **Feasibility:** HIGH
- **Effort:** MEDIUM

| File | Current State | Gaps | Strategy |
|------|---------------|------|----------|
| `causal_frame.py` | Has CausalFrame with parent_id, frame_id, status, content, metadata, children, timestamp | May need to remove complex metadata; ensure FrameStatus matches simplified spec | Simplify to core fields, remove vector embeddings |
| `frame_index.py` | Has FrameIndex class managing storage | May use vector search or complex indexing | Refactor to simple dict[str, CausalFrame] |
| `frame_invalidation.py` | Has propagate_invalidation function | Logic may be tied to old frame structure | Verify works with simplified CausalFrame |
| `context_slice.py` | Has ContextSlice with hash, budget | Budget calculation may need adjustment | Ensure matches v2 spec: hash_id, content, budget |
| `session_artifacts.py` | Has SessionArtifacts, FileRecord | May need new artifact storage model | Update to v2 artifact spec |
| `session_comparison.py` | Has compare_sessions, SessionDiff | Comparison logic may need adjustment | Verify works with simplified CausalFrame |
| `plugin_interface.py` | Defines CoreContext, RLMPlugin | Protocol may need v2 API updates | Review and update definitions |

#### Part 3: Files to REMOVE
- **Feasibility:** HIGH
- **Effort:** MEDIUM
- **Strategy:** Delete in phases:
  1. Remove orchestrators, routers, classifiers first
  2. Remove vector search and telemetry
  3. Remove async stack remnants
  4. Clean up test files for removed modules
- **Risks:**
  - Hidden dependencies may cause import errors
  - Some 'removed' files may actually be needed
  - Test files may reference deleted modules

#### Part 4: Files to CREATE
- **Feasibility:** HIGH
- **Effort:** MEDIUM

| File | Strategy |
|------|----------|
| `llm_client.py` | Simple synchronous client for REPL |
| `frame_store.py` | Implement JSONL frame persistence |
| `extract_frames.py` | Hook script for frame extraction |
| `compare_sessions.py` | Hook script for session diff |
| `hooks.json` | Hook configuration |

---

## Part 7: Critical Path & Recommended Sequence

### Critical Path (in order)
1. Create `llm_client.py` - simple synchronous client
2. Refactor `causal_frame.py` - simplified structure
3. Refactor `frame_index.py` - dict-based storage
4. Create `frame_store.py` - frame serialization
5. Refactor `context_slice.py` - match v2 spec
6. Update `rlaph_loop.py` - use new components
7. Remove old orchestrator and router files
8. Clean up imports and run tests

### Revised Execution Sequence

**Phase 1: Foundation (Low Risk)**
1. Create `src/llm_client.py`
2. Create `src/frame_store.py`
3. Update `src/types.py` if needed

**Phase 2: Core Refactor (Medium Risk)**
1. Refactor `src/causal_frame.py` to simplified structure
2. Refactor `src/frame_index.py` to dict-based storage
3. Refactor `src/context_slice.py` to match v2 spec
4. Update `src/frame_invalidation.py` for new frame structure

**Phase 3: Integration (Medium Risk)**
1. Refactor `src/session_artifacts.py`
2. Refactor `src/session_comparison.py`
3. Refactor `src/plugin_interface.py`
4. Update `src/rlaph_loop.py` to use new components
5. Update `src/repl_environment.py` if needed

**Phase 4: Hooks (Low Risk)**
1. Create `hooks/hooks.json`
2. Create `scripts/extract_frames.py`
3. Create `scripts/compare_sessions.py`

**Phase 5: Cleanup (Medium Risk)**
1. Remove files in batches:
   - Batch A: Orchestrators (`src/orchestrator/`, `src/local_orchestrator.py`, etc.)
   - Batch B: Routers (`src/smart_router.py`, `src/complexity_classifier.py`, etc.)
   - Batch C: Vector search (`src/embedding_retrieval.py`, `src/context_index.py`)
   - Batch D: Async stack (`src/async_*.py`, `src/recursive_handler.py`)
   - Batch E: Telemetry (`src/orchestration_telemetry.py`, etc.)
   - Batch F: Memory stores (`src/memory_*.py`)
   - Batch G: Epistemic (`src/epistemic/`)
   - Batch H: Remaining unused files
2. Fix broken imports after each batch
3. Run tests after each batch

**Phase 6: Verification**
1. Run all tests
2. Verify REPL functions work
3. Verify frame creation/persistence
4. Verify invalidation cascade
5. Verify session comparison

---

## Raw Feasibility JSON

```json
{
  "sections": {
    "keep": {
      "feasibility": "high",
      "strategy": "These 15 files already align with v2 design. Minimal changes needed.",
      "risks": ["Hidden dependencies on removed modules", "Import statements may need updating"],
      "effort": "low"
    },
    "refactor": {
      "feasibility": "high",
      "strategy": "The 7 files are already well-structured. Main work is simplification.",
      "effort": "medium"
    },
    "remove": {
      "feasibility": "high",
      "strategy": "Delete in phases with testing between batches.",
      "risks": ["Hidden dependencies", "Some files may be needed", "Test references"],
      "effort": "medium"
    },
    "create": {
      "feasibility": "high",
      "strategy": "Create foundation files first, then hooks.",
      "effort": "medium"
    }
  },
  "overall": {
    "feasibility": "high",
    "critical_path": ["causal_frame.py", "frame_index.py", "frame_store.py", "rlaph_loop.py"],
    "blockers": ["None identified"],
    "recommended_sequence": [
      "Start with low-risk file creation",
      "Refactor core data structures",
      "Build new functionality",
      "Update integration points",
      "Remove deprecated files incrementally",
      "Final testing and cleanup"
    ]
  }
}
```

---

*Based on: Zhang et al., "Recursive Language Models" (2025)*
*Design doc: docs/plans/2026-02-19-design.md*
*Whitepaper: docs/plans/2026-02-19-whitepaper.md*
