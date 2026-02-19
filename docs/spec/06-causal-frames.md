# SPEC-06: Causal Frames

*February 2026 — v4 with Propagation Control*

---

## Overview

Tree-structured frames for RLM execution with propagation control.

**Key Insight:** The real danger in RLM isn't hallucination — it's **unchecked propagation**. A wrong result at depth-1 becomes evidence at depth-0. CausalFrame's `evidence` + `invalidation_condition` + `invalidate()` cascade provides correction-after-detection.

---

## Core Concepts

### Design Decisions

1. **Flat trace → Call frame tree** — RLM execution is a call tree, storage must match
2. **No DAG structure** — At 10-20 frames, O(n) scan is correct
3. **No BranchPoint dataclass** — `status` + `branched_from` fields suffice
4. **LM decides, Core records** — Context slicing and confidence escalation are LM decisions

### Scale Constraints

```
Max frames:    10-20
Max depth:     2-3
Context size:  2-3M tokens per frame slice
```

---

## Data Structures

### FrameStatus

```python
class FrameStatus(Enum):
    CREATED     = "created"      # Frame created, not yet running
    RUNNING     = "running"      # Actively executing
    COMPLETED   = "completed"    # Finished execution
    VERIFIED    = "verified"     # Output verified
    PROMOTED    = "promoted"     # Promoted to memory_store
    INVALIDATED = "invalidated"  # Cascade invalidation
    SUSPENDED   = "suspended"    # Low confidence / needs human
    UNCERTAIN   = "uncertain"    # Propagated uncertainty
```

### CausalFrame

```python
@dataclass
class CausalFrame:
    frame_id: str                 # Deterministic hash
    depth: int                    # 0 = root
    parent_id: str | None
    children: list[str]
    query: str
    context_slice: ContextSlice
    evidence: list[str]
    conclusion: str | None
    confidence: float
    invalidation_condition: str
    status: FrameStatus
    branched_from: str | None
    escalation_reason: str | None
    created_at: datetime
    completed_at: datetime | None
```

### ContextSlice

```python
@dataclass
class ContextSlice:
    files: dict[str, str]         # file_path → content_hash
    memory_refs: list[str]
    tool_outputs: dict[str, str]
    token_budget: int
```

---

## Core Functions

### compute_frame_id

```python
def compute_frame_id(
    parent_id: str | None,
    query: str,
    context_slice: ContextSlice
) -> str:
    """Deterministic frame ID for caching."""
    content = f"{parent_id}:{query}:{context_slice.hash()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### propagate_invalidation

```python
def propagate_invalidation(
    frame_id: str,
    reason: str,
    index: FrameIndex
) -> list[str]:
    """Invalidate frame and all dependents."""
    # CASCADE DOWN to children
    # SCAN UP + SIDEWAYS for evidence dependencies
```

### FrameIndex Queries

```python
index.get_active_frames()    # status == RUNNING
index.get_suspended_frames() # status == SUSPENDED
index.get_pivots()           # branched_from != None
```

---

## Propagation Control

### Context Partitioning

**Q1 RESOLVED:** Core enforces `token_budget` only. Root LM chooses what files via tool calls:
- `peek(file, lines)` — preview content
- `grep(pattern, scope)` — search files
- `spawn_child(query, files, budget)` — create child frame

### Confidence Escalation

**Q2 RESOLVED:** Core records `confidence`, Root LM decides escalation via tool response.

SUSPENDED/UNCERTAIN statuses available for explicit Root LM use.

---

## Plugin Architecture

### CoreContext

```python
class CoreContext(TypedDict):
    current_frame: CausalFrame | None
    index: dict[str, CausalFrame]
    artifacts: SessionArtifacts | None
    changed_files: list[str]
    invalidated_frames: list[str]
    suspended_frames: list[str]
    confidence_threshold: float
```

### RLMPlugin Protocol

```python
class RLMPlugin(Protocol):
    def transform_input(self, raw_input, ctx) -> dict | PluginError
    def parse_output(self, raw_output, frame) -> Any | PluginError
    def store(self, parsed, frame) -> None
```

---

## Files

| File | Purpose |
|------|---------|
| `src/causal_frame.py` | CausalFrame, FrameStatus, compute_frame_id |
| `src/context_slice.py` | ContextSlice, hash() |
| `src/frame_index.py` | FrameIndex with branch queries |
| `src/frame_lifecycle.py` | State transitions |
| `src/frame_invalidation.py` | propagate_invalidation |
| `src/frame_serialization.py` | JSON serialization |
| `src/session_artifacts.py` | SessionArtifacts, FileRecord |
| `src/session_comparison.py` | compare_sessions |
| `src/plugin_interface.py` | CoreContext, RLMPlugin |
| `src/plugin_registry.py` | Plugin management |
| `src/plugins/default_plugin.py` | Default plugin |

---

## What We Rejected

| Rejected | Why |
|----------|-----|
| DAG structure | n=10-20, O(n) scan is correct |
| Global CausalGraphManager | Breaks session isolation |
| BranchPoint dataclass | Two frame fields sufficient |
| Separate validation layer | invalidation_condition encodes verifiability |
| Majority voting | 3x cost at 2-3M context = impractical |

---

*Based on: Zhang et al., "Recursive Language Models" (2025)*
