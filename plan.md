# RLM RLAPH Refactoring Plan

## Executive Summary

Refactor the RLM orchestrator from a deferred-operation pattern to a clean **RLAPH-style loop** (Recursive Language Agent with Python Handler). This keeps the REPL (core RLM feature) while fixing the confusing multi-turn flow.

---

## Current Issues

### Issue 1: Deferred Operation Confusion

**Problem**: `llm()` returns a `DeferredOperation` object instead of the actual result.

```python
# Current behavior (confusing)
result = llm("What is 2+2?")
print(result)  # Prints: "<<DEFERRED:rq_1>>"

# User must then:
# 1. Wait for orchestrator to process
# 2. Check working_memory['rq_1'] in a separate code block
# 3. Hope the result is there
```

**Impact**:
- LLM gets confused by `<<DEFERRED:rq_1>>` output
- Multi-turn flow is hard to debug
- LLM often hallucinates results instead of waiting

### Issue 2: Missing Trajectory Methods

**Problem**: `StreamingTrajectory` class is missing methods called by `recursive_handler.py`:

| Missing Method | Called From |
|----------------|-------------|
| `emit_recursive_start` | `recursive_handler.py:190` |
| `emit_recursive_error` | `recursive_handler.py:221` |
| `emit_model_downgrade` | `recursive_handler.py:167` |

**Impact**: Recursive LLM calls fail silently or with errors.

### Issue 3: Complex Async Flow

**Problem**: The deferred operation processing requires complex async coordination:

```
REPL executes → DeferredOperation created → Code finishes →
Orchestrator detects pending ops → Processes async →
Stores in working_memory → Next REPL block can access
```

**Impact**: Hard to debug, easy to break, requires multiple code blocks.

### Issue 4: Context Bloat

**Problem**: Full REPL outputs are added to root LM context without truncation.

**Fixed**: Added `MAX_REPL_OUTPUT = 1500` truncation (lines 414-420 in core.py).

### Issue 5: LLM Response Truncation

**Problem**: `max_tokens=4096` was too small for detailed analysis.

**Fixed**: Increased to `max_tokens=16384` in orchestrator files.

---

## Goals

### Primary Goal
Make `llm()` return the **actual result** synchronously, keeping the REPL sandbox.

### Success Criteria

1. **Simple API**: `result = llm("query")` returns string immediately
2. **Depth works**: Recursive calls to depth 2-3 work correctly
3. **REPL preserved**: `peek()`, `search()`, `working_memory` still work
4. **Easy debugging**: Single loop, clear iteration history
5. **Clean code**: Remove ~500 lines of deferred operation handling

---

## Architecture Comparison

### Before (Deferred Operations)

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR LOOP                        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ LLM call │───►│ Parse        │───►│ REPL execute     │  │
│  └──────────┘    └──────────────┘    └──────────────────┘  │
│                                              │              │
│                                              ▼              │
│                                    ┌──────────────────┐    │
│                                    │ llm() creates    │    │
│                                    │ DeferredOp       │    │
│                                    └──────────────────┘    │
│                                              │              │
│                                              ▼              │
│         ┌────────────────────────────────────────────┐      │
│         │ NEXT ITERATION                             │      │
│         │ Process deferred ops async                 │      │
│         │ Store in working_memory[op_id]             │      │
│         │ LLM must check working_memory              │      │
│         └────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### After (RLAPH Loop)

```
┌─────────────────────────────────────────────────────────────┐
│                    RLAPH LOOP (Clean)                       │
│                                                             │
│  for iteration in range(max_iterations):                   │
│      ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│      │ LLM call │───►│ Parse        │───►│ Execute      │  │
│      └──────────┘    └──────────────┘    └──────────────┘  │
│           ▲                                    │            │
│           │                                    ▼            │
│           │                          ┌──────────────────┐  │
│           │                          │ llm() is SYNC    │  │
│           │                          │ Returns result   │  │
│           │                          │ immediately      │  │
│           │                          └──────────────────┘  │
│           │                                    │            │
│           └────────────────────────────────────┘            │
│                     (feed result back)                      │
│                                                             │
│  If FINAL: return answer                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Create RLAPH Loop Core

**File**: `src/rlaph_loop.py` (NEW)

```python
class RLAPHLoop:
    """Clean RLM agent loop with synchronous LLM calls."""

    def __init__(self, max_iterations=20, max_depth=3):
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.repl: RLMEnvironment = None
        self.history: list[dict] = []
        self.depth = 0

    async def run(self, query: str, context: SessionContext) -> str:
        """Main loop - clean, predictable, debuggable."""
        ...

    def llm_sync(self, query: str, context: str = "") -> str:
        """Synchronous LLM call - returns actual result."""
        ...
```

**Estimated lines**: ~150

### Phase 2: Modify REPL Environment

**File**: `src/repl_environment.py`

**Change**: Make `_recursive_query` return actual result instead of `DeferredOperation`.

```python
# Before
def _recursive_query(self, query: str, context: Any = None) -> DeferredOperation:
    op = DeferredOperation(...)
    self.pending_operations.append(op)
    return op

# After
def _recursive_query(self, query: str, context: Any = None) -> str:
    """Returns actual result synchronously."""
    if self.recursive_handler:
        return self.recursive_handler.llm_sync(query, str(context) if context else "")
    raise RuntimeError("No recursive handler available")
```

**Files to modify**:
- `src/repl_environment.py` (~20 lines changed)
- Remove `DeferredOperation` creation for `llm()`

### Phase 3: Simplify Orchestrator

**Files**: `src/orchestrator/core.py`, `src/orchestrator.py`

**Changes**:
1. Remove `_process_deferred_operations` method (~80 lines)
2. Remove `has_pending_operations()` checks
3. Remove `resolve_operation()` calls
4. Simplify REPL result handling

**Estimated removal**: ~200 lines

### Phase 4: Fix Trajectory Methods

**File**: `src/trajectory.py`

**Already fixed**:
- `emit_recursive_start` (added)
- `emit_model_downgrade` (added)

**Still needed**:
- `emit_recursive_error` (add method)
- Review all `emit_*` calls in `recursive_handler.py`

### Phase 5: Update Entry Points

**File**: `scripts/run_orchestrator.py`

**Change**: Use `RLAPHLoop` instead of `RLMOrchestrator`.

```python
# Before
from src.orchestrator import RLMOrchestrator
orchestrator = RLMOrchestrator(config=config)
async for event in orchestrator.run(query, context):
    ...

# After
from src.rlaph_loop import RLAPHLoop
loop = RLAPHLoop(max_depth=depth)
result = await loop.run(query, context)
print(result)
```

---

## File Changes Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `src/rlaph_loop.py` | CREATE | +150 |
| `src/repl_environment.py` | MODIFY | ~30 |
| `src/orchestrator/core.py` | MODIFY | -100 |
| `src/orchestrator.py` | MODIFY | -100 |
| `src/trajectory.py` | MODIFY | +20 |
| `scripts/run_orchestrator.py` | MODIFY | ~20 |

**Net change**: ~0 lines (simpler but new file)

---

## Migration Path

### Step 1: Create RLAPH Loop (Non-Breaking)
- Create `src/rlaph_loop.py`
- Keep existing orchestrator working
- Test with `--rlaph` flag

### Step 2: Add Sync LLM to REPL (Non-Breaking)
- Add `llm_sync()` to recursive handler
- Keep `DeferredOperation` path working
- Test both paths

### Step 3: Switch Default (Breaking)
- Make `llm()` use sync path by default
- Remove deferred operation processing
- Update tests

### Step 4: Cleanup
- Remove unused deferred operation code
- Simplify orchestrator
- Update documentation

---

## Testing Plan

### Unit Tests

```python
# tests/unit/test_rlaph_loop.py

def test_llm_returns_string():
    """llm() should return actual string, not DeferredOperation."""
    loop = RLAPHLoop()
    result = loop.llm_sync("What is 2+2?")
    assert isinstance(result, str)
    assert "4" in result

def test_depth_limit():
    """Should enforce max_depth."""
    loop = RLAPHLoop(max_depth=2)
    # At depth 2, should raise RecursionDepthError
    ...

def test_final_answer():
    """Should return when FINAL: is detected."""
    loop = RLAPHLoop()
    result = await loop.run("What is 2+2? Use FINAL:")
    assert result == "4"
```

### Integration Tests

```bash
# Test depth 2
uv run python scripts/run_orchestrator.py --depth 2 "Use llm() to ask what 3+3 is"

# Test depth 3
uv run python scripts/run_orchestrator.py --depth 3 "Complex analysis requiring recursion"

# Test REPL functions still work
uv run python scripts/run_orchestrator.py "Use peek() and search() to analyze files"
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing behavior | Keep old path with `--legacy` flag |
| Sync calls block event loop | Use `asyncio.run_in_executor` for blocking |
| Depth explosion | Hard limit on max_depth (default 3) |
| API rate limits | Add delays between recursive calls |

---

## Timeline

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Create RLAPH loop | 2 hours |
| 2 | Modify REPL | 1 hour |
| 3 | Simplify orchestrator | 2 hours |
| 4 | Fix trajectory | 30 min |
| 5 | Update entry points | 30 min |
| 6 | Testing | 2 hours |

**Total**: ~8 hours

---

## Success Metrics

1. `llm("query")` returns string immediately
2. Depth 2-3 tests pass
3. No `DeferredOperation` in user-visible code
4. 200+ lines removed from orchestrator
5. All existing tests pass

---

## References

- RLM Paper: https://arxiv.org/abs/2512.24601v1
- RLAPH Concept: Recursive Language Agent with Python Handler
- Claude Code Hooks: https://docs.anthropic.com/en/docs/claude-code/hooks
- Claude Code Subagents: https://docs.anthropic.com/en/docs/claude-code/subagents
