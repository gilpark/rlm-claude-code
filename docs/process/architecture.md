# Architecture Decision Records

This document captures architectural decisions for RLM-Claude-Code. Each decision follows the format: Context → Decision → Consequences.

## ADR Index

| ID | Decision | Status | Date |
|----|----------|--------|------|
| ADR-001 | Plugin over Fork | Accepted | 2025-01 |
| ADR-002 | Python REPL over JavaScript | Accepted | 2025-01 |
| ADR-003 | Complexity-based Activation | Accepted | 2025-01 |
| ADR-004 | Depth=2 Default | Accepted | 2025-01 |
| ADR-005 | Streaming Trajectory | Accepted | 2025-01 |
| ADR-006 | Extended Python Tooling | Accepted | 2025-01 |
| ADR-007 | CPMpy for Constraint Verification | Accepted | 2025-01 |
| ADR-008 | Intelligent Orchestration Layer | Accepted | 2026-01 |
| ADR-009 | Strategy Learning from Trajectories | Accepted | 2026-01 |

---

## ADR-001: Plugin Architecture over Fork

**Status**: Accepted

**Context**: We need to add RLM capabilities to Claude Code. Options:
1. Fork Claude Code and modify directly
2. Build as a plugin using Claude Code's extension system
3. Build a wrapper/proxy layer

**Decision**: Build as a Claude Code plugin.

**Consequences**:
- ✅ Receive automatic updates from Anthropic
- ✅ Community can extend independently
- ✅ Core tool execution, permissions, hooks preserved
- ⚠️ Limited to plugin API capabilities
- ⚠️ Must sync state between RLM layer and Claude Code core

**Spec Reference**: §2.1, §5.1

---

## ADR-002: Python REPL over JavaScript

**Status**: Accepted

**Context**: RLM requires a REPL for context manipulation. Options:
1. Python REPL (as in original RLM paper)
2. JavaScript REPL (native to Node.js)
3. Custom DSL

**Decision**: Python REPL with RestrictedPython sandbox.

**Consequences**:
- ✅ Proven patterns from RLM paper
- ✅ Better string manipulation for context processing
- ✅ Claude stronger at Python code generation
- ✅ Rich ecosystem (pydantic, hypothesis)
- ⚠️ Additional runtime dependency
- ⚠️ Cross-process communication overhead

**Spec Reference**: §4.1, §4.1.1

---

## ADR-003: Complexity-based Activation

**Status**: Accepted

**Context**: When should RLM mode activate? Options:
1. Token count threshold (original RLM paper approach)
2. Task complexity analysis
3. Always on
4. Manual only

**Decision**: Complexity-based activation with bias toward RLM.

**Consequences**:
- ✅ Activates when reasoning benefit is highest
- ✅ Avoids overhead on simple queries
- ✅ More intelligent than token threshold
- ⚠️ Classifier must be fast (<50ms)
- ⚠️ Risk of false negatives (missing complex tasks)
- Mitigation: Bias toward activation (95%+ recall target)

**Spec Reference**: §6.3

---

## ADR-004: Default Depth=2

**Status**: Accepted

**Context**: How deep should recursive calls go? Options:
1. Depth=1 (paper default)
2. Depth=2 (verification chains)
3. Unlimited (dangerous)

**Decision**: Default depth=2, configurable to 3.

---

## ADR-010: RLAPH Loop over Deferred Operations

**Status**: Accepted

**Context**: The deferred operation pattern was confusing:
- `llm()` returned `DeferredOperation` not actual result
- LLM got confused by `<<DEFERRED:rq_1>>` output
- Multi-turn flow was hard to debug
- Complex async coordination required

**Decision**: Switch to RLAPH-style loop with synchronous `llm()`.

**Consequences**:
- ✅ `llm()` returns actual string immediately
- ✅ Simpler code, easier to debug
- ✅ Single clean loop with clear history
- ⚠️ Legacy code kept for backward compatibility (`--legacy` flag)
- ⚠️ Thread-based approach for nested event loops

**Implementation**:
- `src/rlaph_loop.py` - Clean RLAPH loop
- `src/recursive_handler.py` - Added `llm_sync()` method
- `src/repl_environment.py` - Returns actual result, not DeferredOperation

**Spec Reference**: Phase 6 - RLAPH Refactoring

**Consequences**:
- ✅ Enables Root → Analysis → Verification pattern
- ✅ Supports constraint-driven verification workflows
- ✅ Cost manageable with model cascade (Opus→Sonnet→Haiku)
- ⚠️ 3-5x cost of depth=1
- ⚠️ Higher latency for deep queries

**Model Cascade**:
| Depth | Model | Cost/1K tokens |
|-------|-------|----------------|
| 0 | Opus 4.5 | $15/$75 |
| 1 | Sonnet 4 | $3/$15 |
| 2 | Haiku 4.5 | $0.25/$1.25 |

**Spec Reference**: §6.4

---

## ADR-005: Streaming Trajectory Visibility

**Status**: Accepted

**Context**: How should RLM reasoning be presented to users? Options:
1. Hidden (black box)
2. Summary after completion
3. Streaming as it happens
4. Configurable verbosity

**Decision**: Streaming with configurable verbosity (minimal/normal/verbose/debug).

**Consequences**:
- ✅ Users can reason about RLM behavior in real-time
- ✅ Debugging is possible
- ✅ Builds trust through transparency
- ✅ JSON export for analysis
- ⚠️ Terminal rendering complexity
- ⚠️ Output can be noisy at high verbosity

**Spec Reference**: §6.6

---

## ADR-006: Extended Python Tooling in REPL

**Status**: Accepted

**Context**: What tools should be available in the REPL? Options:
1. Minimal (stdlib only)
2. RLM-specific helpers only
3. Production Python tooling

**Decision**: Include uv, ty, ruff, pydantic, and hypothesis.

**Consequences**:
- ✅ Type checking extracted code with ty
- ✅ Linting code before suggesting edits with ruff
- ✅ Schema validation for structured context with pydantic
- ✅ Property-based verification with hypothesis
- ✅ Fast package management with uv
- ⚠️ Larger environment footprint
- ⚠️ Security surface for subprocess calls
- Mitigation: Subprocess allowlist (ty, ruff only)

**Spec Reference**: §4.1.1

---

## ADR-007: CPMpy for Constraint Verification

**Status**: Accepted

**Context**: RLM verification at depth=2 needs to reason about invariants, dependencies, and constraints extracted from code context. Options:
1. Natural language reasoning only
2. Z3 SMT solver directly
3. CPMpy constraint programming library
4. Custom constraint DSL

**Decision**: Include CPMpy in REPL tooling for constraint-driven verification.

**Consequences**:
- ✅ Solver-agnostic: uses OR-Tools by default, can swap to Z3, Gurobi
- ✅ Numpy-based API aligns with data manipulation patterns
- ✅ High-level constraints (AllDifferent, Cumulative, etc.)
- ✅ Incremental solving for iterative verification
- ✅ Aligns with Ananke's constraint-driven code generation philosophy
- ⚠️ Learning curve for constraint modeling
- ⚠️ Solver performance varies by problem structure
- Mitigation: Provide helper functions for common patterns

**Use Cases**:
| Pattern | CPMpy Application |
|---------|-------------------|
| Dependency graphs | Model as precedence constraints |
| Type compatibility | Encode subtyping as logical constraints |
| Resource bounds | Cumulative constraints |
| State machines | Transition constraints |

**Spec Reference**: §4.1.1

---

## ADR-008: Intelligent Orchestration Layer

**Status**: Accepted

**Context**: RLM decisions (model selection, depth budget, tool access) were heuristic-only. Options:
1. Keep heuristic-only routing
2. Use Claude as intelligent orchestrator
3. Train separate orchestrator model
4. Hybrid approach

**Decision**: Claude Haiku as intelligent orchestrator with heuristic fallback.

**Consequences**:
- ✅ Intelligent model selection based on task analysis
- ✅ Fast decisions (~200ms with Haiku)
- ✅ Graceful fallback when API unavailable
- ✅ User-configurable preferences (fast/balanced/thorough)
- ⚠️ Additional API call per query
- ⚠️ Orchestrator accuracy depends on prompt quality
- Mitigation: Cache decisions for similar queries

**Components**:
| Module | Purpose |
|--------|---------|
| orchestration_schema.py | Plan and context types |
| intelligent_orchestrator.py | Claude-powered decisions |
| user_preferences.py | User preference management |
| auto_activation.py | Automatic RLM activation |

**Spec Reference**: §8.1 Phase 2

---

## ADR-009: Strategy Learning from Trajectories

**Status**: Accepted

**Context**: Successful RLM trajectories contain strategy patterns that could inform future queries. Options:
1. No learning (stateless)
2. Rule-based strategy detection
3. Embedding-based similarity matching
4. Full ML training loop

**Decision**: Feature-based similarity matching with strategy cache.

**Consequences**:
- ✅ Learn from successful trajectories
- ✅ Suggest strategies for similar queries
- ✅ No ML training required
- ✅ Interpretable features and decisions
- ⚠️ Cold start problem (empty cache)
- ⚠️ Feature engineering required
- Mitigation: Sensible defaults when cache empty

**Strategy Types**:
| Strategy | Pattern |
|----------|---------|
| Peeking | Sample context before deep processing |
| Grepping | Regex/pattern-based search |
| Partition+Map | Divide context, process via sub-calls |
| Programmatic | One-shot code execution |
| Recursive | Spawn sub-queries for complex reasoning |

**Components**:
| Module | Purpose |
|--------|---------|
| trajectory_analysis.py | Strategy extraction from events |
| strategy_cache.py | Similarity-based caching |
| tool_bridge.py | Controlled tool access for sub-LLMs |

**Spec Reference**: §8.1 Phase 3

---

## Pending Decisions

### ADR-010: REPL State Isolation at Depth=2

**Status**: Proposed

**Context**: When depth=1 spawns a child REPL, should it share state?

**Options**:
1. Fully isolated (no shared state)
2. Read-only access to parent state
3. Copy-on-write semantics

**Considerations**:
- Isolation simplifies reasoning about behavior
- Sharing enables richer verification patterns
- Memory overhead of multiple interpreters

**Recommendation**: Fully isolated with explicit context passing.

---

### ADR-011: Tool Interleaving Strategy

**Status**: Proposed

**Context**: When RLM decides to use a Claude Code tool (bash, edit), how should control flow work?

**Options**:
1. Yield control to Claude Code, resume RLM after
2. RLM orchestrates tool use through its own handler
3. Hybrid based on tool type

**Considerations**:
- Option A preserves Claude Code's permission model
- Option B provides more control but duplicates logic
- Permission checks must happen regardless

**Recommendation**: Option A (yield to Claude Code).

---

## Decision Template

```markdown
## ADR-XXX: [Title]

**Status**: Proposed | Accepted | Deprecated | Superseded

**Context**: [What is the issue? What are the options?]

**Decision**: [What was decided?]

**Consequences**:
- ✅ [Positive consequence]
- ⚠️ [Negative consequence or risk]
- Mitigation: [How we address the risk]

**Spec Reference**: §X.Y
```
