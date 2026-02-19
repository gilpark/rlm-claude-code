# RLM-Claude-Code: Capabilities Specification

## Overview

This specification defines capabilities for the RLM-Claude-Code project, covering both the original recurse port and strategic enhancements based on research synthesis.

## Scope

### Phase 1: Core Capabilities (Complete)

| Priority | Component | Spec Document | Status |
|----------|-----------|---------------|--------|
| P0 | Advanced REPL Functions | [SPEC-01](./01-repl-functions.md) | ✅ Complete |
| P1 | Memory Foundation | [SPEC-02](./02-memory-foundation.md) | ✅ Complete |
| P2 | Memory Evolution | [SPEC-03](./03-memory-evolution.md) | ✅ Complete |
| P3 | Reasoning Traces | [SPEC-04](./04-reasoning-traces.md) | ✅ Complete |
| P4 | Enhanced Budget Tracking | [SPEC-05](./05-budget-tracking.md) | ✅ Complete |

### Phase 2: Strategic Enhancements (Complete)

| Priority | Component | Spec Document | Status |
|----------|-----------|---------------|--------|
| P0 | Smarter RLM | [SPEC-06](./06-smarter-rlm.md) | ✅ Complete |
| P1 | Intelligence Enhancements | [SPEC-07](./07-intelligence.md) | ✅ Complete |
| P1 | Performance Optimizations | [SPEC-08](./08-performance.md) | ✅ Complete |
| P2 | Capability Enhancements | [SPEC-09](./09-capabilities.md) | ✅ Complete |
| P2 | Reliability Improvements | [SPEC-10](./10-reliability.md) | ✅ Complete |
| P3 | User Experience | [SPEC-11](./11-user-experience.md) | ✅ Complete |
| P3 | Architecture Refactoring | [SPEC-12](./12-architecture.md) | ✅ Complete |

### Phase 3: User Experience & Verification (Mostly Complete)

| Priority | Component | Spec Document | Status |
|----------|-----------|---------------|--------|
| P0 | Rich Output Formatting | [SPEC-13](./13-rich-output.md) | ✅ Complete |
| P0 | Always-On Micro-RLM | [SPEC-14](./14-always-on-rlm.md) | ✅ Complete |
| P2 | Lean REPL Integration | [SPEC-15](./15-lean-repl.md) | ⏸️ Deferred |
| P0 | Epistemic Verification | [SPEC-16](./16-epistemic-verification.md) | ✅ Complete |

## Dependencies

```
Phase 1 (Foundation):
SPEC-01 (REPL Functions) ─────────────────────────► Independent
SPEC-02 (Memory Foundation) ──────────────────────► Independent
SPEC-03 (Memory Evolution) ───► SPEC-02 (Memory Foundation)
SPEC-04 (Reasoning Traces) ───► SPEC-02 (Memory Foundation)
SPEC-05 (Budget Tracking) ────────────────────────► Independent

Phase 2 (Enhancements):
SPEC-06 (Smarter RLM) ────► SPEC-01, SPEC-02, SPEC-05
SPEC-07 (Intelligence) ───► SPEC-01, SPEC-05
SPEC-08 (Performance) ────► SPEC-05
SPEC-09 (Capabilities) ───► SPEC-02, SPEC-03
SPEC-10 (Reliability) ────► SPEC-05
SPEC-11 (User Experience) ► SPEC-04, SPEC-06
SPEC-12 (Architecture) ───► All prior specs

Phase 3 (UX & Verification):
SPEC-13 (Rich Output) ────► SPEC-04, SPEC-05
SPEC-14 (Always-On) ──────► SPEC-05, SPEC-06, SPEC-13
SPEC-15 (Lean REPL) ──────► SPEC-01, SPEC-13, SPEC-14 [DEFERRED]
SPEC-16 (Epistemic) ──────► SPEC-01, SPEC-04, SPEC-05

Phase 4 (Causal Frames):
SPEC-17 (Causal Frames) ──► SPEC-02, SPEC-04
```

## Success Criteria

[SPEC-00.01] The system SHALL support all capabilities defined in SPEC-01 through SPEC-05.

[SPEC-00.02] All new capabilities SHALL maintain backward compatibility with existing RLM functionality.

[SPEC-00.03] All new capabilities SHALL have comprehensive test coverage (unit, integration, property tests).

[SPEC-00.04] Performance SHALL NOT degrade by more than 10% for existing operations.

[SPEC-00.05] Phase 2 enhancements SHALL demonstrate measurable improvements:
- Intelligence: 2x success rate improvement on complex tasks
- Performance: 3x latency reduction, 50% cost reduction
- Reliability: 100% adherence to execution guarantees
- User Experience: 50% reduction in user overrides

## Implementation Roadmap

### Phase A: Quick Wins (Complete)
- ✅ SPEC-08.01-08.06: Async recursive calls
- ✅ SPEC-08.10-08.15: Prompt caching
- ✅ SPEC-10.10-10.15: Execution guarantees
- ✅ SPEC-11.01-11.06: Progressive trajectory

### Phase B: Core Improvements (Complete)
- ✅ SPEC-07.10-07.15: Compute-optimal allocation
- ✅ SPEC-09.01-09.07: Embedding-based retrieval
- ✅ SPEC-10.01-10.06: Confidence-weighted synthesis
- ✅ SPEC-12.01-12.07: Modularize orchestrator

### Phase C: Advanced Capabilities (Complete)
- ✅ SPEC-07.01-07.07: ToT integration
- ✅ SPEC-09.20-09.26: Checkpointing
- ✅ SPEC-07.20-07.25: Formal verification
- ✅ SPEC-11.10-11.16: Interactive steering

### Phase D: Learning & Evolution (Complete)
- ✅ SPEC-06.40-06.53: Continuous learning
- ✅ SPEC-09.10-09.15: Cross-session promotion
- ✅ SPEC-11.20-11.25: Learning from corrections
- ✅ SPEC-12.10-12.16: REPL plugin system

### Phase 3A: Rich Output (Complete)
- ✅ SPEC-13.01-13.05: Visual language system
- ✅ SPEC-13.10-13.14: Rich Console integration
- ✅ SPEC-13.20-13.25: Progress and budget display
- ✅ SPEC-13.30-13.33: Depth visualization
- ✅ SPEC-13.40-13.44: Error display
- ✅ SPEC-13.50-13.52: Configuration

### Phase 3B: Always-On Micro-RLM (Complete)
- ✅ SPEC-14.01-14.05: Micro execution mode
- ✅ SPEC-14.10-14.15: Default activation
- ✅ SPEC-14.20-14.25: Progressive escalation
- ✅ SPEC-14.30-14.34: Definitely simple fast path
- ✅ SPEC-14.40-14.44: Context externalization
- ✅ SPEC-14.50-14.55: Memory system integration
- ✅ SPEC-14.60-14.65: Cost management

### Phase 3C: Lean REPL (Deferred)
- ⏸️ SPEC-15.01-15.05: Lean REPL functions
- ⏸️ SPEC-15.10-15.15: Process management
- ⏸️ SPEC-15.20-15.24: JSON protocol
- ⏸️ SPEC-15.30-15.34: Proof state visualization
- ⏸️ SPEC-15.40-15.45: Auto-formalization
- ⏸️ SPEC-15.50-15.54: RLM integration
- ⏸️ SPEC-15.60-15.65: Installation and setup

### Phase 3D: Epistemic Verification (Complete)
- ✅ SPEC-16.01-16.10: Core types & REPL functions
- ✅ SPEC-16.11-16.20: Reasoning traces integration
- ✅ SPEC-16.21-16.30: Orchestrator integration (always-on)
- ✅ SPEC-16.31-16.40: Polish & optimization

### Phase 4: Causal Frames (Core Complete, Integration Pending)

| Priority | Component | Spec Document | Status |
|----------|-----------|---------------|--------|
| P0 | Causal Frame Core | [SPEC-17](./17-causal-frames.md) | ✅ Complete |
| P1 | Frame Index & Lifecycle | [SPEC-17](./17-causal-frames.md) | ✅ Complete |
| P1 | Frame Invalidation | [SPEC-17](./17-causal-frames.md) | ✅ Complete |
| P2 | Session Artifacts | [SPEC-17](./17-causal-frames.md) | ✅ Complete |
| P2 | Plugin Architecture | [SPEC-17](./17-causal-frames.md) | ✅ Complete |
| P1 | RLAPH Integration | — | ⏳ Pending |
| P2 | Session Manager Integration | — | ⏳ Pending |

**Key Changes**:
- Tree-structured `CausalFrame` replacing flat `CausalTrace`
- Propagation control with cascade invalidation
- Plugin architecture for custom RLM behaviors
- "LM decides, Core records" design principle

See [ADR-011](../process/architecture.md) for architecture decision.

### Phase 4: RLAPH Refactoring (Complete - 2026-02)

| Priority | Component | Status |
|----------|-----------|--------|
| P0 | RLAPH Loop Core | ✅ Complete |
| P0 | Synchronous llm() | ✅ Complete |
| P1 | Context Truncation | ✅ Complete |
| P1 | Legacy CLI Removal | ✅ Complete |
| P2 | Response Parser Fix | ✅ Complete |
| P2 | Unit Tests | ✅ Complete (15 tests) |

**Key Changes**:
- `llm()` now returns actual string immediately (not `DeferredOperation`)
- Clean `RLAPHLoop` class with synchronous `llm_sync()`
- Context truncation in API client
- `--legacy` flag for backward compatibility
- `/loop` slash command for TDD-style task execution

See [ADR-010](../process/architecture.md) for architecture decision.

## References

- [RECOMMENDATIONS.md](../RECOMMENDATIONS.md) - Full analysis and research basis
- [RLM Paper](https://arxiv.org/abs/2512.24601) - Foundational research
