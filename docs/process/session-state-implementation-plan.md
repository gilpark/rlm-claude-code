# Session State Consolidation - Implementation Plan

## Overview

This document outlines actionable implementation items derived from the [Session State Consolidation Plan](./session-state-consolidation.md). The goal is to evolve RLM's state management from flat files to a session-centric architecture.

## Current Status

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: Session Directory Infrastructure | In Progress | 10% |
| Phase 2: Cell Infrastructure | Not Started | 0% |
| Phase 3: Hook Migration | Not Started | 0% |
| Phase 4: REPL Bridge Update | Not Started | 0% |
| Phase 5: Cleanup and Deprecation | Not Started | 0% |

---

## Phase 1: Session Directory Infrastructure

### 1.1 Files to Create

| File | Purpose | Status |
|------|---------|--------|
| `src/session_schema.py` | Pydantic models for session.json | Done |
| `src/session_manager.py` | SessionManager class for session-centric operations | Pending |
| `scripts/migrate_sessions.py` | One-time migration script | Pending |
| `tests/unit/test_session_manager.py` | Unit tests for SessionManager | Pending |
| `tests/integration/test_session_migration.py` | Migration integration tests | Pending |

### 1.2 Files to Modify

| File | Changes | Status |
|------|---------|--------|
| `src/state_persistence.py` | Add SessionManager delegation | Pending |
| `scripts/capture_session_context.sh` | Create session directory | Pending |
| `scripts/init_rlm.py` | Create rlm-sessions directory | Pending |

### 1.3 Implementation Tasks

#### Task 1.1: SessionManager Class
- [ ] Create `src/session_manager.py`
- [ ] Implement `SessionManager.__init__()` with base_dir configuration
- [ ] Implement `create_session(session_id, cwd)` method
- [ ] Implement `load_session(session_id)` method
- [ ] Implement `save_session(state)` method with atomic writes
- [ ] Implement `_write_session_json()` for atomic JSON writes
- [ ] Implement `_update_current_symlink()` for active session tracking
- [ ] Implement `_init_cell_index()` for empty cell DAG
- [ ] Implement `delete_session(session_id)` method
- [ ] Implement `list_sessions()` with filtering
- [ ] Implement `get_current_session()` helper

#### Task 1.2: Migration Script
- [ ] Create `scripts/migrate_sessions.py`
- [ ] Implement dry-run mode
- [ ] Implement session discovery from flat files
- [ ] Implement migration logic for each session
- [ ] Implement backup creation
- [ ] Implement rollback capability
- [ ] Add progress reporting

#### Task 1.3: Unit Tests
- [ ] Test session creation
- [ ] Test session loading
- [ ] Test session saving
- [ ] Test atomic writes
- [ ] Test symlink management
- [ ] Test session listing
- [ ] Test session deletion
- [ ] Test error handling

### 1.4 Directory Structure Target

```
~/.claude/rlm-sessions/
├── {session_id}/
│   ├── session.json              # Unified session state
│   ├── transcript.jsonl          # SYMLINK to Claude's native transcript
│   ├── cells/
│   │   ├── index.json            # Cell DAG with dependencies
│   │   └── {cell_id}.json        # Individual cells
│   ├── reasoning/
│   │   ├── traces.jsonl          # Reasoning traces (SPEC-04)
│   │   └── decisions.jsonl       # Decision trees
│   ├── file-access.jsonl         # Per-session file access log
│   └── memory-snapshot.json      # Task-tier memory at session end
├── current -> {session_id}/      # Symlink to active session
└── archive/
    └── {old_session_id}/         # Archived sessions
```

---

## Phase 2: Cell Infrastructure

### 2.1 Files to Create

| File | Purpose |
|------|---------|
| `src/cell_manager.py` | Cell creation, dependency resolution, replay |

### 2.2 Files to Modify

| File | Changes |
|------|---------|
| `src/repl_environment.py` | Create cells for REPL operations |
| `src/recursive_handler.py` | Create cells for recursive calls |
| `src/session_manager.py` | Add cell management methods |

### 2.3 Implementation Tasks

#### Task 2.1: CellManager Class
- [ ] Create `src/cell_manager.py`
- [ ] Implement `CellManager.__init__(cells_dir, session_id)`
- [ ] Implement `generate_cell_id()` with pattern `cell_{hex8}`
- [ ] Implement `create_cell()` with type, input, dependencies
- [ ] Implement `complete_cell()` with result/error
- [ ] Implement `get_cell(cell_id)` loader
- [ ] Implement `get_execution_order()` for topological sort
- [ ] Implement `_recompute_execution_order()` with cycle detection
- [ ] Implement `validate_dependencies()` with cycle check
- [ ] Implement `_would_create_cycle()` helper
- [ ] Implement `get_dependency_chain()` for transitive deps
- [ ] Implement `replay_cell()` for re-execution

#### Task 2.2: Cell DAG Validation
- [ ] Add self-dependency check
- [ ] Add missing dependency check
- [ ] Add cycle detection algorithm
- [ ] Add cycle marking in index

#### Task 2.3: Cell Tests
- [ ] Test cell creation
- [ ] Test dependency tracking
- [ ] Test cycle detection
- [ ] Test execution order computation
- [ ] Test cell replay

---

## Phase 3: Hook Migration

### 3.1 Files to Modify

| File | Changes |
|------|---------|
| `scripts/capture_session_context.sh` | Use SessionManager.create_session() |
| `scripts/sync_context.py` | Use SessionManager.save_session() |
| `scripts/capture_output.py` | Use SessionManager, create cells |
| `scripts/track_file_access.sh` | Write to session directory |
| `scripts/externalize_context.py` | Use session directory |
| `scripts/save_trajectory.py` | Update to new structure |
| `scripts/repl_bridge.py` | Load from session.json |

### 3.2 Implementation Tasks

#### Task 3.1: SessionStart Hook
- [ ] Update `capture_session_context.sh` to call Python init
- [ ] Create `init_session.py` to replace bash logic
- [ ] Create session directory structure
- [ ] Initialize session.json
- [ ] Update 'current' symlink

#### Task 3.2: PreToolUse Hook
- [ ] Update `sync_context.py` to use SessionManager
- [ ] Load session via `load_session()`
- [ ] Update context in session.json
- [ ] Create pending cell if applicable

#### Task 3.3: PostToolUse Hook
- [ ] Update `capture_output.py` to use SessionManager
- [ ] Complete pending cell
- [ ] Update tool_outputs in session.json
- [ ] Update `track_file_access.sh` path

#### Task 3.4: PreCompact Hook
- [ ] Update `externalize_context.py`
- [ ] Create snapshots in session directory

#### Task 3.5: Stop Hook
- [ ] Update `save_trajectory.py`
- [ ] Set `ended_at` in session.json
- [ ] Archive if configured

---

## Phase 4: REPL Bridge Update

### 4.1 Files to Modify

| File | Changes |
|------|---------|
| `scripts/repl_bridge.py` | Load from session.json, support cell queries |

### 4.2 New REPL Operations

#### Task 4.1: Session Operations
- [ ] Implement `op_session()` - Get current session info
- [ ] Implement `op_cells()` - Query cells in session
- [ ] Implement `op_cell()` - Get specific cell details
- [ ] Implement `op_dependency_chain()` - Get cell dependencies

#### Task 4.2: Replay Operations
- [ ] Implement `op_replay()` - Replay specified cells
- [ ] Implement `op_invalidate()` - Invalidate cells on file change
- [ ] Implement `op_promote_to_memory()` - Promote verified output to DB

---

## Phase 5: Cleanup and Deprecation

### 5.1 Deprecation Path

1. **Week 5**: Add deprecation warnings to `state_persistence.py` flat-file methods
2. **Week 6**: Update all documentation to reference new paths
3. **Week 7**: Remove flat-file support (keep migration script)

### 5.2 Files to Deprecate/Remove

| File | Action |
|------|--------|
| `src/state_persistence.py` | Keep but deprecate direct file access methods |
| `scripts/capture_session_context.sh` | Keep but minimize logic, delegate to Python |
| `scripts/track_file_access.sh` | Keep, only path changes |

### 5.3 Documentation Updates

| Document | Changes |
|----------|---------|
| `CLAUDE.md` | Update paths, add cell reference |
| `docs/user-guide.md` | Update state file locations |
| `docs/process/architecture.md` | Add ADR for session-centric state |
| `README.md` | Update directory structure diagram |

---

## Critical Recommendations (from Feasibility Analysis)

| Priority | Recommendation | Rationale |
|----------|----------------|-----------|
| **HIGH** | Extend `StatePersistence`, don't replace | Preserves tested atomic write logic |
| **HIGH** | Add cell support to `RLMEnvironment` | Current REPL has no cell concept |
| **HIGH** | Use SPEC-04 hyperedges for dependencies | Align with existing `membership` table |
| **HIGH** | Add cycle detection to cell DAG | Prevent circular dependencies |
| **MEDIUM** | Batch cell index updates | Write on session save, not every cell |
| **MEDIUM** | Add rollback to migration | Backup + revert script for safety |
| **MEDIUM** | Reuse existing types from `types.py` | `ExecutionResult`, `SessionContext`, etc. |
| **LOW** | Integrate `memory-snapshot.json` with SPEC-02 | Query task-tier nodes by session_id |

---

## Performance Targets

| Operation | Target | Max |
|-----------|--------|-----|
| Create session | <10ms | 50ms |
| Load session | <20ms | 100ms |
| Save session | <30ms | 150ms |
| Create cell | <5ms | 20ms |
| Query cells | <10ms | 50ms |

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data loss during migration | Low | Critical | Dry-run mode, backups, idempotent migration |
| Performance regression | Medium | Medium | Benchmark before/after, lazy loading |
| Hook timing issues | Medium | High | Keep hooks fast, async cell creation |
| Disk space increase | High | Low | Compression, archival policy |

---

## Success Criteria

### Phase 1 Complete When
- [ ] SessionManager can create/load/save sessions
- [ ] Migration script successfully migrates existing sessions
- [ ] All tests pass

### Phase 2 Complete When
- [ ] Cells can be created with dependencies
- [ ] Cell DAG can be queried and traversed
- [ ] Cell replay works for simple cases

### Phase 3 Complete When
- [ ] All hooks use SessionManager
- [ ] No writes to old flat files
- [ ] Integration tests pass

### Phase 4 Complete When
- [ ] REPL bridge uses session.json
- [ ] Cell queries work from REPL
- [ ] All REPL operations functional

### Phase 5 Complete When
- [ ] Legacy code removed
- [ ] Documentation updated
- [ ] No deprecation warnings in normal use

---

## Next Actions

1. **Implement SessionManager class** (`src/session_manager.py`)
2. **Create migration script** (`scripts/migrate_sessions.py`)
3. **Write unit tests** (`tests/unit/test_session_manager.py`)
4. **Run migration in dry-run mode** to verify
5. **Update hooks** to use SessionManager
