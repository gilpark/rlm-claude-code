# Session State Consolidation Plan

## Executive Summary

This document outlines a phased plan to evolve RLM's state management from flat files to a session-centric architecture with unified state, replayable cells, and per-session isolation.

---

## Feasibility Analysis (RLM Orchestrator Review)

**Status**: Technically sound with refinements needed

### Critical Recommendations

| Priority | Recommendation | Rationale |
|----------|----------------|-----------|
| **HIGH** | Extend `StatePersistence`, don't replace | Preserves tested atomic write logic and session lifecycle |
| **HIGH** | Add cell support to `RLMEnvironment` | Current REPL has no cell concept, uses `history` list |
| **HIGH** | Use SPEC-04 hyperedges for dependencies | Align with existing `membership` table model |
| **HIGH** | Add cycle detection to cell DAG | Prevent circular dependencies |
| **MEDIUM** | Batch cell index updates | Write on session save, not every cell creation |
| **MEDIUM** | Add rollback to migration | Backup + revert script for safety |
| **MEDIUM** | Reuse existing types from `types.py` | `ExecutionResult`, `SessionContext`, `Message`, `ToolOutput` |
| **LOW** | Integrate `memory-snapshot.json` with SPEC-02 | Query task-tier nodes by session_id |

### Performance Validation

| Operation | Target | Current | Status |
|-----------|--------|---------|--------|
| Load session | <20ms | ~5-10ms | ✅ Achievable |
| Save session | <30ms | ~8-15ms | ✅ Achievable |
| Create cell | <5ms | ~1ms | ✅ Achievable |
| Query cells | <10ms | ~2ms | ✅ Achievable |

**Bottleneck**: Cell index rewriting on every creation - batch updates recommended.

### SPEC Alignment

| Spec | Integration Point | Action Needed |
|------|-------------------|---------------|
| SPEC-02 (memory_store) | `memory-snapshot.json` | Query task-tier by session_id, not separate storage |
| SPEC-04 (reasoning_traces) | Cell dependencies | Use hyperedges (`membership` table) instead of separate DAG |

### Missing from Original Plan

1. **Cycle detection** for cell dependencies
2. **Rollback script** for migration failures
3. **`session-metadata.json` handling** - must merge `transcript_path`, `cwd` fields
4. **Tool outputs truncation** - implement sliding window with configurable size

---

## 1. Current State Analysis

### 1.1 Existing State Files

The current implementation stores state in multiple flat files under `~/.claude/rlm-state/`:

| File | Purpose | Producer | Consumer |
|------|---------|----------|----------|
| `context.json` | Conversation, files, tool_outputs, working_memory | sync_context.py, capture_output.py | repl_bridge.py |
| `{session_id}.json` | RLMSessionState (activation, depth, tokens) | state_persistence.py | state_persistence.py |
| `{session_id}_context.json` | SessionContext (messages, files, tool_outputs) | state_persistence.py | state_persistence.py |
| `session-metadata.json` | session_id, transcript_path, cwd, started_at | capture_session_context.sh | track_file_access.sh, repl_bridge.py |
| `file-access-{session_id}.jsonl` | File read/write tracking | track_file_access.sh | (future analysis) |
| `current-transcript.jsonl` | Symlink to Claude transcript | capture_session_context.sh | repl_bridge.py (op_conversation) |

### 1.2 Additional State Locations

| Location | Purpose | Format |
|----------|---------|--------|
| `~/.claude/rlm-memory.db` | Persistent hypergraph memory | SQLite with WAL |
| `~/.claude/rlm-config.json` | RLM configuration | JSON |
| `~/.claude/rlm-externalized/{session_id}/` | Pre-compaction context backups | JSON files with timestamps |
| `~/.claude/rlm-trajectories/` | Session summaries and trajectories | JSON files |

### 1.3 Data Flow Diagram

```
                          ┌─────────────────────────────────────────────────────────┐
                          │                    Claude Code Host                      │
                          │                                                          │
                          │  SessionStart ──────► capture_session_context.sh         │
                          │        │                       │                          │
                          │        │                       ▼                          │
                          │        │              session-metadata.json              │
                          │        │                       │                          │
                          │        ▼                       ▼                          │
                          │  UserPromptSubmit ──► check_complexity.py                │
                          │        │                                                  │
                          │        ▼                                                  │
                          │  PreToolUse ────────► sync_context.py                    │
                          │        │                       │                          │
                          │        │                       ▼                          │
                          │        │           state_persistence.py                  │
                          │        │           {session_id}.json                      │
                          │        │           {session_id}_context.json              │
                          │        │           context.json ◄─────────────────────────┤
                          │        │                       │                          │
                          │        ▼                       ▼                          │
                          │  Tool Execution ──────► [Tool runs]                       │
                          │        │                                                  │
                          │        ▼                                                  │
                          │  PostToolUse ────────► capture_output.py                  │
                          │        │                       │                          │
                          │        │                       ▼                          │
                          │        │           state_persistence.py (add_tool_output)│
                          │        │           context.json (update)                  │
                          │        │                       │                          │
                          │        │                       ▼                          │
                          │        │           track_file_access.sh                   │
                          │        │           file-access-{session_id}.jsonl         │
                          │        │                                                  │
                          │        ▼                                                  │
                          │  PreCompact ─────────► externalize_context.py             │
                          │        │                       │                          │
                          │        │                       ▼                          │
                          │        │           rlm-externalized/{session_id}/         │
                          │        │                                                  │
                          │        ▼                                                  │
                          │  Stop ───────────────► save_trajectory.py                 │
                          │                                │                          │
                          │                                ▼                          │
                          │                    rlm-trajectories/{session_id}_summary  │
                          │                                                          │
                          └─────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                          ┌─────────────────────────────────────────────────────────┐
                          │                   repl_bridge.py                         │
                          │                                                          │
                          │  Reads: context.json, session-metadata.json              │
                          │  Reads: rlm-memory.db (via memory_query)                 │
                          │  Operations: peek, search, summarize, llm, map_reduce    │
                          │                                                          │
                          └─────────────────────────────────────────────────────────┘
```

### 1.4 Problems with Current Architecture

1. **State Fragmentation**: Related data spread across 5+ files per session
2. **Redundant Storage**: SessionContext stored in both `{session_id}_context.json` and `context.json`
3. **No Cell Concept**: No replayable computation units for debugging
4. **Complex Recovery**: Restoring session requires loading multiple files in correct order
5. **No Dependency Tracking**: Cannot trace which data influenced which decisions
6. **Symlink Fragility**: `current-transcript.jsonl` symlink can break

---

## 2. Target Architecture

### 2.1 Key Insight: Claude's Native Transcript

Claude Code already captures transcripts at:
```
~/.claude/projects/{project_path_hash}/{session_id}.jsonl
```

For example:
```
~/.claude/projects/-Users-gilpark--dotfiles--claude-plugins-marketplaces-rlm-claude-core-rand/949d76fb-5d94-4a37-8bf1-ee850e6992ec.jsonl
```

**Design Decision**: Use Claude's native transcript as source of truth for conversation history. Do NOT duplicate messages in session.json.

### 2.2 Directory Structure

```
~/.claude/rlm-sessions/
├── {session_id}/
│   ├── session.json              # Unified session state (metadata + activation + context + budget)
│   ├── transcript.jsonl          # SYMLINK to Claude's native transcript
│   ├── cells/
│   │   ├── index.json            # Cell DAG with dependencies
│   │   ├── {cell_id_1}.json      # Individual cell (input + output + metadata)
│   │   ├── {cell_id_2}.json
│   │   └── ...
│   ├── reasoning/
│   │   ├── traces.jsonl          # Reasoning traces (SPEC-04)
│   │   └── decisions.jsonl       # Decision trees with evidence
│   ├── file-access.jsonl         # Per-session file access log
│   └── memory-snapshot.json      # Task-tier memory at session end
├── current -> {session_id}/      # Symlink to active session
└── archive/
    └── {old_session_id}/         # Archived sessions (compressed)
```

### 2.3 Session Discovery Flow

```python
# How to find the transcript for a session
def get_transcript_path(session_id: str, project_cwd: str) -> Path:
    """Get Claude's native transcript path for a session."""
    # Claude uses a hashed project path
    import hashlib
    project_hash = hashlib.md5(project_cwd.encode()).hexdigest()
    # Or Claude may use sanitized path with dashes
    sanitized = project_cwd.replace("/", "-").replace(" ", "-")
    return Path.home() / ".claude" / "projects" / sanitized / f"{session_id}.jsonl"
```

### 2.4 Unified session.json Schema

Note: Conversation history is NOT stored here - it's in the symlinked transcript.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "metadata": {
      "type": "object",
      "properties": {
        "session_id": {"type": "string"},
        "created_at": {"type": "number"},
        "updated_at": {"type": "number"},
        "ended_at": {"type": ["number", "null"]},
        "cwd": {"type": "string"},
        "claude_transcript_path": {"type": "string", "description": "Path to Claude's native transcript"},
        "rlm_version": {"type": "string"}
      },
      "required": ["session_id", "created_at", "updated_at", "cwd"]
    },
    "activation": {
      "type": "object",
      "properties": {
        "rlm_active": {"type": "boolean"},
        "activation_mode": {"type": "string", "enum": ["complexity", "manual", "always", "never"]},
        "activation_reason": {"type": ["string", "null"]},
        "complexity_score": {"type": ["number", "null"]},
        "current_depth": {"type": "integer", "minimum": 0},
        "max_depth": {"type": "integer", "minimum": 1, "maximum": 3}
      },
      "required": ["rlm_active", "activation_mode", "current_depth"]
    },
    "context": {
      "type": "object",
      "description": "Derived context - conversation is in transcript symlink",
      "properties": {
        "files": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "hash": {"type": "string"},
              "size_bytes": {"type": "integer"},
              "first_access": {"type": "number"},
              "last_access": {"type": "number"}
            }
          }
        },
        "tool_outputs": {
          "type": "array",
          "description": "Recent tool outputs for context",
          "items": {
            "type": "object",
            "properties": {
              "tool_name": {"type": "string"},
              "content_preview": {"type": "string", "maxLength": 1000},
              "exit_code": {"type": ["integer", "null"]},
              "timestamp": {"type": "number"},
              "cell_id": {"type": ["string", "null"]}
            }
          },
          "maxItems": 100
        },
        "working_memory": {
          "type": "object",
          "additionalProperties": true
        }
      }
    },
    "budget": {
      "type": "object",
      "properties": {
        "total_tokens_used": {"type": "integer"},
        "total_recursive_calls": {"type": "integer"},
        "max_recursive_calls": {"type": "integer"},
        "cost_usd": {"type": "number"},
        "by_model": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "calls": {"type": "integer"},
              "tokens": {"type": "integer"},
              "cost_usd": {"type": "number"}
            }
          }
        }
      }
    },
    "cells": {
      "type": "object",
      "properties": {
        "count": {"type": "integer"},
        "index_path": {"type": "string"}
      }
    },
    "trajectory": {
      "type": "object",
      "properties": {
        "events_count": {"type": "integer"},
        "export_path": {"type": ["string", "null"]}
      }
    }
  },
  "required": ["metadata", "activation", "context", "budget"]
}
```

### 2.3 Cell Schema

Each cell represents a replayable computation unit inspired by Jupyter notebooks. The goal is to turn recursive LLM invocations and REPL operations into modular, self-contained units that can be logged, replayed, and selectively updated.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "cell_id": {"type": "string", "pattern": "^cell_[a-z0-9]{8}$"},
    "created_at": {"type": "number"},
    "type": {
      "type": "string",
      "enum": ["repl", "tool", "llm_call", "map_reduce", "verification"]
    },
    "input": {
      "type": "object",
      "description": "Minimal relevant context snapshot - only what's needed for this cell",
      "properties": {
        "source": {"type": "string"},
        "operation": {"type": "string"},
        "args": {"type": "object"},
        "context_snapshot": {
          "type": "object",
          "description": "Tailored input slice to avoid token bloat",
          "properties": {
            "files": {"type": "object", "additionalProperties": {"type": "string"}},
            "conversation_slice": {"type": "array"},
            "memory_query": {"type": "string"}
          }
        }
      }
    },
    "code": {
      "type": "string",
      "description": "The Python/LLM/tool logic to execute (for replay)"
    },
    "output": {
      "type": "object",
      "properties": {
        "result": {"type": ["object", "string", "number", "boolean", "null"]},
        "error": {"type": ["string", "null"]},
        "execution_time_ms": {"type": "number"},
        "cached_response": {
          "type": ["string", "null"],
          "description": "Exact LLM response for non-determinism handling"
        }
      }
    },
    "dependencies": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Cell IDs this cell depends on (explicit DAG tracking)"
    },
    "dependents": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Cell IDs that depend on this cell (populated on index build)"
    },
    "trace": {
      "type": "object",
      "description": "Decision trace from trajectory.py",
      "properties": {
        "goal": {"type": "string"},
        "decision": {"type": "string"},
        "rationale": {"type": "string"}
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "depth": {"type": "integer"},
        "model": {"type": ["string", "null"]},
        "tokens_used": {"type": "integer"},
        "temperature": {"type": "number", "description": "Pin to 0 for reproducibility"},
        "promoted_to_db": {"type": "boolean", "description": "Whether output was saved to memory"}
      }
    }
  },
  "required": ["cell_id", "created_at", "type", "input", "output"]
}
```

### 2.4 Cell DAG Index (cells/index.json)

```json
{
  "version": "1.0",
  "session_id": "abc123",
  "cells": {
    "cell_a1b2c3d4": {
      "type": "repl",
      "dependencies": [],
      "dependents": ["cell_e5f6g7h8"]
    },
    "cell_e5f6g7h8": {
      "type": "llm_call",
      "dependencies": ["cell_a1b2c3d4"],
      "dependents": ["cell_i9j0k1l2"]
    }
  },
  "roots": ["cell_a1b2c3d4"],
  "leaves": ["cell_i9j0k1l2"],
  "execution_order": ["cell_a1b2c3d4", "cell_e5f6g7h8", "cell_i9j0k1l2"]
}
```

### 2.5 Good vs Bad Patterns

This comparison guides usage and helps avoid common pitfalls:

| Aspect | Good Pattern | Bad Pattern |
|--------|--------------|-------------|
| **Cell Granularity** | Medium-sized: One logical sub-task per cell (e.g., "search auth" or "synthesize fixes"). Keeps them re-runnable and focused. | Too fine (every line as a cell) → overhead bloat. Too coarse (whole recursion in one) → loses modularity, hard to replay partially. |
| **Dependencies** | Explicit DAG: Track what cells feed into others (e.g., cell 3 uses output of 1 & 2). Use graph tools for auto-invalidation on changes. | Implicit/no tracking: Leads to stale replays if upstream changes aren't propagated. |
| **Input Capture** | Snapshot minimal relevant context: Only what's needed for that cell (e.g., file snippet, not whole project). Saves storage/tokens. | Dump everything: Bloats logs, makes replays inefficient, risks token limits in children. |
| **Aggregation** | Parent uses `map_reduce()` or structured synthesis: Combines child outputs with clear reasoning trace. Promotes aggregates to DB if useful. | Ad-hoc merging: Skips traces → loses "why" behind final output, harder to debug. |
| **Replay Usage** | Selective: Re-run only affected cells on changes (e.g., after code update). Cache aggressively for unchanged paths. | Full re-run always: Wastes tokens, defeats the purpose. Or never replay: Loses continuity. |
| **DB Integration** | Promote after verification: Save cell outputs as facts/experiences only if accurate (e.g., post-replay). Query DB to enrich cell inputs. | Blind promotion: Pollutes DB with noisy/unverified data. Or ignore DB: Misses cross-session learning. |
| **Error Handling** | Cells include retry logic (e.g., if LLM fails, re-run with better prompt). Log errors in metadata for easy debugging. | Ignore failures: Propagates bad outputs downstream. No logging → black-box issues. |
| **LLM Non-determinism** | Cache exact responses or pin params (temperature=0). Store `cached_response` for reproducibility. | Always re-run LLM calls: Different outputs each time → unreliable replays. |

### 2.6 Cell Decomposition & Aggregation Flow

```
Parent Orchestrator
        │
        ▼ Decomposes task
┌───────────────────────────────────────────────────────────────┐
│                    CHILD CELLS (Parallel)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │ Cell 1      │  │ Cell 2      │  │ Cell 3              │   │
│  │ Type: repl  │  │ Type: llm   │  │ Type: tool          │   │
│  │ Op: search  │  │ Query: auth │  │ Op: memory_query    │   │
│  │ Context:    │  │ Context:    │  │ Context:            │   │
│  │   files:{}  │  │   search    │  │   memory_store      │   │
│  │   (snippet) │  │   result    │  │   query             │   │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘   │
│         │                │                    │               │
│         └────────────────┴────────────────────┘               │
│                          │                                    │
│                          ▼ Tailored input slices               │
│              (avoid token bloat per child)                    │
└───────────────────────────────────────────────────────────────┘
                           │
                           ▼ Aggregation
┌───────────────────────────────────────────────────────────────┐
│                    PARENT CELL (map_reduce)                    │
│  Combines child outputs with clear reasoning trace             │
│  Optional: Promote verified aggregates to DB                   │
└───────────────────────────────────────────────────────────────┘
```

### 2.7 Replay & Selective Update

On reload (next session or refinement):

1. **Load cells from file** → Build dependency DAG
2. **Detect changes** via hooks (PostToolUse detects file edits)
3. **Invalidate dependent cells** → Mark cells whose inputs changed
4. **Re-run only invalidated cells** → Reuse cached outputs for unchanged paths
5. **Promote good outputs to DB** → Save distilled insights after verification

Example flow:
```
Edit: auth.py modified
  │
  ▼
Invalidate: cells that depend on auth.py (Cell 2, Cell 5)
  │
  ▼
Re-run: Cell 2 → Cell 5 (Cell 1, 3, 4 unchanged → use cache)
  │
  ▼
Save: ~80% token savings vs full re-run
```

---

## 3. Implementation Phases

### Phase 1: Session Directory Infrastructure (Week 1-2)

> **Revised Estimate**: Extended from Week 1 to Week 1-2 (Risk: Medium)
> **Key Change**: Extend `StatePersistence` class instead of creating separate `SessionManager`

**Goal**: Create the session directory structure and migrate metadata.

#### 3.1.1 Files to Create

| File | Purpose |
|------|---------|
| `src/session_manager.py` | New SessionManager class for session-centric operations |
| `src/session_schema.py` | Pydantic models for session.json schema |
| `scripts/migrate_sessions.py` | One-time migration script |

#### 3.1.2 Files to Modify

| File | Changes |
|------|---------|
| `src/state_persistence.py` | Add SessionManager delegation, deprecate flat-file methods |
| `scripts/capture_session_context.sh` | Create session directory instead of just metadata file |
| `scripts/init_rlm.py` | Create rlm-sessions directory structure |

#### 3.1.3 Implementation Details

```python
# src/session_manager.py

from pathlib import Path
from typing import Any
from pydantic import BaseModel
import json
import time
import shutil

class SessionMetadata(BaseModel):
    session_id: str
    created_at: float
    updated_at: float
    ended_at: float | None = None
    cwd: str
    claude_transcript_path: str | None = None
    rlm_version: str = "0.1.0"

class SessionActivation(BaseModel):
    rlm_active: bool = False
    activation_mode: str = "complexity"
    activation_reason: str | None = None
    complexity_score: float | None = None
    current_depth: int = 0
    max_depth: int = 2

class SessionBudget(BaseModel):
    total_tokens_used: int = 0
    total_recursive_calls: int = 0
    max_recursive_calls: int = 10
    cost_usd: float = 0.0
    by_model: dict[str, dict[str, float]] = {}

class SessionState(BaseModel):
    metadata: SessionMetadata
    activation: SessionActivation
    context: dict[str, Any]  # Uses SessionContext internally
    budget: SessionBudget
    cells: dict[str, Any] = {"count": 0, "index_path": "cells/index.json"}
    trajectory: dict[str, Any] = {"events_count": 0, "export_path": None}

class SessionManager:
    """
    Manages session-centric state in ~/.claude/rlm-sessions/{session_id}/

    Implements: Session State Consolidation Plan
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (Path.home() / ".claude" / "rlm-sessions")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._current_session: SessionState | None = None
        self._current_session_dir: Path | None = None

    def create_session(self, session_id: str, cwd: str) -> SessionState:
        """Create a new session directory and initialize session.json."""
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (session_dir / "cells").mkdir(exist_ok=True)
        (session_dir / "reasoning").mkdir(exist_ok=True)

        now = time.time()
        state = SessionState(
            metadata=SessionMetadata(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                cwd=cwd,
            ),
            activation=SessionActivation(),
            context={
                "conversation": [],
                "files": {},
                "tool_outputs": [],
                "working_memory": {},
            },
            budget=SessionBudget(),
        )

        # Write initial session.json
        self._write_session_json(session_dir, state)

        # Update current symlink
        self._update_current_symlink(session_id)

        # Initialize empty cell index
        self._init_cell_index(session_dir)

        self._current_session = state
        self._current_session_dir = session_dir
        return state

    def load_session(self, session_id: str) -> SessionState:
        """Load session state from disk."""
        session_dir = self.base_dir / session_id
        session_file = session_dir / "session.json"

        if not session_file.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(session_file) as f:
            data = json.load(f)

        state = SessionState(**data)
        self._current_session = state
        self._current_session_dir = session_dir
        return state

    def save_session(self, state: SessionState | None = None) -> None:
        """Save current session state to disk."""
        state = state or self._current_session
        if state is None:
            raise ValueError("No session to save")

        session_dir = self._current_session_dir or (
            self.base_dir / state.metadata.session_id
        )

        state.metadata.updated_at = time.time()
        self._write_session_json(session_dir, state)

    def _write_session_json(self, session_dir: Path, state: SessionState) -> None:
        """Atomically write session.json."""
        import tempfile
        import os

        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="session_",
            dir=session_dir
        )
        try:
            with os.fdopen(temp_fd, "w") as f:
                f.write(state.model_dump_json(indent=2))
            os.rename(temp_path, session_dir / "session.json")
        except Exception:
            os.unlink(temp_path)
            raise

    def _update_current_symlink(self, session_id: str) -> None:
        """Update 'current' symlink to point to active session."""
        current_link = self.base_dir / "current"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(session_id)

    def _init_cell_index(self, session_dir: Path) -> None:
        """Initialize empty cell index."""
        index = {
            "version": "1.0",
            "session_id": self._current_session.metadata.session_id if self._current_session else "",
            "cells": {},
            "roots": [],
            "leaves": [],
            "execution_order": [],
        }
        index_file = session_dir / "cells" / "index.json"
        index_file.write_text(json.dumps(index, indent=2))

    # ... additional methods for cells, file access logging, etc.
```

#### 3.1.4 Migration Script

```python
# scripts/migrate_sessions.py

"""
One-time migration from flat files to session directories.

Usage:
    uv run python scripts/migrate_sessions.py --dry-run
    uv run python scripts/migrate_sessions.py --execute
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

def migrate_session(old_state_dir: Path, session_id: str, new_base: Path, dry_run: bool = True) -> dict:
    """Migrate a single session from flat files to session directory."""
    result = {"session_id": session_id, "status": "pending", "files_migrated": []}

    old_session_file = old_state_dir / f"{session_id}.json"
    old_context_file = old_state_dir / f"{session_id}_context.json"
    old_file_access = old_state_dir / f"file-access-{session_id}.jsonl"

    if not old_session_file.exists():
        result["status"] = "skipped"
        result["reason"] = "no_session_file"
        return result

    # Create new session directory
    new_session_dir = new_base / session_id

    if dry_run:
        result["status"] = "would_create"
        result["new_path"] = str(new_session_dir)
        return result

    new_session_dir.mkdir(parents=True, exist_ok=True)
    (new_session_dir / "cells").mkdir(exist_ok=True)
    (new_session_dir / "reasoning").mkdir(exist_ok=True)

    # Load old state
    with open(old_session_file) as f:
        old_state = json.load(f)

    # Load old context if exists
    old_context = {"messages": [], "files": {}, "tool_outputs": [], "working_memory": {}}
    if old_context_file.exists():
        with open(old_context_file) as f:
            old_context = json.load(f)

    # Build new unified session.json
    new_state = {
        "metadata": {
            "session_id": session_id,
            "created_at": old_state.get("created_at", 0),
            "updated_at": old_state.get("updated_at", 0),
            "ended_at": None,
            "cwd": "",
            "claude_transcript_path": None,
            "rlm_version": "0.1.0",
        },
        "activation": {
            "rlm_active": old_state.get("rlm_active", False),
            "activation_mode": "complexity",
            "activation_reason": None,
            "complexity_score": None,
            "current_depth": old_state.get("current_depth", 0),
            "max_depth": 3,
        },
        "context": {
            "conversation": old_context.get("messages", []),
            "files": old_context.get("files", {}),
            "tool_outputs": old_context.get("tool_outputs", []),
            "working_memory": old_state.get("working_memory", {}),
        },
        "budget": {
            "total_tokens_used": old_state.get("total_tokens_used", 0),
            "total_recursive_calls": old_state.get("total_recursive_calls", 0),
            "max_recursive_calls": 10,
            "cost_usd": 0.0,
            "by_model": {},
        },
        "cells": {"count": 0, "index_path": "cells/index.json"},
        "trajectory": {
            "events_count": old_state.get("trajectory_events_count", 0),
            "export_path": old_state.get("trajectory_path"),
        },
    }

    # Write new session.json
    session_json = new_session_dir / "session.json"
    session_json.write_text(json.dumps(new_state, indent=2))
    result["files_migrated"].append("session.json")

    # Copy file access log if exists
    if old_file_access.exists():
        shutil.copy(old_file_access, new_session_dir / "file-access.jsonl")
        result["files_migrated"].append("file-access.jsonl")

    result["status"] = "migrated"
    return result

def main():
    parser = argparse.ArgumentParser(description="Migrate RLM sessions to new structure")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--execute", action="store_true", help="Actually perform migration")
    args = parser.parse_args()

    old_state_dir = Path.home() / ".claude" / "rlm-state"
    new_base = Path.home() / ".claude" / "rlm-sessions"

    if not old_state_dir.exists():
        print("No old state directory found, nothing to migrate")
        return

    # Find all session files
    session_ids = set()
    for f in old_state_dir.glob("*.json"):
        name = f.stem
        if name.endswith("_context"):
            session_ids.add(name[:-8])  # Remove _context suffix
        elif name not in ("context", "session-metadata"):
            session_ids.add(name)

    print(f"Found {len(session_ids)} sessions to migrate")

    dry_run = not args.execute
    results = []

    for session_id in sorted(session_ids):
        result = migrate_session(old_state_dir, session_id, new_base, dry_run)
        results.append(result)
        print(f"  {session_id}: {result['status']}")

    if dry_run:
        print("\nDry run complete. Run with --execute to perform migration.")
    else:
        print(f"\nMigration complete. {len([r for r in results if r['status'] == 'migrated'])} sessions migrated.")

if __name__ == "__main__":
    main()
```

#### 3.1.5 Tests Required

- [ ] `tests/unit/test_session_manager.py` - Unit tests for SessionManager
- [ ] `tests/integration/test_session_migration.py` - Migration script tests
- [ ] `tests/property/test_session_schema.py` - Property tests for schema validation

---

### Phase 2: Cell Infrastructure (Week 2-3)

> **Revised Estimate**: Extended from Week 2 to Week 2-3 (Risk: **HIGH**)
> **Key Changes**:
> - Add cell support to `RLMEnvironment` via optional `CellManager` parameter
> - Use SPEC-04 hyperedges for dependencies instead of separate DAG
> - Add cycle detection and validation

**Goal**: Implement replayable cells with dependency tracking.

#### 3.2.1 Files to Create

| File | Purpose |
|------|---------|
| `src/cell_manager.py` | Cell creation, dependency resolution, replay |
| `src/cell_schema.py` | Pydantic models for cell schema |

#### 3.2.2 Files to Modify

| File | Changes |
|------|---------|
| `src/repl_environment.py` | Create cells for REPL operations |
| `src/recursive_handler.py` | Create cells for recursive calls |
| `src/session_manager.py` | Add cell management methods |

#### 3.2.3 Implementation Details

```python
# src/cell_manager.py

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

class CellInput(BaseModel):
    source: str  # "repl", "tool", "llm_call", etc.
    operation: str
    args: dict[str, Any]

class CellOutput(BaseModel):
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0

class CellMetadata(BaseModel):
    depth: int = 0
    model: str | None = None
    tokens_used: int = 0

class Cell(BaseModel):
    cell_id: str
    created_at: float
    type: Literal["repl", "tool", "llm_call", "map_reduce", "verification"]
    input: CellInput
    output: CellOutput
    dependencies: list[str] = []
    dependents: list[str] = []
    metadata: CellMetadata = CellMetadata()

class CellManager:
    """
    Manages replayable cells with dependency DAG.

    Implements: Session State Consolidation Plan, Phase 2
    """

    def __init__(self, cells_dir: Path, session_id: str):
        self.cells_dir = cells_dir
        self.session_id = session_id
        self.index_path = cells_dir / "index.json"
        self._index: dict[str, Any] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load cell index from disk."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                self._index = json.load(f)
        else:
            self._index = {
                "version": "1.0",
                "session_id": self.session_id,
                "cells": {},
                "roots": [],
                "leaves": [],
                "execution_order": [],
            }

    def _save_index(self) -> None:
        """Save cell index to disk."""
        import tempfile
        import os

        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="index_",
            dir=self.cells_dir
        )
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(self._index, f, indent=2)
            os.rename(temp_path, self.index_path)
        except Exception:
            os.unlink(temp_path)
            raise

    def generate_cell_id(self) -> str:
        """Generate a unique cell ID."""
        return f"cell_{uuid.uuid4().hex[:8]}"

    def create_cell(
        self,
        cell_type: str,
        source: str,
        operation: str,
        args: dict[str, Any],
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Cell:
        """Create a new cell and add to index."""
        cell_id = self.generate_cell_id()
        now = time.time()

        cell = Cell(
            cell_id=cell_id,
            created_at=now,
            type=cell_type,
            input=CellInput(source=source, operation=operation, args=args),
            output=CellOutput(),
            dependencies=dependencies or [],
            metadata=CellMetadata(**(metadata or {})),
        )

        # Write cell file
        cell_file = self.cells_dir / f"{cell_id}.json"
        cell_file.write_text(cell.model_dump_json(indent=2))

        # Update index
        self._index["cells"][cell_id] = {
            "type": cell_type,
            "dependencies": cell.dependencies,
            "dependents": [],
        }

        # Update dependents of dependencies
        for dep_id in cell.dependencies:
            if dep_id in self._index["cells"]:
                self._index["cells"][dep_id]["dependents"].append(cell_id)

        # Update roots/leaves
        if not cell.dependencies:
            self._index["roots"].append(cell_id)
        self._index["leaves"].append(cell_id)

        # Recompute execution order (topological sort)
        self._recompute_execution_order()

        self._save_index()
        return cell

    def complete_cell(self, cell_id: str, result: Any, error: str | None = None) -> None:
        """Mark a cell as complete with its output."""
        cell_file = self.cells_dir / f"{cell_id}.json"
        if not cell_file.exists():
            raise ValueError(f"Cell not found: {cell_id}")

        with open(cell_file) as f:
            data = json.load(f)

        data["output"]["result"] = result
        data["output"]["error"] = error

        cell_file.write_text(json.dumps(data, indent=2, default=str))

    def get_cell(self, cell_id: str) -> Cell | None:
        """Load a cell by ID."""
        cell_file = self.cells_dir / f"{cell_id}.json"
        if not cell_file.exists():
            return None
        with open(cell_file) as f:
            return Cell(**json.load(f))

    def get_execution_order(self) -> list[str]:
        """Get topologically sorted execution order."""
        return self._index.get("execution_order", [])

    def _would_create_cycle(self, cell_id: str, dep_id: str) -> bool:
        """Check if adding dep_id as dependency of cell_id would create a cycle."""
        # If dep_id depends on cell_id (transitively), adding this would cycle
        visited = set()
        queue = [dep_id]

        while queue:
            current = queue.pop(0)
            if current == cell_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            # Check what current depends on
            for d in self._index["cells"].get(current, {}).get("dependencies", []):
                queue.append(d)

        return False

    def validate_dependencies(self, cell_id: str, dependencies: list[str]) -> None:
        """
        Validate that adding these dependencies won't create issues.

        Raises:
            ValueError: If validation fails
        """
        for dep_id in dependencies:
            # Self-dependency check
            if dep_id == cell_id:
                raise ValueError(f"Self-dependency not allowed: {cell_id}")

            # Missing dependency check
            if dep_id not in self._index["cells"]:
                raise ValueError(f"Dependency not found: {dep_id}")

            # Cycle check
            if self._would_create_cycle(cell_id, dep_id):
                raise ValueError(f"Cycle detected: {cell_id} -> {dep_id}")

    def _recompute_execution_order(self) -> None:
        """Recompute topological execution order with cycle detection."""
        visited = set()
        in_stack = set()  # For cycle detection
        order = []
        cycles = []

        def visit(cell_id: str) -> bool:
            if cell_id in visited:
                return True
            if cell_id in in_stack:
                # Cycle detected
                cycles.append(cell_id)
                return False

            in_stack.add(cell_id)

            for dep_id in self._index["cells"].get(cell_id, {}).get("dependencies", []):
                if dep_id in self._index["cells"]:
                    if not visit(dep_id):
                        return False

            in_stack.remove(cell_id)
            visited.add(cell_id)
            order.append(cell_id)
            return True

        for cell_id in self._index["cells"]:
            if cell_id not in visited:
                if not visit(cell_id):
                    # Log cycle but continue with partial order
                    pass

        if cycles:
            # Mark cycles in index for debugging
            self._index["cycles_detected"] = cycles

        self._index["execution_order"] = order

    def get_dependency_chain(self, cell_id: str) -> list[str]:
        """Get all cells that this cell depends on (transitively)."""
        chain = []
        visited = set()

        def visit(cid: str) -> None:
            if cid in visited:
                return
            visited.add(cid)
            cell_info = self._index["cells"].get(cid, {})
            for dep_id in cell_info.get("dependencies", []):
                visit(dep_id)
            chain.append(cid)

        visit(cell_id)
        return chain[:-1]  # Exclude the cell itself

    def replay_cell(self, cell_id: str) -> Any:
        """
        Replay a cell's computation.

        This requires that all dependencies have been replayed first.
        """
        cell = self.get_cell(cell_id)
        if cell is None:
            raise ValueError(f"Cell not found: {cell_id}")

        # For now, just return the stored result
        # In a full implementation, this would re-execute the operation
        return cell.output.result
```

#### 3.2.4 Tests Required

- [ ] `tests/unit/test_cell_manager.py` - Cell CRUD, dependency resolution
- [ ] `tests/integration/test_cell_replay.py` - End-to-end cell replay
- [ ] `tests/property/test_cell_dag.py` - DAG property tests

---

### Phase 3: Hook Migration (Week 3-4)

> **Revised Estimate**: Extended from Week 3 to Week 3-4 (Risk: Medium)
> **Key Changes**:
> - Update hooks to use extended `StatePersistence`
> - Add rollback script for migration safety
> - Batch cell index updates on session save

**Goal**: Update all hooks to use SessionManager instead of flat files.

#### 3.3.1 Files to Modify

| File | Changes |
|------|---------|
| `scripts/capture_session_context.sh` | Use SessionManager.create_session() |
| `scripts/sync_context.py` | Use SessionManager.save_session() |
| `scripts/capture_output.py` | Use SessionManager, create cells |
| `scripts/track_file_access.sh` | Write to session directory |
| `scripts/externalize_context.py` | Use session directory |
| `scripts/save_trajectory.py` | Update to new structure |
| `scripts/repl_bridge.py` | Load from session.json |

#### 3.3.2 Updated Hook Flow

```
SessionStart:
  1. capture_session_context.sh
     - Call init_session.py (new, replaces bash logic)
     - Creates ~/.claude/rlm-sessions/{session_id}/
     - Initializes session.json
     - Creates cells/, reasoning/ directories
     - Updates 'current' symlink

  2. init_rlm.py (updated)
     - Creates rlm-sessions/ base directory
     - Validates rlm-config.json

PreToolUse:
  1. sync_context.py (updated)
     - Uses SessionManager.load_session()
     - Updates context in session.json
     - Creates pending cell if applicable

PostToolUse:
  1. capture_output.py (updated)
     - Uses SessionManager.save_session()
     - Completes pending cell
     - Updates tool_outputs in session.json

  2. track_file_access.sh (updated)
     - Writes to {session_dir}/file-access.jsonl

PreCompact:
  1. externalize_context.py (updated)
     - Creates snapshot in {session_dir}/snapshots/

Stop:
  1. save_trajectory.py (updated)
     - Sets ended_at in session.json
     - Archives if configured
```

#### 3.3.3 Tests Required

- [ ] `tests/integration/test_hook_flow.py` - Full hook integration tests
- [ ] `tests/integration/test_session_lifecycle.py` - Session creation to archival

---

### Phase 4: REPL Bridge Update (Week 4-5)

> **Revised Estimate**: Extended from Week 4 to Week 4-5 (Risk: Low)
> **Key Changes**:
> - Reuse existing types from `types.py` (`ExecutionResult`, `SessionContext`)
> - Add rollback operation to migration

**Goal**: Update REPL bridge to work with session-centric state.

#### 3.4.1 Files to Modify

| File | Changes |
|------|---------|
| `scripts/repl_bridge.py` | Load from session.json, support cell queries |

#### 3.4.2 New REPL Operations

```python
# New operations to add to repl_bridge.py

def op_session(args: dict[str, Any]) -> dict[str, Any]:
    """Get current session info."""
    session_dir = get_current_session_dir()
    session_file = session_dir / "session.json"
    if not session_file.exists():
        return {"error": "No active session"}
    with open(session_file) as f:
        data = json.load(f)
    return {
        "session_id": data["metadata"]["session_id"],
        "rlm_active": data["activation"]["rlm_active"],
        "current_depth": data["activation"]["current_depth"],
        "tokens_used": data["budget"]["total_tokens_used"],
    }

def op_cells(args: dict[str, Any]) -> dict[str, Any]:
    """Query cells in current session."""
    session_dir = get_current_session_dir()
    cells_dir = session_dir / "cells"
    index_file = cells_dir / "index.json"

    if not index_file.exists():
        return {"cells": [], "count": 0}

    with open(index_file) as f:
        index = json.load(f)

    # Filter by type if specified
    cell_type = args.get("type")
    cells = []
    for cell_id, info in index["cells"].items():
        if cell_type and info["type"] != cell_type:
            continue
        cells.append({
            "cell_id": cell_id,
            "type": info["type"],
            "dependencies": info["dependencies"],
            "dependents": info["dependents"],
        })

    return {
        "cells": cells,
        "count": len(cells),
        "execution_order": index.get("execution_order", []),
    }

def op_cell(args: dict[str, Any]) -> dict[str, Any]:
    """Get details of a specific cell."""
    cell_id = args.get("cell_id")
    if not cell_id:
        return {"error": "cell_id required"}

    session_dir = get_current_session_dir()
    cell_file = session_dir / "cells" / f"{cell_id}.json"

    if not cell_file.exists():
        return {"error": f"Cell not found: {cell_id}"}

    with open(cell_file) as f:
        return json.load(f)

def op_dependency_chain(args: dict[str, Any]) -> dict[str, Any]:
    """Get dependency chain for a cell."""
    cell_id = args.get("cell_id")
    if not cell_id:
        return {"error": "cell_id required"}

    session_dir = get_current_session_dir()
    index_file = session_dir / "cells" / "index.json"

    with open(index_file) as f:
        index = json.load(f)

    # BFS to get all dependencies
    chain = []
    visited = set()
    queue = [cell_id]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        cell_info = index["cells"].get(current, {})
        for dep_id in cell_info.get("dependencies", []):
            if dep_id not in visited:
                queue.append(dep_id)
                chain.append(dep_id)

    return {"chain": chain, "depth": len(chain)}

def op_replay(args: dict[str, Any]) -> dict[str, Any]:
    """
    Replay specified cells with optional force re-run.

    Usage: repl_bridge.py --op replay --args '{"cells": "2-4", "force": false}'

    Args:
        cells: Cell IDs to replay (comma-separated, ranges supported like "2-4")
        force: If true, re-run even if cached; if false, only re-run invalidated cells
    """
    cells_arg = args.get("cells", "")
    force = args.get("force", False)

    session_dir = get_current_session_dir()
    index_file = session_dir / "cells" / "index.json"

    if not index_file.exists():
        return {"error": "No cells found in session"}

    with open(index_file) as f:
        index = json.load(f)

    # Parse cell specification
    cell_ids_to_replay = []
    if "-" in cells_arg:
        # Range like "2-4"
        parts = cells_arg.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            all_cells = list(index["cells"].keys())
            cell_ids_to_replay = all_cells[start-1:end]  # 1-indexed
    else:
        # Comma-separated IDs
        cell_ids_to_replay = [c.strip() for c in cells_arg.split(",") if c.strip()]

    if not cell_ids_to_replay:
        return {"error": "No valid cells specified"}

    results = []
    for cell_id in cell_ids_to_replay:
        cell_file = session_dir / "cells" / f"{cell_id}.json"
        if not cell_file.exists():
            results.append({"cell_id": cell_id, "status": "not_found"})
            continue

        with open(cell_file) as f:
            cell = json.load(f)

        if not force and cell.get("output", {}).get("result") is not None:
            # Use cached result
            results.append({
                "cell_id": cell_id,
                "status": "cached",
                "result_preview": str(cell["output"]["result"])[:200]
            })
        else:
            # Would re-execute in full implementation
            results.append({
                "cell_id": cell_id,
                "status": "re_run_required",
                "note": "Full replay requires orchestrator execution"
            })

    return {
        "replay_results": results,
        "total": len(results),
        "cached": len([r for r in results if r["status"] == "cached"]),
        "re_run": len([r for r in results if r["status"] == "re_run_required"])
    }

def op_invalidate(args: dict[str, Any]) -> dict[str, Any]:
    """
    Invalidate cells affected by a file change.

    Called by PostToolUse hook when files are edited.
    """
    file_path = args.get("file_path")
    if not file_path:
        return {"error": "file_path required"}

    session_dir = get_current_session_dir()
    index_file = session_dir / "cells" / "index.json"

    if not index_file.exists():
        return {"invalidated": [], "count": 0}

    with open(index_file) as f:
        index = json.load(f)

    # Find cells that reference this file
    invalidated = []
    for cell_id, info in index["cells"].items():
        cell_file = session_dir / "cells" / f"{cell_id}.json"
        if cell_file.exists():
            with open(cell_file) as cf:
                cell = json.load(cf)
            # Check if file is in input context
            input_context = cell.get("input", {}).get("context_snapshot", {})
            if file_path in input_context.get("files", {}):
                invalidated.append(cell_id)
                # Mark for re-run by clearing output
                cell["output"]["result"] = None
                cell["output"]["error"] = "invalidated_by_file_change"
                cell_file.write_text(json.dumps(cell, indent=2))

    return {
        "file_path": file_path,
        "invalidated": invalidated,
        "count": len(invalidated)
    }
```

#### 3.4.3 Learning Loop Integration

Cells feed into DB promotion after verification:

```python
def op_promote_to_memory(args: dict[str, Any]) -> dict[str, Any]:
    """
    Promote verified cell output to persistent memory.

    Good pattern: Promote after verification (post-replay).
    Bad pattern: Blind promotion without checking accuracy.
    """
    cell_id = args.get("cell_id")
    memory_type = args.get("memory_type", "fact")  # "fact" or "experience"
    confidence = args.get("confidence", 0.8)

    if not cell_id:
        return {"error": "cell_id required"}

    session_dir = get_current_session_dir()
    cell_file = session_dir / "cells" / f"{cell_id}.json"

    if not cell_file.exists():
        return {"error": f"Cell not found: {cell_id}"}

    with open(cell_file) as f:
        cell = json.load(f)

    result = cell.get("output", {}).get("result")
    if result is None:
        return {"error": "Cell has no output to promote"}

    try:
        from src.memory_store import get_memory_store

        store = get_memory_store()

        if memory_type == "fact":
            # Add as a verified fact
            store.add_fact(
                content=str(result),
                source=f"cell:{cell_id}",
                confidence=confidence,
            )
        elif memory_type == "experience":
            # Add as an experience with trace
            store.add_experience(
                task=cell.get("input", {}).get("operation", "unknown"),
                context=cell.get("input", {}).get("args", {}),
                outcome=result,
                trace=cell.get("trace", {}),
            )

        # Mark cell as promoted
        cell["metadata"]["promoted_to_db"] = True
        cell_file.write_text(json.dumps(cell, indent=2))

        return {
            "status": "promoted",
            "cell_id": cell_id,
            "memory_type": memory_type,
            "confidence": confidence
        }
    except ImportError:
        return {"error": "Memory store not available"}
```

#### 3.4.4 Tests Required

- [ ] `tests/unit/test_repl_bridge_session.py` - New session operations
- [ ] `tests/integration/test_repl_cells.py` - Cell query operations

---

### Phase 5: Cleanup and Deprecation (Week 5)

**Goal**: Remove legacy code, update documentation, ensure smooth transition.

#### 3.5.1 Deprecation Path

1. **Week 5**: Add deprecation warnings to `state_persistence.py` flat-file methods
2. **Week 6**: Update all documentation to reference new paths
3. **Week 7**: Remove flat-file support (keep migration script for stragglers)

#### 3.5.2 Files to Deprecate/Remove

| File | Action |
|------|--------|
| `src/state_persistence.py` | Keep but deprecate direct file access methods |
| `scripts/capture_session_context.sh` | Keep but minimize logic, delegate to Python |
| `scripts/track_file_access.sh` | Keep, only path changes |

#### 3.5.3 Documentation Updates

| Document | Changes |
|----------|---------|
| `CLAUDE.md` | Update paths, add cell reference |
| `docs/user-guide.md` | Update state file locations |
| `docs/process/architecture.md` | Add ADR-010 for session-centric state |
| `README.md` | Update directory structure diagram |

---

## 4. Risks and Mitigations

### 4.1 Migration Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data loss during migration | Low | Critical | Dry-run mode, backups, idempotent migration |
| Performance regression | Medium | Medium | Benchmark before/after, lazy loading |
| Hook timing issues | Medium | High | Keep hooks fast, async cell creation |
| Disk space increase | High | Low | Compression, archival policy |

### 4.2 Compatibility Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| External tools expect old paths | Medium | Medium | Symlinks during transition |
| Break existing user workflows | Low | High | Preserve CLI interface |
| Break test suites | Medium | Medium | Update tests in each phase |

### 4.3 Mitigation Strategies

1. **Dual-Write Period**: During transition, write to both old and new locations
2. **Symlink Fallback**: Create symlinks from old paths to new for compatibility
3. **Version Detection**: Check session.json version field for schema changes
4. **Rollback Script**: Create script to revert migration if issues arise

---

## 5. Success Criteria

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

## 6. Benefits

### 6.1 Token & Cost Savings

Reuse cached cell outputs instead of re-prompting full histories:
- Example: In a 10-cell task, update 2 cells → save ~80% on LLM calls
- Selective replay: Only re-run invalidated cells, cache aggressively for unchanged paths

### 6.2 Continuous Context Understanding

The AI "remembers" via:
- **Replayable cells**: Short-term evolutions within and across sessions
- **DB (memory_store)**: Long-term facts and experiences

Come back later, tweak one part, and see updated results without losing prior work.

### 6.3 Debuggability & Traceability

Full audit trail:
- See exactly how the agent decided/built context
- Replay to reproduce bugs or experiment
- Each cell includes `trace` field with goal, decision, rationale

### 6.4 Modularity & Scalability

Like Jupyter, cells make complex reasoning composable:
- Easier to iterate on code updates
- Refine one part (e.g., auth flow) without re-analyzing unchanged parts
- Clear dependency graph shows what affects what

### 6.5 Learning Loop

Cells feed into DB promotion → system gets smarter over time:
- Promote verified cell outputs as facts/experiences
- Reuse successful aggregation patterns as procedures
- Cross-session learning via memory_store integration

---

## 7. Appendix

### A. File Size Estimates

| File Type | Typical Size | Max Size | Notes |
|-----------|-------------|----------|-------|
| session.json | 50KB | 5MB | Depends on context size |
| cell files | 1KB each | 100KB each | Most are small |
| file-access.jsonl | 10KB | 1MB | Grows with file ops |
| transcript.jsonl | 100KB | 50MB | Full conversation |

### B. Performance Targets

| Operation | Target | Max |
|-----------|--------|-----|
| Create session | <10ms | 50ms |
| Load session | <20ms | 100ms |
| Save session | <30ms | 150ms |
| Create cell | <5ms | 20ms |
| Query cells | <10ms | 50ms |

### C. Backward Compatibility

The migration maintains these backward-compatible paths:

| Old Path | New Path | Compatibility |
|----------|----------|---------------|
| `~/.claude/rlm-state/context.json` | `~/.claude/rlm-sessions/current/session.json` | Symlink during transition |
| `~/.claude/rlm-state/session-metadata.json` | Embedded in session.json | Migration copies data |
| `~/.claude/rlm-state/{id}.json` | Embedded in session.json | Migration merges data |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 0.3.0 | 2026-02-17 | Feasibility analysis: revised estimates (Phases 1-4), cycle detection code, critical recommendations table, SPEC alignment, performance validation |
| 0.2.0 | 2026-02-17 | Added: cell granularity patterns, good vs bad patterns table, decomposition flow diagram, replay/selective update section, replay/invalidate/promote operations, learning loop integration, benefits section |
| 0.1.0 | 2026-02-17 | Initial plan document |
