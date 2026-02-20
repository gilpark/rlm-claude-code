# AGENTS.md - CausalFrame Project Guide

This is a Claude Code plugin that provides **causal awareness** through REPL-based context navigation and CausalFrame persistence.

## Project Overview

**CausalFrame** gives Claude Code the ability to:
1. Externalize context into a navigable REPL (spatial awareness)
2. Persist reasoning chains across sessions (temporal awareness)
3. Detect when prior conclusions are invalidated by code changes (causal awareness)

**Architecture Principle:** Core enforces structure. LM decides content.

## File Structure

```
src/
├── rlaph_loop.py          # Main execution loop - REPL orchestration
├── repl_environment.py    # REPL sandbox with RestrictedPython
├── llm_client.py          # Synchronous LLM client with model cascade
├── tool_bridge.py         # Controlled tool access for sub-LLMs
├── causal_frame.py        # CausalFrame dataclass (query, context, conclusion, invalidation)
├── context_slice.py       # Context window management
├── frame_index.py         # In-memory frame index
├── frame_invalidation.py  # Cascade invalidation logic
├── frame_store.py         # JSONL persistence
├── session_artifacts.py   # Session metadata
├── session_comparison.py  # Cross-session diff
├── plugin_interface.py    # Claude Code plugin interface
├── config.py              # Configuration management
├── prompts.py             # System prompts
├── types.py               # Shared types
└── __init__.py            # Public API exports
```

## Core Commands

| Command | Purpose |
|---------|---------|
| `uv run pytest tests/ -v` | Run all tests |
| `uv run pytest tests/ -v --tb=short` | Run tests with short traceback |
| `uv run pytest tests/unit/test_<module>.py -v` | Run specific unit tests |
| `uv run pytest tests/integration/ -v -o asyncio_mode=auto` | Run integration tests |
| `uv run python scripts/run_orchestrator.py "<query>"` | Test orchestrator manually |

## Development Workflow

### Running Tests

Always run tests before committing:
```bash
uv run pytest tests/ -v --tb=short
```

For integration tests (which use asyncio):
```bash
uv run pytest tests/integration/ -v -o asyncio_mode=auto
```

### Code Style

- Python 3.11+
- Type hints required for all public functions
- Docstrings for all classes and public methods
- Max line length: 100 (soft), 120 (hard)

### Architecture Constraints

**DO NOT:**
- Add async complexity to `llm_client.call()` - it must be synchronous
- Import from deleted modules (orchestrator/, router*, async_*, etc.)
- Use SQLite or external dependencies - JSONL only
- Add vector search or embedding retrieval

**DO:**
- Use dataclasses for all data structures
- Keep REPL functions pure and side-effect-free
- Validate results before trusting LLM output
- Track file access to detect hallucinations

## Configuration

```json
{
  "causalframe": {
    "activation": {"mode": "micro"},
    "depth": {"default": 2, "max": 3}
  }
}
```

Config priority: **Plugin config file > Dataclass defaults > Environment variables**

Default models (benchmark-tested):
- `root_model: glm-4.6` (100% accuracy, 3.1s avg)
- `recursive_depth_1: glm-4.6`
- `recursive_depth_2: glm-4.7`
- `recursive_depth_3: glm-4.7`
- `temperature: 0.1` (deterministic for REPL)

## Key Concepts

### REPL Functions
The REPL provides these functions to the LLM:
- `peek(var, start, end)` - View portion of context
- `search(var, pattern)` - Find patterns in context
- `llm(query, context)` - Immediate recursive LLM call
- `read_file(path)` - Read file from disk
- `glob_files(pattern)` - Find files matching pattern

### CausalFrame
Every reasoning step captures:
- `query` - What was asked
- `context_slice` - What was visible (files, memory, tool outputs)
- `evidence` - What it relied on (parent frame IDs)
- `conclusion` - What it concluded
- `invalidation_condition` - What would make this wrong
- `status` - RUNNING, COMPLETED, SUSPENDED, INVALIDATED, PROMOTED

### FrameStatus Lifecycle
- `RUNNING` → Currently executing
- `COMPLETED` → Successfully finished
- `SUSPENDED` → Paused, resumable (pivot point)
- `INVALIDATED` → No longer valid (cascade from code changes)
- `PROMOTED` → Persisted as long-term knowledge

## Anti-Hallucination Rules

The orchestrator has built-in verification:
1. File paths in answers must actually exist
2. Answers must reference files that were actually accessed
3. Fake hash patterns (e.g., `a1b2c3d4e5f6a7b8`) are rejected
4. System prompt blocks import statements (pre-loaded libraries only)

## Hooks

| Hook | Script | Purpose |
|------|--------|---------|
| SessionStart | `scripts/compare_sessions.py` | Compare with prior session, surface invalidated frames |
| PostToolUse | (not yet implemented) | Capture tool output into active CausalFrame |
| Stop | `scripts/extract_frames.py` | Extract frame tree, save to FrameStore |

## Documentation

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project instructions for Claude Code |
| `README.md` | Human-readable project overview |
| `docs/CHANGELOG.md` | Version history and what's been done |
| `docs/ROADMAP.md` | Future work (Phase 9-10 and beyond) |
| `docs/DESIGN.md` | Architecture specification |
| `docs/WHITEPAPER.md` | Theoretical foundation |

## Current Status

- **Version:** 0.0.1
- **Tests:** 201 passing (197 unit + 4 integration)
- **Src files:** 18 (from ~90 original)
- **Blocking issues:** None

See `docs/CHANGELOG.md` for detailed history.
