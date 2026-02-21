# CausalFrame Changelog

All notable changes to CausalFrame are documented in this file.

## [0.0.2] - 2026-02-21

### Phase 18.5+ Cleanup - Codebase Hygiene

**Overview:** Removed 22 unused files after Phase 18.5 completion.

### Deleted Files

**Migration Scripts (one-time use):**
- `scripts/migrate_sessions.py`
- `scripts/migrate_to_frames.py`
- `scripts/merge-plugin-hooks.py`

**Setup Scripts (obsolete):**
- `scripts/setup-rlm-core.sh`
- `scripts/set-api-key.sh`
- `scripts/setup_repl_env.sh`
- `scripts/capture_session_context.sh`
- `scripts/verify_rlm.py`

**Shell Hook Scripts (replaced by Python hooks):**
- Entire `scripts/hooks/` directory (11 files)

**Old Test Files:**
- `test_llm_repl.py`
- `test_rlaph_fast.py`
- `test_rlaph_loop.py`

### Documentation Updates

- Updated README.md: removed reference to deleted `/verification` skill
- Updated DESIGN.md: corrected file list to match current structure

### Remaining Scripts (Active)

| Script | Purpose |
|--------|---------|
| `causal_cli.py` | CLI entry point for `/causal` skill |
| `compare_sessions.py` | SessionStart hook |
| `capture_output.py` | PostToolUse hook |
| `extract_frames.py` | Stop hook |
| `get_session_id.py` | Session ID utilities |
| `run_orchestrator.py` | Standalone orchestrator (dev/test) |

## [0.0.1] - 2026-02-19

### Initial Release - RLM v2 Refactoring Complete

**Overview:** Complete rewrite from ~90 files to 18 src files, focusing on REPL + CausalFrame persistence.

### Core Features

- **REPL Layer**: Externalized context navigation with `peek()`, `search()`, `llm()`, `llm_batch()`, `map_reduce()`
- **CausalFrame**: Temporal persistence across sessions with invalidation cascade
- **FrameStore**: JSONL-based storage (zero dependencies, human-readable)
- **Config Management**: Plugin config takes priority over env vars

### Files Structure (18 src files)

| Layer | Files |
|-------|-------|
| **REPL** | `rlaph_loop.py`, `repl_environment.py`, `llm_client.py`, `response_parser.py`, `tokenization.py` |
| **Causal** | `causal_frame.py`, `context_slice.py`, `frame_index.py`, `frame_invalidation.py`, `frame_store.py` |
| **Session** | `session_artifacts.py`, `session_comparison.py` |
| **Plugin** | `plugin_interface.py`, `plugins/default_plugin.py` |
| **Config** | `config.py`, `prompts.py`, `types.py`, `tool_bridge.py`, `__init__.py` |

### Phase 11-12: Bug Fixes & Reliability

**Priority Fixes (P0-P1):**
- Fixed 7 broken scripts importing deleted modules
- Fixed 2 scripts with import paths (extract_frames.py, compare_sessions.py)
- Deleted 3 broken tests
- **Fixed RLM orchestrator hallucination** - added result verification and retry loop

**Hallucination Fix Details:**
- Added `_verify_result()` method to detect fake file paths and fake hashes
- Added retry loop when verification fails
- Strengthened system prompt with import restrictions and anti-hallucination rules
- Added file access tracking to validate LLM responses
- Added 4 integration tests for orchestrator accuracy

### Configuration

**Default Model Cascade (benchmark-tested):**
- `root_model: glm-4.6` (100% accuracy, 3.1s avg)
- `recursive_depth_1: glm-4.6`
- `recursive_depth_2: glm-4.7`
- `recursive_depth_3: glm-4.7`
- `temperature: 0.1` (deterministic for REPL)

**Config Priority:** Plugin config file > Dataclass defaults > Environment variables

### Test Status

- **197 unit tests** passing
- **4 integration tests** passing
- **Total: 201 tests**

### Commits of Note

1. `573cd05` - Simplify hooks.json to v2 structure
2. `91ec58d` - Add v2 exports to __init__.py
3. `fc87f58` - Replace ModelRouter with LLMClient
4. `eb0ba0b` - Remove orchestration_schema dependency
5. `58b41c7` - Add FrameIndex integration
6. `1cf4f50` - Phase 5 cleanup Batch A-C
7. `f20e599` - Phase 5 cleanup Batch D-F
8. `456d87b` - Phase 5 cleanup Batch G
9. `bbf1792` - Remove obsolete tests
10. `7be7893` - Update refactoring plan with completion status
11. `5b30b96` - Rename to Causeway and complete Phase 7
12. `20d9c7b` - Update plan - Phase 11-12 Complete

### Known Limitations

1. `llm()` depth management is simplified - full recursive behavior needs testing
2. Frame lifecycle (`RUNNING` → `COMPLETED`) is implemented but not validated end-to-end
3. Hook integration is basic - full integration requires Claude Code runtime testing

### Migration Notes

From original RLM-Claude-Code:
- Removed: orchestrators, routers, vector search, async stack, epistemic verification
- Kept: REPL environment, CausalFrame concept, JSONL storage
- Simplified: Model selection, config management, plugin interface

---

## Next Release (Roadmap)

See [ROADMAP.md](./ROADMAP.md) for planned features.
