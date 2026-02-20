# CausalFrame Roadmap

Future development plans for the CausalFrame plugin.

## Phase 9: Living Documentation

As described in the whitepaper, these features enable the system to keep documentation in sync with code changes.

| Feature | Description | Status |
|---------|-------------|--------|
| Git diff → invalidation cascade | Detect code changes and invalidate dependent frames | Not Started |
| Frame re-execution | Re-run reasoning with preserved intent when code changes | Not Started |
| Selective documentation update | Update only affected docs based on invalidation graph | Not Started |

**Goal:** When code changes, the system can identify which documentation became stale and re-generate it with the original reasoning intent.

## Phase 10: Further Simplification

Optional cleanup to reduce file count from 18 to target 12.

| File | Lines | Action | Priority |
|------|-------|--------|----------|
| `response_parser.py` | 193 | Inline into `rlaph_loop.py` | Low |
| `tokenization.py` | 509 | Extract needed functions to `repl_environment.py` | Low |

**Target:** 12 src files total (as per original v2 spec)

## Phase 13: Claude Code Runtime Integration ✓

Full integration testing with actual Claude Code environment.

| Task | Description | Status |
|------|-------------|--------|
| FrameIndex persistence | Add save/load to JSON for cross-session frames | ✅ Complete |
| RLAPHLoop save on exit | Save frames before session ends | ✅ Complete |
| extract_frames hook | Load saved frames, persist to FrameStore | ✅ Complete |
| SessionArtifacts persistence | Track file hashes for comparison | ✅ Complete |
| compare_sessions hook | Compare with prior session, find invalidated frames | ✅ Complete |
| Orchestrator status events | Add progress output for visibility | ✅ Complete |

## Phase 14: Enhanced Verification ✓

Build on Phase 12's hallucination fixes with more sophisticated validation.

| Feature | Description | Status |
|---------|-------------|--------|
| Async file I/O | Parallel file reads (sync LLM + async I/O hybrid) | ✅ Complete |
| Architecture docs | Document design decisions in ARCHITECTURE.md | ✅ Complete |
| Semantic validation | Check if answer actually addresses the question | Future |
| Code execution validation | Ensure code actually ran (not just claimed) | Future |
| Multi-step verification | Verify each step in multi-step reasoning | Future |

## Phase 15: Performance Optimization

Optimize for speed and cost.

| Feature | Description |
|---------|-------------|
| Prompt caching | Cache repeated prompts to reduce token usage |
| Parallel llm_batch() | True parallel execution for independent queries |
| Result caching | Cache file reads and search results |
| Task agent for orchestrator | Switch from Bash to Task tool for better progress tracking and background execution |

## Phase 16: Advanced REPL Features

Expand REPL capabilities for more complex workflows.

| Feature | Description |
|---------|-------------|
| Custom function registration | Allow users to add custom REPL functions |
| Shell command execution | Safe access to shell commands |
| HTTP requests | Controlled network access for external data |

## Backlog Ideas

Lower priority ideas for future consideration:

- **Visual trajectory viewer**: Web UI for browsing frame trees
- **Frame export formats**: Markdown, JSON, Graphviz
- **Cross-session search**: Search across all historical frames
- **Frame editing**: Manual correction of invalid frames
- **Frame promotion UI**: Interactive promotion of frames to long-term knowledge

---

## Completed Phases

For historical reference, see [CHANGELOG.md](./CHANGELOG.md).

- Phase 1-8: Core refactoring ✓
- Phase 11: Cleanup broken scripts/tests ✓
- Phase 12: Fix RLM orchestrator hallucination ✓
- Phase 13: Claude Code Runtime Integration ✓ (2026-02-20)
- Phase 14: Enhanced Verification ✓ (2026-02-20)
