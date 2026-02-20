# CausalFrame Roadmap

Future development plans for the CausalFrame plugin.

## Priority Order

```
┌─────────────────────────────────────────────────────────────┐
│  NOW: Phase 17 → Phase 18 → Phase 19                       │
│  ContextMap    Tree/Evidence   UX                           │
├─────────────────────────────────────────────────────────────┤
│  LATER: Phase 9 → Phase 10 → Phase 15 → Phase 16           │
│  Living Docs   Simplification  Performance   Advanced REPL  │
├─────────────────────────────────────────────────────────────┤
│  BACKLOG: Visual viewer, export formats, cross-session      │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 17: ContextMap + Git-Aware Loading

**Priority: HIGH** — Foundational for spatial externalization and cross-session invalidation.

Externalize context so REPL navigates instead of reading files directly.

| Task | Description | Status |
|------|-------------|--------|
| Create `src/context_map.py` | Minimal ContextMap class (paths, hashes, lazy contents) | Not Started |
| Modify `repl_environment.py` | REPL uses ContextMap instead of direct file reads | Not Started |
| Add `commit_hash` to FrameIndex | Store git HEAD at session start | Not Started |
| Git diff at frame load | Detect changed files when loading old frames | Not Started |
| Update PostToolUse hook | Refresh ContextMap after Write/Edit | Not Started |

**Goal:** REPL becomes a navigator of externalized context, not a file-system reader. Cross-session change detection via git diff.

**Plan:** [2026-02-20-phase-17-contextmap.md](./plans/2026-02-20-phase-17-contextmap.md)

---

## Phase 18: Tree Structure + Evidence + Cascade

**Priority: HIGH** — Core gaps identified in code review feedback.

Transform linear chain into proper tree structure with evidence tracking.

| Task | Description | Status |
|------|-------------|--------|
| Populate `children` in frames | When sub-task spawned, add to parent.children | Not Started |
| Auto evidence tracking | Child frames → parent.evidence, tool outputs → evidence | Not Started |
| Auto-generate `invalidation_condition` | Default: "any file in context_slice.files changes" | Not Started |
| Stronger cascade propagation | Recurse children + evidence consumers on invalidation | Not Started |
| `find_dependent_frames()` method | Scan frames for evidence references | Not Started |

**Goal:** Proper tree structure with cascade invalidation when premises change.

**Plan:** [2026-02-20-phase-18-tree-evidence.md](./plans/2026-02-20-phase-18-tree-evidence.md)

---

## Phase 19: Causal Awareness UX

**Priority: MEDIUM** — Make the system feel "alive" with query/navigation capabilities.

| Task | Description | Status |
|------|-------------|--------|
| `/causalframe status <topic>` skill | Query valid frames, summarize still-valid conclusions | Not Started |
| Branch resumption | Resume from suspended/invalidated frame with new evidence | Not Started |
| Surface invalidated frames | Auto-show invalidated frames on session start | Not Started |

**Goal:** Model actively navigates causal store, not just passive persistence.

---

# Future Phases

*These phases depend on Phase 17-19 being complete first.*

---

## Phase 9: Living Documentation

As described in the whitepaper, these features enable the system to keep documentation in sync with code changes.

| Feature | Description | Status |
|---------|-------------|--------|
| Git diff → invalidation cascade | Detect code changes and invalidate dependent frames | → Phase 17/18 |
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
