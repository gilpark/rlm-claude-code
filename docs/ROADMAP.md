# CausalFrame Roadmap

Future development plans for the CausalFrame plugin.

## Priority Order

```
┌─────────────────────────────────────────────────────────────┐
│  NOW: Phase 18.5 — Structured LLM Output (JSON)            │
│  Token savings + reliable parsing + explicit recursion     │
├─────────────────────────────────────────────────────────────┤
│  NEXT: Phase 19 — UX & Skills Standardization              │
│  Make it feel like a real plugin (slash commands, discoverable)│
├─────────────────────────────────────────────────────────────┤
│  THEN: Phase 20 — Multi-Session Demo & Branch Resumption   │
│  Prove the causal evolution loop works                      │
├─────────────────────────────────────────────────────────────┤
│  LATER: Phase 9 — Living Documentation (surgical updates)  │
│  Show real value in CI/CD / git push workflows              │
├─────────────────────────────────────────────────────────────┤
│  BACKLOG: Phase 15, 16, 10, cross-repo, visual viewer, etc.│
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 17: ContextMap + Git-Aware Loading ✓

**Priority: HIGH** — Foundational for spatial externalization and cross-session invalidation.

Externalize context so REPL navigates instead of reading files directly.

| Task | Description | Status |
|------|-------------|--------|
| Create `src/frame/context_map.py` | Minimal ContextMap class (paths, hashes, lazy contents) | ✅ Complete |
| Modify `repl_environment.py` | REPL uses ContextMap instead of direct file reads | ✅ Complete |
| Add `commit_hash` to FrameIndex | Store git HEAD at session start | ✅ Complete |
| Git diff at frame load | Detect changed files when loading old frames | ✅ Complete |
| Dynamic discovery + security guard | Auto-add files within root, reject outside | ✅ Complete |

**Goal:** REPL becomes a navigator of externalized context, not a file-system reader. Cross-session change detection via git diff.

**Plan:** [2026-02-20-phase-17-contextmap.md](./plans/completed/2026-02-20-phase-17-contextmap.md)

---

## Phase 18: Tree Structure + Evidence + Cascade + Recursion ✓

**Priority: HIGH** — Core gaps identified in code review feedback.

Transform linear chain into proper tree structure with evidence tracking AND enable working recursion via synchronous `llm(sub_query)`.

### Part A: Recursion Infrastructure

| Task | Description | Status |
|------|-------------|--------|
| Synchronous `llm(sub_query)` | Creates child frames, proper depth tracking (no subprocess) | ✅ Complete |
| Enhanced system prompt | Explicit recursion guidance with examples | ✅ Complete |
| Query/intent tracking | `initial_query` and `query_summary` in FrameIndex | ✅ Complete |

### Part B: Tree Structure + Evidence + Cascade

| Task | Description | Status |
|------|-------------|--------|
| Populate `children` in frames | When sub-task spawned, add to parent.children | ✅ Complete |
| Auto evidence tracking | Child frames → parent.evidence, tool outputs → evidence | ✅ Complete |
| Auto-generate `invalidation_condition` | Structured dict with files/tools/memory_refs | ✅ Complete |
| Stronger cascade propagation | Recurse children + evidence consumers on invalidation | ✅ Complete |
| `find_dependent_frames()` method | Scan frames for evidence references (with caching) | ✅ Complete |

### Part C: Intent Normalization

| Task | Description | Status |
|------|-------------|--------|
| `CanonicalTask` dataclass | task_type, target, scope, params with stable hash | ✅ Complete |
| `extract_canonical_task()` | Language-agnostic intent extraction | ✅ Complete |
| Frame identity by intent | hash(canonical_task + context_slice) prevents duplication | ✅ Complete |

**Key Design Decision:** Use synchronous `llm(sub_query)` instead of subprocess.
- Simpler, faster, no IPC overhead
- Fits RLM ethos (recursive calls in same process)
- Depth derived from parent frame, not global counter

**Goal:** Proper tree structure with working recursion and cascade invalidation when premises change.

**Plan:** [2026-02-20-phase-18-tree-evidence.md](./plans/completed/2026-02-20-phase-18-tree-evidence.md)

---

## Phase 18.5: Structured LLM Output (JSON)

**Priority: HIGH** — Token savings + reliable parsing + explicit recursion.

Make `llm()` return structured JSON instead of raw text.

| Task | Description | Status |
|------|-------------|--------|
| Update system prompt | Request JSON format with schema | Not Started |
| Safe JSON parser | `parse_llm_response()` with fallback | Not Started |
| Use parsed confidence | Frame creation uses JSON confidence | Not Started |
| Handle `sub_tasks` | Auto-recurse when sub_tasks present | Not Started |
| Unit tests | Parser edge cases | Not Started |
| Integration tests | Token savings verification | Not Started |

**JSON Schema:**
```json
{
  "reasoning": "Step-by-step thinking",
  "conclusion": "Final answer",
  "confidence": 0.85,
  "files": ["auth.py"],
  "sub_tasks": [{"query": "...", "priority": 1}],
  "needs_more_info": false
}
```

**Benefits:**
- 30-40% token savings on complex responses
- Reliable parsing (no fragile regex)
- Explicit recursion via `sub_tasks`
- Safe fallback when parsing fails

**Goal:** Make LLM responses parseable, efficient, and recursion-aware.

**Plan:** [2026-02-20-phase-18.5-structured-output.md](./plans/2026-02-20-phase-18.5-structured-output.md)

---

## Phase 19: UX & Skills Standardization

**Priority: MEDIUM**
**Dependencies:** Phase 18.5 (Structured Output) ✓

Make the system feel "alive" with discoverable slash commands.

| Task | Description | Status |
|------|-------------|--------|
| `/causal` router skill | Single entry point, dispatches to sub-commands | Not Started |
| `/causal analyze <target>` | Run RLAPH with canonical_task extraction | Not Started |
| `/causal status [topic]` | Query valid frames, summarize still-valid conclusions | Not Started |
| `/causal resume <frame_id>` | Resume suspended/invalidated branch with new evidence | Not Started |
| `/causal tree [session]` | Visualize frame tree with depths and statuses | Not Started |
| `run_rlaph()` library function | Refactor orchestrator as importable, callable function | Not Started |
| SessionStart enhancement | Surface invalidated frames on session start | Not Started |
| `--verbose` / `--depth` flags | Transparency and control via Claude's flag syntax | Not Started |

**Goal:** Model actively navigates causal store with discoverable UX. Skills import `run_rlaph()` as library.

**Plan:** [2026-02-20-phase-19-ux-skills.md](./plans/2026-02-20-phase-19-ux-skills.md)

---

## Phase 20: Multi-Session Demo & Branch Resumption

**Priority: MEDIUM** — Prove the causal evolution loop works across sessions.

| Task | Description | Status |
|------|-------------|--------|
| 2-session change demo | Change file → targeted invalidation + reuse of unaffected frames | Not Started |
| Branch resumption logic | Resume from suspended/invalidated frame with preserved intent | Not Started |
| Intent reuse demo | Same canonical_task → reuse prior frame (no re-computation) | Not Started |
| Living doc prototype | Change code → auto-update only affected doc sections | Not Started |

**Goal:** Demonstrate real value: prior knowledge survives code changes, task reuse becomes visible.

---

# Future Phases

*These phases depend on Phase 19-20 being complete first.*

---

## Phase 9: Living Documentation

As described in the whitepaper, these features enable the system to keep documentation in sync with code changes.

| Feature | Description | Status |
|---------|-------------|--------|
| Git diff → invalidation cascade | Detect code changes and invalidate dependent frames | → Phase 17/18 ✓ |
| Frame re-execution | Re-run reasoning with preserved intent when code changes | → Phase 20 |
| Selective documentation update | Update only affected docs based on invalidation graph | Not Started |

**Goal:** When code changes, the system can identify which documentation became stale and re-generate it with the original reasoning intent.

## Phase 10: Further Simplification

Optional cleanup to reduce file count from 18 to target 12.

| File | Lines | Action | Priority |
|------|-------|--------|----------|
| `response_parser.py` | 193 | Inline into `rlaph_loop.py` | Low |
| `tokenization.py` | 509 | Extract needed functions to `repl_environment.py` | Low |

**Target:** 12 src files total (as per original v2 spec)

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
- Phase 17: ContextMap + Git-Aware Loading ✓ (2026-02-20)
- Phase 18: Tree Structure + Evidence + Cascade + Recursion ✓ (2026-02-20)
