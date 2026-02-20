# CausalFrame - Claude Code Instructions

This is the CausalFrame plugin for Claude Code.

## What CausalFrame Does

CausalFrame gives Claude Code **causal awareness** — the ability to:
1. Externalize context into a navigable REPL
2. Persist reasoning chains across sessions
3. Detect when prior conclusions are invalidated by changes

## Core Concepts

### REPL Layer
The model actively navigates context instead of receiving it passively:
- `peek(var, start, end)` - View portion of context
- `search(var, pattern)` - Find patterns in context
- `llm(query, context)` - Immediate recursive LLM call

### CausalFrame
Every reasoning step is captured with:
- `query` - What was asked
- `context_slice` - What was visible
- `evidence` - What it relied on
- `conclusion` - What it concluded
- `invalidation_condition` - What would make this wrong

### FrameStatus
- `RUNNING` - Currently executing
- `COMPLETED` - Successfully finished
- `SUSPENDED` - Paused, resumable
- `INVALIDATED` - No longer valid
- `PROMOTED` - Persisted as long-term knowledge

## Architecture Principle

**Core enforces structure. LM decides content.**

| Core enforces | LM decides |
|---------------|------------|
| token_budget | what files to include |
| max depth | whether to recurse |
| CausalFrame structure | confidence interpretation |
| invalidation cascade | escalation policy |

## Skills

- `/causalframe` - Activate for complex multi-step reasoning
- `/verification` - Constraint verification with CPMpy

## Hooks

| Hook | What it does |
|------|--------------|
| SessionStart | Compare with prior session, surface invalidated frames |
| PostToolUse | Capture tool outputs into active CausalFrame |
| Stop | Extract frame tree, save to FrameStore |

## File Structure

```
src/
├── rlaph_loop.py          # Main execution loop
├── repl_environment.py    # REPL sandbox and functions
├── llm_client.py          # LLM provider abstraction
├── tool_bridge.py         # Controlled tool access
├── causal_frame.py        # CausalFrame dataclass
├── context_slice.py       # Context window management
├── frame_index.py         # In-memory frame index
├── frame_invalidation.py  # Cascade invalidation
├── frame_store.py         # JSONL persistence
├── session_artifacts.py   # Session metadata
└── session_comparison.py  # Cross-session diff
```

## Key Design Decisions

1. **Immediate execution** - `llm()` returns results synchronously, no async queues
2. **JSONL per session** - No SQLite, human-readable, zero dependencies
3. **O(n) scan** - 10-20 frames, no vector search needed
4. **Two-field branches** - `status` + `branched_from`, no BranchPoint class

## References

- [Design Doc](docs/DESIGN.md)
- [Whitepaper](docs/WHITEPAPER.md)
- [Refactoring Plan](docs/plans/2026-02-19-rlm-v2-refactoring.md)
