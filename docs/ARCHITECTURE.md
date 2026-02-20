# CausalFrame Architecture Decisions

This document records key architectural decisions made during the development of CausalFrame.

---

## Sync LLM + Async I/O Hybrid

**Date:** 2026-02-20
**Phase:** 13-14

### Problem

Pure async LLM calls are hard to debug due to non-deterministic execution order. Pure sync I/O is slow when reading multiple files.

### Solution

Hybrid approach:
- **LLM calls are synchronous** - predictable execution order, easy to debug
- **File I/O is async** - parallel reads, non-blocking disk access

### Code Pattern

```python
# Sync LLM - predictable
response = await llm_call_sync(query)

# Async I/O - fast
files = await asyncio.gather(*[
    read_file_async(path) for path in paths
])
```

### Rationale

| Sync LLM | Async I/O |
|----------|-----------|
| Easy to debug | Parallel file reads |
| Clear execution order | Non-blocking disk access |
| No race conditions | Better UX (responsive) |

---

## Frame Persistence (Hybrid A+B)

**Date:** 2026-02-20
**Phase:** 13-14

### Problem

Claude Code hooks run as separate processes. They cannot access the in-memory FrameIndex from the running Python session that created the frames.

### Solution

Two-level persistence:
1. **RLAPHLoop saves directly to disk** - safety backup, frames never lost
2. **extract_frames.py hook loads from disk** - proper Claude Code hook integration

### Data Flow

```
RLAPHLoop.run()
    ↓
FrameIndex.save(session_id)  → ~/.claude/rlm-frames/{session}/index.json
    ↓
Session ends
    ↓
extract_frames.py hook
    ↓
FrameIndex.load(session_id)
    ↓
FrameStore.save()  → ~/.claude/rlm-frames/{session}/frames.jsonl
```

### Session Folder Structure

```
~/.claude/rlm-frames/
├── session-abc123/
│   ├── index.json       # FrameIndex saved by RLAPHLoop
│   ├── frames.jsonl     # FrameStore persisted by extract_frames.py
│   ├── artifacts.json   # SessionArtifacts for comparison
│   └── tools.jsonl      # Tool outputs captured by capture_output.py
├── session-def456/
│   └── ...
```

---

## Orchestrator Status Events (Phase 13-14)

**Date:** 2026-02-20

### Problem

Running the RLM orchestrator via bash provides no visibility into progress. Users don't know if it's thinking, crashed, or loading.

### Solution

Keep bash execution, add structured status markers:

```
[RLM:START] Initializing RLAPH loop
[RLM:QUERY] Processing (1234 chars)
[RLM:DONE] Completed in 3 iterations
[RLM:DONE] Tokens: 1234, Time: 5000ms
```

### Rationale

- Simple implementation (~5 lines of code)
- No architectural changes
- Task agent approach deferred to Phase 15 (see ROADMAP.md)

---

## Session ID Coordination

**Date:** 2026-02-20
**Phase:** 13-14

### Problem

Multiple hooks (PostToolUse, Stop) and the orchestrator run in separate processes. They need to coordinate to ensure all session artifacts land in the same folder.

### Solution

Coordination file pattern:

1. **PostToolUse hook** receives `session_id` from Claude Code via stdin
2. Hook writes to coordination file: `~/.claude/rlm-frames/.current_session`
3. **Orchestrator** reads coordination file and uses the same `session_id`
4. **Stop hook** reads coordination file to find frames

### Coordination File Format

```json
{
  "session_id": "039f5c9f-e804-44d9-9946-1098a64c8c1b",
  "pid": 12345,
  "updated_at": "2026-02-20T12:00:00"
}
```

### Priority Chain

The orchestrator resolves session_id in this order:
1. Explicit `--session-id` argument
2. `CLAUDE_SESSION_ID` environment variable
3. Coordination file (`~/.claude/rlm-frames/.current_session`)
4. Generated `UUID[:8]`

### Result

All artifacts in ONE folder per Claude session:
- `tools.jsonl` - Tool outputs captured by PostToolUse hook
- `index.json` - FrameIndex saved by RLAPHLoop
- `frames.jsonl` - CausalFrames persisted by Stop hook

---

## Related Documents

- [Design Doc](DESIGN.md)
- [Whitepaper](WHITEPAPER.md)
- [Roadmap](ROADMAP.md)
