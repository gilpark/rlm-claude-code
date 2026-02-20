# RLM Data Flow

This document describes the complete data flow from Main LLM (Claude Code) to RLM (Recursive Language Model) and back, including how CausalFrames are saved.

## Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant MainLLM as Main LLM<br/>(Claude Code)
    participant Skill as /rlm-orchestrator<br/>Skill
    participant Bash as Bash<br/>Tool
    participant Script as run_orchestrator.py
    participant Loop as RLAPHLoop
    participant REPL as RLMEnvironment<br/>(REPL)
    participant LLM as LLM Client
    participant Index as FrameIndex
    participant Disk as Disk Storage<br/>~/.claude/rlm-frames/

    User->>MainLLM: User query with /rlm-orchestrator
    Note over MainLLM: Skill invoked

    rect rgb(240, 248, 255)
        Note over MainLLM,Skill: Phase 1: Context Isolation
        MainLLM->>MainLLM: Glob/Grep to find files
        MainLLM->>MainLLM: Compose prompt (task + paths)
        Note over MainLLM: Does NOT read file contents
    end

    MainLLM->>Bash: uv run python scripts/run_orchestrator.py "<prompt>"
    Bash->>Script: Execute script

    rect rgb(255, 250, 240)
        Note over Script,LLM: Phase 2: RLM Execution
        Script->>Script: build_context(files={})
        Script->>Loop: new RLAPHLoop(max_depth=2)
        Script->>Loop: await loop.run(query, context)

        Loop->>REPL: new RLMEnvironment(context)
        Loop->>REPL: enable_file_access(working_dir)
        Loop->>Index: new FrameIndex()

        loop Each Iteration (max 20)
            Loop->>LLM: llm_client.call(prompt)
            LLM-->>Loop: response_content
            Loop->>Loop: parser.parse(response)

            alt REPL Code Found
                Loop->>REPL: execute(code)
                Note over REPL: Reads files on-demand<br/>via read_file(), glob_files()
                REPL-->>Loop: exec_result

                rect rgb(255, 240, 245)
                    Note over Loop,Index: CausalFrame Creation
                    Loop->>Loop: Create ContextSlice
                    Loop->>Index: CausalFrame(query, context_slice, ...)
                    Loop->>Index: frame_index.add(frame)
                    Note over Index: Frame tracked in memory
                end

                Loop->>Loop: Append result to messages
            else Final Answer
                Loop->>Loop: _verify_result(answer)
                Note over Loop: Hallucination check
            end
        end

        rect rgb(240, 255, 240)
            Note over Loop,Disk: Phase 3: Persistence
            Loop->>Index: frame_index.save(session_id)
            Index->>Disk: Write index.json
            Note over Disk: ~/.claude/rlm-frames/{session_id}/index.json
        end

        Loop-->>Script: RLPALoopResult(answer, iterations, ...)
    end

    Script-->>Bash: Print answer
    Bash-->>MainLLM: Return answer
    MainLLM->>User: Report answer

    rect rgb(245, 245, 255)
        Note over MainLLM,Disk: Phase 4: Post-Session Hooks

        alt Stop Hook (Session End)
            MainLLM->>Script: extract_frames.py
            Script->>Index: FrameIndex.load(session_id)
            Index-->>Script: frames dict
            Script->>Disk: FrameStore.save(frame)
            Note over Disk: Append to frames.jsonl
        end

        alt SessionStart Hook (Next Session)
            Script->>Disk: find_most_recent_session()
            Script->>Disk: SessionArtifacts.load(prior_id)
            Script->>Disk: FrameIndex.load(prior_id)
            Script->>Script: compare_sessions(current, prior)
            Script-->>MainLLM: invalidated_frames, changed_files
            Note over MainLLM: Resume from invalidated frames
        end
    end
```

## Data Flow Summary

### Phase 1: Context Isolation (Main LLM)

| Step | Component | Action |
|------|-----------|--------|
| 1 | User | Invokes `/rlm-orchestrator` with task |
| 2 | Main LLM | Uses Glob/Grep to find relevant files |
| 3 | Main LLM | Composes prompt with task + file paths |
| 4 | Main LLM | Does NOT read file contents |

### Phase 2: RLM Execution (Python Process)

| Step | Component | Action |
|------|-----------|--------|
| 5 | Bash | Executes `run_orchestrator.py` |
| 6 | Script | Builds empty context (files read on-demand) |
| 7 | RLAPHLoop | Creates REPL environment |
| 8 | RLAPHLoop | Iterates: LLM call → Parse → Execute REPL |
| 9 | REPL | Reads files via `read_file()`, `glob_files()` |
| 10 | RLAPHLoop | Creates CausalFrame for each execution |
| 11 | FrameIndex | Stores frames in memory |

### Phase 3: Persistence (On Exit)

| Step | Component | Action |
|------|-----------|--------|
| 12 | RLAPHLoop | Calls `frame_index.save(session_id)` |
| 13 | FrameIndex | Writes `index.json` to disk |

### Phase 4: Post-Session Hooks

| Hook | When | Action |
|------|------|--------|
| `extract_frames.py` | Session Stop | Loads `index.json`, writes `frames.jsonl` |
| `compare_sessions.py` | Next Session Start | Compares with prior session, finds invalidated frames |

## CausalFrame Structure

```json
{
  "frame_id": "abc123",
  "depth": 0,
  "parent_id": null,
  "children": [],
  "query": "read_file('src/auth.py')",
  "context_slice": {
    "files": {"src/auth.py": "hash123"},
    "memory_refs": ["auth_token"],
    "tool_outputs": {},
    "token_budget": 4000
  },
  "evidence": ["frame_id_1"],
  "conclusion": "Auth uses JWT tokens",
  "confidence": 0.85,
  "invalidation_condition": "auth.py modified",
  "status": "COMPLETED",
  "branched_from": null,
  "escalation_reason": null,
  "created_at": "2026-02-20T10:30:00",
  "completed_at": "2026-02-20T10:30:05"
}
```

## Storage Layout

```
~/.claude/rlm-frames/
├── {session_id_1}/
│   ├── index.json      # FrameIndex snapshot (from RLAPHLoop)
│   ├── frames.jsonl    # CausalFrames (from extract_frames hook)
│   ├── artifacts.json  # SessionArtifacts (session metadata)
│   └── tools.jsonl     # Tool outputs (from PostToolUse hook)
├── {session_id_2}/
│   └── ...
└── {session_id_n}/
    └── ...
```

## Session ID Coordination

To ensure all artifacts land in the same folder:

1. **PostToolUse hook** (`capture_output.py`) receives `session_id` from Claude Code via stdin
2. Hook writes `session_id` to coordination file: `~/.claude/rlm-frames/.current_session`
3. **Orchestrator** (`run_orchestrator.py`) reads coordination file and uses the same `session_id`
4. **Stop hook** (`extract_frames.py`) reads coordination file to find frames

This ensures:
- Tool outputs → `{session_id}/tools.jsonl`
- RLM frames → `{session_id}/index.json`
- All artifacts in ONE folder per Claude session

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

## Key Design Decisions

1. **Context Isolation**: Main LLM never reads file contents, only passes paths
2. **On-Demand Reading**: REPL reads files only when needed
3. **Immediate Persistence**: RLAPHLoop saves frames on exit (safety backup)
4. **Hook Integration**: Hooks load from disk for proper cross-session flow
5. **JSONL Format**: Human-readable, append-only, zero dependencies
