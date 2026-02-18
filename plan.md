# RLM Implementation Plan

## Current State (Feb 2026)

Working architecture:
- Agent-based: Orchestrator finds files → spawns Worker → Worker reads in isolated context → returns answer
- Python script: `run_orchestrator.py` with `--files`, `--dir` args, full REPL functions
- Hooks: SessionStart, UserPromptSubmit, PreCompact, Stop

## Short Term: Streaming + Visibility ✅ DONE

**Goal**: Add progress feedback during long LLM subprocess calls

### Tasks
- [x] Add progress callback to `ClaudeHeadlessClient.complete()`
- [x] Emit progress events every 10s while waiting
- [x] Test with quick query (works, no events for fast responses)

### Files modified
- `src/api_client.py` - Added `ProgressCallback` type and progress tracking

### Usage
```python
def on_progress(elapsed: int, timeout: float):
    print(f"[LLM] Waiting... {elapsed}s")

response = await client.complete(
    messages=[...],
    progress_callback=on_progress,
)
```

### Expected behavior (for long calls >10s)
```
[LLM] Calling sonnet...
[LLM] Waiting... 10s
[LLM] Waiting... 20s
[LLM] Response received
```

## Mid Term: Claude Agent SDK Migration

**Goal**: Migrate orchestrator loop to Claude Agent SDK for cleaner architecture

### Benefits
- In-process execution (no spawn overhead)
- Native streaming events
- Better structured output parsing
- Built-in checkpointing/resume
- Full asyncio control with `asyncio.gather()`

### Tasks
- [ ] Install and evaluate `claude-agent-sdk`
- [ ] Design SDK-native orchestrator class
- [ ] Implement subagent spawning with SDK API
- [ ] Migrate REPL sandbox to work with SDK context
- [ ] Port existing tests to SDK-based tests
- [ ] Benchmark: subprocess vs SDK performance

### Files to create/modify
- `src/sdk_orchestrator.py` - New SDK-based orchestrator
- `src/sdk_client.py` - SDK wrapper with RLM patterns
- `tests/integration/test_sdk_orchestrator.py` - SDK tests

### Migration path
1. Keep existing subprocess implementation as fallback
2. Add SDK implementation in parallel
3. Feature flag to switch between implementations
4. Gradual rollout after testing

## Long Term: Advanced RLM Features

### Potential enhancements
- [ ] Memory evolution across sessions (SPEC-03)
- [ ] Strategy cache from successful trajectories
- [ ] Adaptive depth based on complexity
- [ ] Multi-model routing (Opus for hard, Haiku for simple)
- [ ] Budget tracking and cost optimization (SPEC-05)

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601v1)
- [RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk)
