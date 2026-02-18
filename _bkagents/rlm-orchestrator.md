---
name: rlm-orchestrator
description: |
  RLM orchestrator agent. Analyzes task, finds files, runs Python orchestrator.
  Main context stays small - Python script reads files in subprocess.
tools: Glob, Grep, Bash
model: sonnet
---

# RLM Orchestrator Agent (Legacy)

> **Note**: This agent file is archived. The current implementation uses
> `commands/rlm-orchestrator.md` which runs the Python orchestrator directly.

## Current Flow

The orchestrator now uses a simplified Python-based flow:

```
User: /rlm-orchestrator explain the auth flow
        ↓
Main Claude:
  1. Interpret task
  2. Glob/Grep to find relevant files (NO reading contents)
  3. Compose prompt with task + file paths
  4. Run: uv run python scripts/run_orchestrator.py "<composed prompt>"
        ↓
Python orchestrator:
  5. LLM reasoning
  6. REPL can read files as needed
  7. Recursive sub-calls if needed
  8. Returns answer
```

## Available Options

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Print trajectory events |
| `--stream` | `-s` | Stream tokens in real-time |
| `--depth N` | `-d N` | Max recursion depth (default: 2) |
| `--validate` | | Validate dependencies |
| `--status` | | Show RLM status |

## Historical Context

The previous worker-based pattern has been replaced with a simpler
Python subprocess approach for better context isolation and reliability.

### Old Pattern (Deprecated)

~~1. Find Relevant Files~~
~~2. Spawn Worker with File Paths~~
~~3. Return Answer~~

### New Pattern (Current)

1. Find Relevant Files (Glob/Grep only)
2. Compose Prompt with Paths
3. Run Python Orchestrator
4. Report Answer

## Related

- `commands/rlm-orchestrator.md` - Current command implementation
- `scripts/run_orchestrator.py` - Python orchestrator script
