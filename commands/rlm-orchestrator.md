---
name: rlm-orchestrator
description: |
  RLM-style orchestration using subagent isolation.
  Context stays external - subagents read files in their own context.
---

# RLM Orchestrator

Run RLM-style analysis with context isolation via the Python orchestrator.

## Usage

`/rlm-orchestrator [task description]`

## Flow

```
User: /rlm-orchestrator explain the auth flow
        ↓
Main Claude (this command):
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
        ↓
Main Claude:
  9. Report answer to user
```

## Options

| Flag | Short | Description |
|------|-------|-------------|
| `--legacy` | | Use legacy orchestrator (default: RLAPH) |
| `--verbose` | `-v` | Print trajectory events and progress |
| `--stream` | `-s` | Stream tokens in real-time (experimental) |
| `--depth N` | `-d N` | Max recursion depth (default: 2) |
| `--validate` | | Validate dependencies and exit |
| `--status` | | Show RLM status and exit |
| `--bypass` | | Set bypass flag |

## Instructions

When this command is invoked, follow these steps:

### Step 1: Find Relevant Files

Use Glob and Grep to find relevant files. **Do NOT read file contents.**

```
Glob: src/**/*.py
Grep: "auth|login|token" in src/
```

### Step 2: Compose Prompt with Paths

Build a prompt that includes the task and file paths:

```
Task: <user's original task>

Relevant files (read as needed):
- /full/path/to/src/auth/login.py
- /full/path/to/src/auth/token.py
- /full/path/to/src/api/middleware.py
```

### Step 3: Run Python Orchestrator

Use Bash to run the Python script with the composed prompt:

```bash
cd ${CLAUDE_PLUGIN_ROOT}
uv run python scripts/run_orchestrator.py --stream --verbose "<composed prompt>"
```

### Step 4: Report Answer

The Python script returns the answer. Report it to the user.

## Key Rules

1. **Main context stays small** - Only use Glob/Grep, never Read
2. **Pass paths in prompt** - Let Python script read files
3. **One Bash call** - Run Python script once with full prompt
4. **Report only** - Don't analyze, just pass through the result

## Examples

### Basic Usage

User: `/rlm-orchestrator explain how context management works`

You:
1. Grep for "context" in src/ → find context_manager.py, types.py
2. Compose:
   ```
   Task: explain how context management works

   Relevant files:
   - /path/to/src/context_manager.py
   - /path/to/src/types.py
   ```
3. Run: `uv run python scripts/run_orchestrator.py "Task: ..."`
4. Report the answer

### With Verbose Output

User: `/rlm-orchestrator --verbose analyze the auth flow`

Shows trajectory events like:
```
[RLM] depth=0/2 • routing: unknown → sonnet
[REASON] depth=0 tokens=5549
[REPL] result = ...
```

### With RLAPH Mode (Recommended)

User: `/rlm-orchestrator --rlaph analyze the auth flow`

RLAPH mode uses a clean loop where `llm()` returns actual results immediately (not deferred). This is faster and easier to debug.

### Direct Script Usage

```bash
# RLAPH mode (recommended)
uv run python scripts/run_orchestrator.py --rlaph "analyze auth"

# Verbose mode
uv run python scripts/run_orchestrator.py --stream --verbose "analyze auth"

# With streaming (experimental)
uv run python scripts/run_orchestrator.py --stream "explain architecture"

# Custom depth
uv run python scripts/run_orchestrator.py --depth 3 "complex analysis"

# Utility commands
uv run python scripts/run_orchestrator.py --status
uv run python scripts/run_orchestrator.py --validate
```

## Related

- `/rlm` - Configure RLM mode
- `/simple` - Bypass RLM
