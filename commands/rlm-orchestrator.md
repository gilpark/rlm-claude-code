---
name: rlm-orchestrator
description: |
  Invoke the RLM orchestrator agent for complex context management tasks.
  Bypasses complexity checking - RLM activates immediately.
hooks:
  # Validate orchestrator dependencies before running
  PreToolUse:
    - matcher: "*"
      hooks:
        - type: command
          command: 'cd "${CLAUDE_PLUGIN_ROOT}" && .venv/bin/python scripts/run_orchestrator.py --validate'
          timeout: 2000
          description: "Validate orchestrator dependencies"
---

# RLM Orchestrator

Invoke the RLM orchestrator agent for complex context management tasks.

## Usage

`/rlm-orchestrator [task description]`

## When to Use

Use this when you need:
- Multi-file context management across large codebases
- Parallel tracking of multiple work streams
- Complex reasoning with context externalization
- REPL-based context decomposition

**Note**: When you explicitly call this command, complexity checking is bypassed - RLM activates immediately.

## How It Works

This command runs the RLM orchestrator which:

1. Loads context from `~/.claude/rlm-state/context.json` (written by hooks)
2. Uses Claude CLI for LLM calls (no API key needed, uses subscription)
3. Can spawn recursive sub-queries for deep analysis
4. Manages depth budgets and model cascades (Opus → Sonnet → Haiku)

## Running the Orchestrator

```bash
# Basic usage
uv run python scripts/run_orchestrator.py "analyze the auth module"

# With custom depth
uv run python scripts/run_orchestrator.py --depth 3 "complex analysis task"

# Verbose output
uv run python scripts/run_orchestrator.py --verbose "debug this code"

# View current context
uv run python scripts/run_orchestrator.py --context
```

## LLM Provider Selection

The orchestrator automatically selects the best available provider:
1. **Claude API** - If `ANTHROPIC_API_KEY` is set
2. **OpenAI API** - If `OPENAI_API_KEY` is set
3. **Claude CLI** - Default, uses your subscription (no API key needed)

## Instructions

When this skill is invoked, run the orchestrator:

```bash
uv run python scripts/run_orchestrator.py "[user's task description]"
```

Or use the Task tool to launch the RLM orchestrator agent:

```
Task(
  subagent_type="rlm-claude-code:rlm-orchestrator",
  prompt="[user's task description]",
  description="RLM orchestration"
)
```

## Related

- `/rlm` - Configure RLM mode settings
- `/rlm status` - Check current configuration
- `/simple` - Bypass RLM for simple operations
