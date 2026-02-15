---
name: rlm-orchestrator
description: Invoke the RLM orchestrator agent for complex context management tasks. Use when you need multi-file context management across large codebases, parallel tracking of multiple work streams, complex reasoning with context externalization, or REPL-based context decomposition.
context: fork
agent: rlm-claude-code:rlm-orchestrator
---

# RLM Orchestrator

Execute this task with full tool access using the RLM orchestrator agent:

$ARGUMENTS

## What the Agent Does

1. Externalizes conversation context to Python variables
2. Uses REPL operations (peek, search, summarize) for efficient context access
3. Can spawn recursive sub-queries for deep analysis
4. Manages depth budgets and model cascades

## Related

- `/rlm` - Configure RLM mode settings
- `/rlm status` - Check current configuration
- `/simple` - Bypass RLM for simple operations
