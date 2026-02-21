---
name: causal
description: CausalFrame commands for causal reasoning and frame navigation
invocable: true
---

CausalFrame slash command router. Dispatches to sub-commands for causal reasoning, frame navigation, and analysis.

## Commands

| Command | Description |
|---------|-------------|
| `/causal analyze <target>` | Run detailed analysis on target |
| `/causal summarize <target>` | Quick summary of target |
| `/causal debug <target>` | Debug issues in target |
| `/causal status [topic]` | Show valid/invalidated frames |
| `/causal tree` | Visualize frame tree |
| `/causal resume <frame_id>` | Resume invalidated branch |
| `/causal clear-cache` | Force fresh ContextMap |
| `/causal help` | Show all commands |

## Flags

| Flag | Effect |
|------|--------|
| `--verbose` | Show recursion logs and frame details |
| `--depth N` | Max recursion depth (default: 3) |
| `--scope X` | Analysis scope (correctness, security, architecture) |
| `--last` | Target most recent session |
| `--session ID` | Target specific session |

## How to Handle Commands

When the user invokes `/causal <command> [args]`:

1. **Parse the command and arguments** from the user's input
2. **Execute the CLI script** to process the command:
   ```bash
   python $CLAUDE_PLUGIN_ROOT/scripts/causal_cli.py "{command} {args}"
   ```
3. **Return the output** to the user

## Pre-defined Agents

The router uses specialized agents for different task types:
- **analyzer** - Detailed code analysis (depth 4)
- **summarizer** - Concise summaries (depth 2)
- **debugger** - Bug hunting (depth 5)
- **security** - Security auditing (depth 4)

## Examples

```
/causal analyze src/auth.py --scope security
/causal summarize src/frame/
/causal debug src/auth/login.py
/causal status auth --last
/causal tree
/causal resume 82ab3024
```
