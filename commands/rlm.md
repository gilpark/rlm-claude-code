---
name: rlm
description: Toggle or configure RLM mode settings. Use /rlm status to check configuration, /rlm on|off|auto to change activation mode, /rlm activate to launch the orchestrator.
argument-hint: [status|on|off|auto|activate|verbose|debug]
---

# RLM Mode Configuration

Manage RLM (Recursive Language Model) mode settings.

## Arguments: $ARGUMENTS

### If argument is empty or "status":
Show current RLM status and configuration.

### If argument is "activate" or "now":
**MUST immediately invoke the orchestrator:**
```
Task(
  subagent_type="rlm-claude-code:rlm-orchestrator",
  prompt="[current conversation context]",
  description="RLM orchestration"
)
```

### If argument is "on":
Update `~/.claude/rlm-config.json` to set `activation.mode` to `"always"`.

### If argument is "off":
Update `~/.claude/rlm-config.json` to set `activation.mode` to `"manual"`.

### If argument is "auto":
Update `~/.claude/rlm-config.json` to set `activation.mode` to `"complexity"`.

### If argument is "verbose":
Update `~/.claude/rlm-config.json` to set `trajectory.verbosity` to `"verbose"`.

### If argument is "debug":
Update `~/.claude/rlm-config.json` to set `trajectory.verbosity` to `"debug"`.

## Configuration File

Settings are stored in `~/.claude/rlm-config.json`:
- `activation.mode`: "complexity" | "always" | "manual"
- `depth.default`: 2
- `trajectory.verbosity`: "minimal" | "normal" | "verbose" | "debug"

## Related

- `/rlm-claude-code:rlm-orchestrator` — Launch RLM orchestrator agent
- `/rlm-claude-code:simple` — Bypass RLM for simple operations
