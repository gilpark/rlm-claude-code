# ADR-001: Remove PostToolUse Prompt Hook

## Status

Accepted

## Context

The RLM plugin had a `PostToolUse` prompt hook with `"matcher": ".*"` that fired after every tool call (Edit, Bash, Read, Grep, etc.). The hook injected this prompt:

> "If the tool output reveals unexpected complexity (large error traces, many files affected, cross-module dependencies), consider suggesting RLM escalation to the user..."

This caused the model to stop after each tool use and write analysis paragraphs like:

> "The edit changes logical operator token tags from `.op_and_and, .op_pipe_pipe` to `.kw_and, .kw_or` in a single location within the `checkBinary` function. This is a targeted, single-file change with no apparent cross-module dependencies or unexpected complexity. The scope is clear and localized. No RLM escalation is warranted for this straightforward token tag replacement."

**Problem**: This halted productive work. Users saw the model pause after every edit to write justifications for not escalating to RLM, wasting time and breaking flow.

## Decision

Remove the `PostToolUse` prompt hook entirely. Complexity classification happens once per user prompt via the `UserPromptSubmit` command hook, not per tool call.

## Consequences

### Positive
- Model no longer halts after each tool to evaluate complexity
- Faster, more fluid interaction
- Classification happens at the right granularity (per prompt, not per tool)

### Negative
- If a tool output reveals unexpected complexity mid-turn, the model won't be explicitly prompted to reconsider RLM mode
- Mitigation: The model can still choose to escalate based on tool output; it just won't be prompted to evaluate every time

### Neutral
- The `PostToolUse` array in hooks.json is now empty for RLM
- Other plugins can still use PostToolUse hooks

## Alternatives Considered

1. **Narrow the matcher**: Only fire on specific tools (e.g., Bash with errors). Rejected because any per-tool prompt still introduces latency and cognitive overhead.

2. **Make it non-blocking**: Change hook type from "prompt" to something lighter. Not possible with current hook system — prompts inject text that the model processes.

3. **Conditional firing based on output size**: Only inject prompt if tool output exceeds threshold. Adds complexity and still causes halting on large outputs.

## References

- SPEC-17: Complexity Classification
- GitHub commit: 034c899
