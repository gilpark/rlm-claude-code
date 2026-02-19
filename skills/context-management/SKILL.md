# Causeway Context Management Skill

## When to Activate

This skill activates when:
- Context exceeds complexity threshold (cross-file reasoning, debugging with large output, multi-module tasks)
- User explicitly requests causeway mode (`/causeway`)
- Task requires verification chains (depth=2 patterns)

## Capabilities

### Context Externalization
- Conversation history → `conversation` variable
- Cached files → `files` dict
- Tool outputs → `tool_outputs` list
- Session state → `working_memory` dict

### REPL Operations
- `peek(var, start, end)` — View portion of context
- `search(var, pattern, regex=False)` — Find patterns in context
- `summarize(var, max_tokens=500)` — Summarize via sub-call
- `llm(query, context)` — Immediate recursive LLM call

### Causal Operations
- CausalFrame tree tracking
- Invalidation cascade on file changes
- Suspended branch recovery

## Strategy Selection

### For Large Context (>80K tokens)
1. Peek first 2K chars to understand structure
2. Search for relevant patterns
3. Partition and map over chunks if needed
4. Recursive queries for semantic understanding

### For Debugging Tasks
1. Peek recent tool output for error
2. Search codebase for relevant files
3. Recursive query to analyze each candidate
4. Verify fix won't break other code (depth=2)

### For Multi-File Refactoring
1. Identify all affected files via search
2. Recursive query for each file's current state
3. Plan changes with dependency awareness
4. Track changes in CausalFrame tree

## Output Protocol

Signal completion with:
- `FINAL(answer)` — Direct answer
- `FINAL_VAR(var_name)` — Answer stored in variable

## Configuration

```json
{
  "causeway": {
    "activation": {"mode": "complexity"},
    "depth": {"default": 2, "max": 3}
  }
}
```

## References

- Design: docs/plans/2026-02-19-design.md
- Whitepaper: docs/plans/2026-02-19-whitepaper.md
- RLM Paper: https://arxiv.org/abs/2512.24601v1
