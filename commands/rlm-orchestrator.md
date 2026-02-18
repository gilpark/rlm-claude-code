---
name: rlm-orchestrator
description: |
  RLM-style orchestration using subagent isolation.
  Context stays external - subagents read files in their own context.
---

# RLM Orchestrator

Run RLM-style analysis with context isolation via subagents.

## Usage

`/rlm-orchestrator [task description]`

## How It Works

```
Main Context (stays small):
  - Original query
  - File paths found via Glob/Grep
  - Final answer from subagent

Subagent Context (isolated):
  - Task description
  - File contents (reads itself)
  - Analysis and reasoning
  - Returns only answer to main
```

## Instructions

When this command is invoked, follow this RLM workflow:

### Step 1: Find Relevant Files (stays in main context)

Use Glob and Grep to find relevant files. Do NOT read file contents yet.

```bash
# Find relevant files
Glob: src/**/*.py
Grep: "auth|login|token" in src/
```

### Step 2: Spawn Subagent with File Paths Only

Pass file PATHS to subagent, not contents. The subagent will read files in its own isolated context.

```
Task(
  subagent_type="rlm-claude-core-rand:rlm-worker",
  prompt="Task: <user's task>

Files to analyze:
- src/auth/login.py
- src/auth/token.py
- src/api/middleware.py

Read these files and answer the task. Return ONLY your final answer.",
  description="RLM analysis"
)
```

### Step 3: Report Answer

The subagent returns a small answer. Report it to the user.

## Key Rules

1. **Main context stays small** - Only file paths, not contents
2. **Subagent reads files** - Context bloat happens in subagent, not main
3. **Pass paths, not contents** - Never paste file contents in Task prompt
4. **Sequential is fine** - Run subagents one at a time if multiple needed

## Example

User: `/rlm-orchestrator explain the authentication flow`

You:
1. Glob/Grep to find auth-related files
2. Spawn Task with file paths:
   ```
   Task(prompt="Explain the authentication flow.
   Files: src/auth/login.py, src/auth/token.py
   Read these and explain the flow.")
   ```
3. Report the answer

## For Complex Tasks (Multiple Subagents)

If the task is large, spawn multiple subagents sequentially:

```
# Main context only has: file paths + answers
answer1 = Task("Analyze src/auth/")  # Wait for result
answer2 = Task("Analyze src/api/")   # Wait for result
final = Synthesize(answer1, answer2)
```

## Related

- `/rlm` - Configure RLM mode
- `/simple` - Bypass RLM
