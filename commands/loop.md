---
name: loop
description: |
  Run a task in the RLAPH loop with TDD support.
  Useful for implementing features with test-driven development.
---

# RLAPH Loop Task Runner

Run a task in the RLAPH loop for iterative development with REPL access.

## Usage

`/loop <task description>`

## Options

| Flag | Short | Description |
|------|-------|-------------|
| `--tdd` | `-t` | Enable TDD mode (write tests first) |
| `--depth N` | `-d N` | Max recursion depth (default: 2) |
| `--verbose` | `-v` | Show detailed progress |
| `--test` | | Run tests after implementation |

## Flow

```
User: /loop --tdd implement a factorial function
        ↓
1. Find relevant files (tests/, src/)
2. Compose prompt with TDD instructions
3. Run RLAPH loop
4. Verify tests pass
5. Report result
```

## Instructions

When this command is invoked, follow these steps:

### Step 1: Find Relevant Files

Use Glob to find test and source files:

```
Glob: tests/**/*.py
Glob: src/**/*.py
```

### Step 2: Compose Prompt

Build a prompt for the RLAPH loop:

**For TDD mode (`--tdd`):**
```
Task: <user's task>

TDD Instructions:
1. FIRST write a failing test in the appropriate test file
2. Run the test to confirm it fails: uv run pytest tests/path/to/test.py -v
3. Implement the minimal code to make the test pass
4. Run the test to confirm it passes
5. Refactor if needed
6. Repeat if more tests needed

Relevant files:
- /full/path/to/tests/test_module.py
- /full/path/to/src/module.py

Use the REPL to:
- Read files: files['/path/to/file.py']
- Run tests: Use Bash tool for pytest commands
- Search for patterns: search(files, 'pattern')

End with: FINAL: <summary of what was implemented and test results>
```

**For regular mode:**
```
Task: <user's task>

Relevant files (read using files['/path']):
- /full/path/to/src/module.py

Instructions:
1. Analyze the existing code
2. Implement the requested changes
3. Verify the implementation works
4. End with: FINAL: <summary of changes>
```

### Step 3: Run RLAPH Loop

```bash
uv run python scripts/run_orchestrator.py --depth 2 --verbose "<composed prompt>"
```

### Step 4: Verify (if --test flag)

If `--test` flag was provided, run tests after:

```bash
uv run pytest tests/ -v --tb=short
```

### Step 5: Report Result

Report the RLAPH loop output to the user.

## Examples

### TDD Mode

User: `/loop --tdd add input validation to the login function`

You:
1. Find test and source files
2. Compose TDD prompt
3. Run RLAPH loop
4. Report: "Implemented input validation. Tests pass."

### Regular Mode with Depth

User: `/loop --depth 3 refactor the auth module for better testability`

You:
1. Find auth module files
2. Compose refactor prompt
3. Run with depth 3 for complex analysis
4. Report the refactoring summary

### With Test Verification

User: `/loop --tdd --test implement password hashing`

You:
1. TDD prompt for password hashing
2. Run RLAPH loop
3. Run full test suite to verify
4. Report: "Password hashing implemented. All 15 tests pass."

## Key Rules

1. **TDD first**: When `--tdd`, always write failing test before implementation
2. **Verify with tests**: Use `--test` to run full test suite after
3. **Use REPL**: Let the loop read files and run commands
4. **Clear FINAL**: Ensure loop outputs FINAL: with summary

## Related

- `/rlm-orchestrator` - General RLM analysis
- `/rlm` - Configure RLM mode
- `/simple` - Bypass RLM
