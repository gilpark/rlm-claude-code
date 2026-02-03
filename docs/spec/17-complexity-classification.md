# SPEC-17: Complexity Classification

## Overview

[SPEC-17.01] The complexity-check binary classifies user prompts to determine RLM activation mode. It replaces the DP-phase-only stub with real signal extraction and scoring.

## Signal Categories

[SPEC-17.02] The classifier extracts 14 signal categories from user prompts:

### High-Signal (Instant Activation)

[SPEC-17.03] These signals trigger immediate RLM activation:

| Signal | Pattern Examples | Rationale |
|--------|-----------------|-----------|
| Cross-context reasoning | `why.*when`, `how.*relate`, `trace.*across` | Requires connecting information across files |
| Debugging task | `error`, `debug`, `stacktrace`, `troubleshoot` | Needs systematic root cause analysis |
| Exhaustive search | `find all`, `every instance`, `comprehensive` | Requires systematic enumeration |
| Security review | `vulnerability`, `audit`, `injection`, `xss` | Needs careful multi-file analysis |
| Architecture analysis | `architecture`, `refactor`, `how does.*work` | Requires system-wide understanding |
| Multi-module task | Multiple file extensions + module pairs | Cross-cutting changes |

### Accumulative Signals

[SPEC-17.04] These signals contribute to a cumulative score (threshold ≥ 2):

| Signal | Weight | Pattern Examples |
|--------|--------|-----------------|
| Multiple files | +2 | 2+ file extensions mentioned |
| Temporal reasoning | +2 | `before`, `after`, `history`, `commit` |
| Pattern search | +2 | `find.*where`, `how many`, `list all` |
| User wants thorough | +2 | `make sure`, `be careful`, `thorough` |
| Prior confusion | +2 | Previous turn showed confusion indicators |
| Multi-domain context | +1 | >2 active modules in context |
| Task continuation | +1 | `continue`, `same`, `also` at start |

### Suppression Signals

[SPEC-17.05] Fast intent suppresses activation:
- `quick`, `just show`, `briefly`, `simple answer`

## Fast-Path Bypass

[SPEC-17.06] Trivial prompts bypass classification entirely:
- `git status`, `git log`, `git diff`, `git branch`
- `yes`, `no`, `ok`, `sure`, `thanks`
- `run pytest`, `run npm`, `run cargo`
- `show <file>`, `cat <file>`

## Mode Selection

[SPEC-17.07] RLM mode is selected based on activation + DP phase:

| DP Phase | Suggested Mode |
|----------|---------------|
| spec, review | thorough |
| test, implement, orient, decide | balanced |
| unknown / none | balanced |

[SPEC-17.08] If not activated, mode defaults to `micro`.

## Integration

[SPEC-17.09] The complexity-check binary:
1. Reads user prompt from stdin (JSON with `user_prompt` field)
2. Checks `RLM_DISABLED` env var — exits early if set
3. Checks DP rigor level — skips if "trivial"
4. Applies fast-path bypass for trivial prompts
5. Extracts signals and computes score
6. Returns approval with mode suggestion in reason field
7. Emits `rlm_activation_suggested` event

[SPEC-17.10] Hook placement: `UserPromptSubmit` (once per prompt), not `PreToolUse` (per tool).

## Implementation

[SPEC-17.11] Go package: `internal/classify`
- `ExtractSignals(prompt string) Signals` — regex-based extraction
- `Score(s Signals) (int, []string)` — accumulative scoring
- `IsFastPath(prompt string) bool` — trivial query detection
- `ShouldActivate(prompt, dpPhase, rigor string) (bool, string, string)` — full decision

[SPEC-17.12] Tests: 22 unit tests covering all signal types, scoring, mode mapping.
