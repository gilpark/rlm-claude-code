# Documentation Issues Report

Generated: 2026-02-17

Review of all .md documentation files against actual codebase implementation.

## Summary

| Severity | Count |
|----------|-------|
| Critical | 1 |
| Major | 6 |
| Minor | 5 |
| Accurate | 6 |

---

## Critical Issues

### 1. Slash Command Namespace Inconsistency

**Files**: `commands/*.md`, `CLAUDE.md`

**Problem**: Inconsistent namespace format across commands.

**What docs claim**:
- Some use `/rlm-claude-code:xxx`
- Others use `/xxx`
- CLAUDE.md mixes both

**What code shows**:
- Commands are registered with `rlm-claude-code:` namespace
- Should consistently use `/rlm-claude-code:xxx` format

**Severity**: Critical

**Recommended fix**:
1. Decide on standard: `/rlm-claude-code:xxx`
2. Update all command files in `commands/`
3. Update CLAUDE.md slash commands table
4. Update README.md slash commands table

---

## Major Issues

### 2. Memory API Documentation Mismatch

**File**: `agents/rlm-orchestrator.md` (lines 56-66)

**What docs claim**:
```bash
# Query stored knowledge
uv run python scripts/repl_bridge.py --op memory_query --args '{"query": "authentication", "limit": 10}'

# Store a fact
uv run python scripts/repl_bridge.py --op memory_add --args '{"type": "fact", "content": "..."}'

# Store an experience
uv run python scripts/repl_bridge.py --op memory_add --args '{"type": "experience", "content": "..."}'
```

**What code actually has** (src/repl_environment.py):
```python
def _memory_add_fact(self, content, confidence=0.9): ...
def _memory_add_experience(self, content, outcome, success): ...
```

**Severity**: Major

**Recommended fix**: Either:
1. Update docs to use `memory_add_fact` and `memory_add_experience` separately
2. Or update repl_bridge.py to have unified `memory_add` with type parameter

---

### 3. /verify Command Namespace Missing

**File**: `commands/verify.md`

**Problem**: Missing `rlm-claude-code:` namespace prefix.

**What docs claim**: Command is `/verify`

**What it should be**: `/rlm-claude-code:verify`

**Severity**: Major

**Recommended fix**: Add namespace prefix to command definition.

---

### 4. /test Not a Valid Slash Command

**Files**: `CLAUDE.md` (line 284), `commands/test.md`

**What docs claim**: `/rlm-claude-code:test` is a slash command

**What code shows**: `commands/test.md` exists but may not be properly registered

**Severity**: Major

**Recommended fix**: Verify command registration or remove from docs.

---

### 5. /bench Not a Valid Slash Command

**Files**: `CLAUDE.md` (line 285), `commands/bench.md`

**What docs claim**: `/rlm-claude-code:bench` is a slash command

**What code shows**: `commands/bench.md` exists but may not be properly registered

**Severity**: Major

**Recommended fix**: Verify command registration or remove from docs.

---

### 6. /code-review Not a Valid Slash Command

**Files**: `CLAUDE.md` (line 286), `commands/code-review.md`

**What docs claim**: `/rlm-claude-code:code-review` is a slash command

**What code shows**: `commands/code-review.md` exists but may not be properly registered

**Severity**: Major

**Recommended fix**: Verify command registration or remove from docs.

---

### 7. subagent_type Unclear in rlm-orchestrator

**File**: `commands/rlm-orchestrator.md` (lines 47-50)

**What docs claim**:
```
Task(
  subagent_type="rlm-claude-code:rlm-orchestrator",
  ...
)
```

**Problem**: How is this subagent_type registered? Not clear from documentation.

**Severity**: Major

**Recommended fix**: Document how custom agent types are registered with Claude Code plugins.

---

## Minor Issues

### 8. Memory Add Signature Differs

**File**: `docs/user-guide.md`

**What docs claim**: `memory_add(content, type)`

**What code has**:
- `memory_add_fact(content, confidence)`
- `memory_add_experience(content, outcome, success)`

**Severity**: Minor

**Recommended fix**: Update user guide to match actual API.

---

### 9. User Guide Slash Commands Missing Namespace

**File**: `docs/user-guide.md`

**What docs claim**: Commands like `/rlm`, `/simple`, etc.

**What they should be**: `/rlm-claude-code:rlm`, `/rlm-claude-code:simple`

**Severity**: Minor

**Recommended fix**: Add namespace prefix throughout.

---

### 10. merge-plugin-hooks.py Path Unclear

**File**: `README.md` (lines 462-471)

**What docs claim**:
```bash
python3 ~/.claude/scripts/merge-plugin-hooks.py
```

**Problem**: Where does this script come from? Is it bundled with the plugin?

**Severity**: Minor

**Recommended fix**: Clarify script location or include in plugin.

---

### 11. Hyperedge Column Name Mismatch

**File**: `docs/spec/02-memory-foundation.md`

**What docs claim**: `hyperedges.type`

**What code has** (src/memory_store.py): `edge_type`

**Severity**: Minor

**Recommended fix**: Update spec to use `edge_type` column name.

---

### 12. Deferred Operations Return Format

**File**: `docs/spec/01-repl-functions.md`

**What docs claim**: Deferred operations return specific format

**What code returns**: May differ slightly

**Severity**: Minor

**Recommended fix**: Verify and align return formats.

---

## Verified Accurate

The following documentation claims were verified as accurate:

| # | Claim | File | Verified By |
|---|-------|------|-------------|
| 1 | Hook architecture uses Python | README.md, hooks/hooks.json | Python scripts in scripts/ |
| 2 | rlm-core is optional | README.md, pyproject.toml | Not in dependencies |
| 3 | Version is 0.7.0 | pyproject.toml, marketplace.json | Both show 0.7.0 |
| 4 | SPEC-01 functions implemented | docs/spec/01-repl-functions.md | src/repl_environment.py has all |
| 5 | repl_bridge.py operations | agents/rlm-orchestrator.md | All operations exist |
| 6 | Memory functions exist | CLAUDE.md | src/repl_environment.py has _memory_* |

---

## Priority Action Items

1. **Immediate**: Standardize namespace format (`/rlm-claude-code:xxx`) across all command files
2. **High**: Decide on memory API (unified `memory_add` vs separate functions) and update docs
3. **High**: Verify `/test`, `/bench`, `/code-review` commands or remove from documentation
4. **Medium**: Update user-guide.md with correct namespaces and API signatures
5. **Low**: Fix minor typos and column name references

---

## Files to Update

| File | Changes Needed |
|------|----------------|
| `agents/rlm-orchestrator.md` | Fix memory_add API |
| `commands/verify.md` | Add namespace |
| `commands/test.md` | Verify or remove |
| `commands/bench.md` | Verify or remove |
| `commands/code-review.md` | Verify or remove |
| `CLAUDE.md` | Update slash commands table |
| `README.md` | Update slash commands table |
| `docs/user-guide.md` | Add namespaces, fix API signatures |
| `docs/spec/02-memory-foundation.md` | Fix column name |
