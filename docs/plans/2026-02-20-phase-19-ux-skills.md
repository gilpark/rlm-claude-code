# Phase 19: UX & Skills Standardization

**Status:** Draft for Review (v2 - incorporates feedback)
**Priority:** MEDIUM
**Dependencies:** Phase 18 (Tree + Evidence + Recursion) ✓, Phase 18.5 (Structured Output)

---

## Goal

Make CausalFrame feel like a real plugin with discoverable slash commands and clean library interface.

**Key outcomes:**
1. `/causal` prefix for all commands (discoverable, with `/rlm` alias)
2. Skills use `RLMSubAgent` pattern (reusable personas, no subprocess)
3. Transparency via `--verbose` / `--depth` / `--last` flags
4. SessionStart surfaces invalidated frames + proactive resume suggestion

---

## Design Principles

### 1. Slash Commands as Core UX

Single router file, discoverable via `/causal` prefix (with `/rlm` alias).

```
/causal analyze <target> [--scope security] [--depth 3] [--verbose]
/causal summarize <target>                  # Quick summary
/causal status [topic] [--last]             # Query valid frames
/causal resume <frame_id>                   # Resume branch
/causal tree [--last]                       # Visualize tree
/causal clear-cache                         # Force fresh ContextMap
/causal help                                # Show all commands
```

**Aliases:** `/causal` = `/rlm` = `/cf` (user preference)

### 2. RLMSubAgent Pattern

Instead of raw `run_rlaph()` calls, introduce `RLMSubAgent` — a reusable wrapper with persona/prompt overrides.

**Benefits:**
- Specialized agents (analyzer, summarizer, debugger)
- No subprocess overhead
- Shared memory (ContextMap)
- Composable & maintainable

```python
# Skills call specialized agents, not raw loop
analyzer_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="analyzer",
        system_prompt_override="You are a detailed code analyst...",
        default_max_depth=4,
    )
)

result = await analyzer_agent.run(query="Analyze auth.py")
```

### 3. Flag Parsing (Simple & Reliable)

Use `shlex` + manual parsing (zero deps, reliable):

```python
import shlex

def parse_flags(args_str: str) -> dict:
    """Parse --flag value pairs from args string."""
    tokens = shlex.split(args_str)
    result = {"_positional": []}

    i = 0
    while i < len(tokens):
        if tokens[i].startswith("--"):
            key = tokens[i][2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                result[key] = tokens[i + 1]
                i += 2
            else:
                result[key] = True
                i += 1
        else:
            result["_positional"].append(tokens[i])
            i += 1

    return result
```

### 4. Session Targeting

| Flag | Behavior |
|------|----------|
| (none) | Default to most recent session |
| `--last` | Explicit alias for most recent |
| `--session ID` | Target specific session by ID |

### 5. Output Formatting

**Markdown tables with emoji icons:**

| Icon | Meaning |
|------|---------|
| ✓ | COMPLETED (valid) |
| ✗ | INVALIDATED |
| ⏸ | SUSPENDED |
| 🔄 | RUNNING |

**Example:**
```
## Frame Status

| Frame | Query | Status | Confidence |
|-------|-------|--------|------------|
| ✓ 82ab3024 | Analyze auth.py... | COMPLETED | 0.9 |
| ✗ f4c81a9e | Review OAuth... | INVALIDATED | 0.8 |
| ⏸ 3d7b1c2f | Debug login... | SUSPENDED | 0.7 |

**3 frames invalidated.** Try `/causal resume` or `/causal status --full`?
```

### 6. Verbose Mode Expansion

When `--verbose` is set, show:

```python
if verbose:
    print(f"[RLM] Frame: {frame.frame_id[:8]}")
    print(f"[RLM]   depth: {frame.depth}")
    print(f"[RLM]   canonical_task: {frame.canonical_task}")
    print(f"[RLM]   invalidation_condition: {frame.invalidation_condition}")
    print(f"[RLM]   evidence: {frame.evidence}")
```

---

## Implementation Tasks

### Task 1: `run_rlaph()` Library Function

**File:** `src/repl/rlaph_loop.py`

```python
async def run_rlaph(
    query: str,
    working_dir: Path | str | None = None,
    session_id: str | None = None,
    max_depth: int = 3,
    verbose: bool = False,
    context: SessionContext | None = None,
) -> RLPALoopResult:
    """Library interface for running RLAPH loop."""
    loop = RLAPHLoop(max_depth=max_depth, verbose=verbose)
    ctx = context or SessionContext()
    wd = Path(working_dir) if working_dir else Path.cwd()
    return await loop.run(query, ctx, wd, session_id)
```

### Task 2: `RLMSubAgent` Class

**File:** `src/agents/sub_agent.py` (NEW)

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.repl.rlaph_loop import RLAPHLoop, RLPALoopResult
from src.frame.canonical_task import CanonicalTask
from src.types import SessionContext


@dataclass
class RLMSubAgentConfig:
    """Configuration for a specialized sub-agent."""
    name: str                         # e.g., "analyzer", "debugger"
    system_prompt_override: str = ""  # Optional specialized prompt
    default_max_depth: int = 3
    default_scope: str = "overview"
    verbose: bool = False


class RLMSubAgent:
    """Reusable sub-agent wrapper around RLAPHLoop."""

    def __init__(self, config: RLMSubAgentConfig):
        self.config = config
        self.loop = RLAPHLoop(
            max_depth=config.default_max_depth,
            verbose=config.verbose,
        )

    async def run(
        self,
        query: str,
        working_dir: Path | None = None,
        session_id: str | None = None,
        canonical_task: CanonicalTask | None = None,
        max_depth: Optional[int] = None,
        context: SessionContext | None = None,
    ) -> RLPALoopResult:
        """Run this sub-agent with optional overrides."""
        effective_depth = max_depth or self.config.default_max_depth
        effective_task = canonical_task or CanonicalTask(
            task_type="analyze",
            target="**/*",
            analysis_scope=self.config.default_scope,
            params={}
        )

        # Inject persona into query if override exists
        full_query = query
        if self.config.system_prompt_override:
            full_query = f"{self.config.system_prompt_override}\n\n{query}"

        # Create new loop with effective depth
        loop = RLAPHLoop(max_depth=effective_depth, verbose=self.config.verbose)
        ctx = context or SessionContext()
        wd = Path(working_dir) if working_dir else Path.cwd()

        result = await loop.run(full_query, ctx, wd, session_id)

        # Prefix answer with agent name for clarity
        result.answer = f"[{self.config.name.upper()}] {result.answer}"

        return result
```

### Task 3: Pre-defined Sub-Agents

**File:** `src/agents/presets.py` (NEW)

```python
"""Pre-configured sub-agents for common tasks."""
from src.agents.sub_agent import RLMSubAgent, RLMSubAgentConfig

analyzer_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="analyzer",
        system_prompt_override="You are a detailed code analyst. Focus on architecture, structure, and correctness.",
        default_max_depth=4,
        default_scope="overview",
    )
)

summarizer_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="summarizer",
        system_prompt_override="You are a concise summarizer. Keep answers short, structured, and to the point.",
        default_max_depth=2,
        default_scope="overview",
    )
)

debugger_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="debugger",
        system_prompt_override="You are a bug hunter. Focus on reproduction steps, root causes, and fixes.",
        default_max_depth=5,
        default_scope="correctness",
    )
)

security_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="security",
        system_prompt_override="You are a security auditor. Focus on vulnerabilities, attack vectors, and mitigations.",
        default_max_depth=4,
        default_scope="security",
    )
)
```

### Task 4: `/causal` Router Skill

**File:** `skills/causal.md` (or update existing `causeway:causal`)

```markdown
---
name: causal
description: CausalFrame commands for causal reasoning and frame navigation
invocable: true
---

CausalFrame slash command router. Dispatches to sub-commands.

<!-- Routing logic handled by Python handler below -->
```

**Handler implementation:**

```python
import shlex
from src.agents.presets import analyzer_agent, summarizer_agent, debugger_agent, security_agent
from src.frame.canonical_task import CanonicalTask
from src.frame.frame_store import FrameStore

# Command registry (for dynamic help)
COMMANDS = {
    "analyze": {
        "description": "Run detailed analysis on target",
        "example": "/causal analyze src/auth.py --scope security",
        "agent": analyzer_agent,
    },
    "summarize": {
        "description": "Quick summary of target",
        "example": "/causal summarize src/frame/",
        "agent": summarizer_agent,
    },
    "debug": {
        "description": "Debug issues in target",
        "example": "/causal debug src/auth/login.py",
        "agent": debugger_agent,
    },
    "status": {
        "description": "Show valid/invalidated frames",
        "example": "/causal status auth --last",
        "agent": None,  # Special handler
    },
    "resume": {
        "description": "Resume suspended branch",
        "example": "/causal resume 82ab3024",
        "agent": None,  # Special handler
    },
    "tree": {
        "description": "Visualize frame tree",
        "example": "/causal tree --last",
        "agent": None,  # Special handler
    },
    "clear-cache": {
        "description": "Force fresh ContextMap",
        "example": "/causal clear-cache",
        "agent": None,  # Special handler
    },
    "help": {
        "description": "Show all commands",
        "example": "/causal help",
        "agent": None,  # Special handler
    },
}

def parse_flags(args_str: str) -> dict:
    """Parse --flag value pairs from args string."""
    tokens = shlex.split(args_str) if args_str else []
    result = {"_positional": []}

    i = 0
    while i < len(tokens):
        if tokens[i].startswith("--"):
            key = tokens[i][2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                result[key] = tokens[i + 1]
                i += 2
            else:
                result[key] = True
                i += 1
        else:
            result["_positional"].append(tokens[i])
            i += 1

    return result

def get_session_id(args: dict) -> str | None:
    """Determine target session from args."""
    if args.get("session"):
        return args["session"]
    if args.get("last"):
        return FrameStore.find_most_recent_session()
    return FrameStore.find_most_recent_session()  # Default to most recent

async def handle_causal_command(args_str: str) -> str:
    """Main dispatcher for /causal commands."""
    args = parse_flags(args_str)
    positional = args.get("_positional", [])

    command = positional[0] if positional else "help"
    target = positional[1] if len(positional) > 1 else "**/*"

    verbose = args.get("verbose", False)
    depth = int(args.get("depth", 3))
    scope = args.get("scope", "overview")

    if command == "help":
        return generate_help_text()

    elif command in ("analyze", "summarize", "debug"):
        agent_config = COMMANDS[command]
        agent = agent_config["agent"]

        if command == "analyze" and scope == "security":
            agent = security_agent

        canonical_task = CanonicalTask(
            task_type=command,
            target=[target],
            analysis_scope=scope,
        )

        result = await agent.run(
            query=f"{command.capitalize()} {target}",
            max_depth=depth,
            canonical_task=canonical_task,
        )
        return result.answer

    elif command == "status":
        return await cmd_status(topic=target, args=args)

    elif command == "resume":
        frame_id = target if target != "**/*" else None
        return await cmd_resume(frame_id=frame_id, args=args)

    elif command == "tree":
        return await cmd_tree(args=args)

    elif command == "clear-cache":
        return cmd_clear_cache()

    else:
        return f"Unknown command: {command}. Try `/causal help`"

def generate_help_text() -> str:
    """Dynamically generate help from command registry."""
    help_text = "## /causal Commands\n\n"
    help_text += "| Command | Description | Example |\n"
    help_text += "|---------|-------------|----------|\n"

    for cmd, info in COMMANDS.items():
        help_text += f"| {cmd} | {info['description']} | `{info['example']}` |\n"

    help_text += "\n### Flags\n\n"
    help_text += "| Flag | Effect |\n"
    help_text += "|------|--------|\n"
    help_text += "| `--verbose` | Show recursion logs and frame details |\n"
    help_text += "| `--depth N` | Max recursion depth (default: 3) |\n"
    help_text += "| `--scope X` | Analysis scope (correctness, security, etc.) |\n"
    help_text += "| `--last` | Target most recent session |\n"
    help_text += "| `--session ID` | Target specific session |\n"

    return help_text
```

### Task 5: `/causal status` Handler

```python
async def cmd_status(topic: str | None, args: dict) -> str:
    """Show valid/invalidated frames with emoji icons."""
    session_id = get_session_id(args)
    index = FrameStore.load(session_id)

    if not index:
        return "No frames found. Run `/causal analyze` first."

    frames = list(index._frames.values())

    # Filter by topic if provided
    if topic and topic != "**/*":
        frames = [f for f in frames if topic.lower() in f.query.lower()]

    # Group by status
    completed = [f for f in frames if f.status == FrameStatus.COMPLETED]
    invalidated = [f for f in frames if f.status == FrameStatus.INVALIDATED]
    suspended = [f for f in frames if f.status == FrameStatus.SUSPENDED]

    output = "## Frame Status\n\n"
    output += f"**Session:** `{session_id}`\n\n"
    output += "| Status | Frame | Query | Confidence |\n"
    output += "|--------|-------|-------|------------|\n"

    for f in completed[:5]:
        query_short = f.query[:40] + "..." if len(f.query) > 40 else f.query
        output += f"| ✓ | `{f.frame_id[:8]}` | {query_short} | {f.confidence:.1f} |\n"

    for f in invalidated[:3]:
        query_short = f.query[:40] + "..." if len(f.query) > 40 else f.query
        output += f"| ✗ | `{f.frame_id[:8]}` | {query_short} | {f.confidence:.1f} |\n"

    for f in suspended[:3]:
        query_short = f.query[:40] + "..." if len(f.query) > 40 else f.query
        output += f"| ⏸ | `{f.frame_id[:8]}` | {query_short} | {f.confidence:.1f} |\n"

    # Proactive suggestion
    if invalidated:
        output += f"\n**{len(invalidated)} frames invalidated.** Try `/causal resume` or `/causal status --full`?\n"

    return output
```

### Task 6: `/causal tree` Handler

```python
async def cmd_tree(args: dict) -> str:
    """Visualize frame tree with emoji icons."""
    session_id = get_session_id(args)
    index = FrameStore.load(session_id)

    if not index:
        return "No frames found."

    def status_icon(frame) -> str:
        if frame.status == FrameStatus.COMPLETED:
            return "✓"
        elif frame.status == FrameStatus.INVALIDATED:
            return "✗"
        elif frame.status == FrameStatus.SUSPENDED:
            return "⏸"
        return "🔄"

    def render_tree(parent_id: str | None, indent: int = 0) -> list[str]:
        lines = []
        for f in index._frames.values():
            if f.parent_id == parent_id:
                icon = status_icon(f)
                prefix = "  " * indent + ("└── " if indent > 0 else "")
                query_short = f.query[:30] + "..." if len(f.query) > 30 else f.query
                lines.append(f"{prefix}{icon} `{f.frame_id[:8]}` (depth={f.depth}) {query_short}")
                lines.extend(render_tree(f.frame_id, indent + 1))
        return lines

    output = f"## Frame Tree\n\n**Session:** `{session_id}`\n\n```\n"
    output += "\n".join(render_tree(None))
    output += "\n```\n"

    return output
```

### Task 7: `/causal resume` Handler

```python
async def cmd_resume(frame_id: str | None, args: dict) -> str:
    """Resume suspended/invalidated branch."""
    session_id = get_session_id(args)
    index = FrameStore.load(session_id)

    if not index:
        return "No frames found."

    if not frame_id:
        # Find most recent invalidated frame
        invalidated = [f for f in index._frames.values()
                       if f.status == FrameStatus.INVALIDATED]
        if invalidated:
            frame_id = invalidated[0].frame_id
        else:
            return "No invalidated frames to resume."

    frame = index.get(frame_id)
    if not frame:
        return f"Frame `{frame_id}` not found."

    # Re-run with preserved intent
    if frame.canonical_task:
        query = f"Resume: {frame.canonical_task.task_type} {frame.canonical_task.target}"
    else:
        query = f"Resume: {frame.query}"

    result = await analyzer_agent.run(
        query=query,
        session_id=session_id,  # Continue in same session
    )

    return f"## Resumed Frame `{frame_id[:8]}`\n\n{result.answer}"
```

### Task 8: `/causal clear-cache` Handler

```python
def cmd_clear_cache() -> str:
    """Force fresh ContextMap for testing."""
    from src.frame.context_map import ContextMap

    # Clear any cached ContextMap
    ContextMap.clear_cache()

    return "✓ ContextMap cache cleared. Next run will re-scan files."
```

### Task 9: SessionStart Enhancement

**File:** `hooks/session_start.py`

```python
def session_start_hook(context):
    """Load prior session and surface invalidated frames + proactive suggestion."""
    prior_session = FrameStore.find_most_recent_session()
    if not prior_session:
        return

    index = FrameStore.load(prior_session)
    if not index:
        return

    invalidated = [f for f in index._frames.values()
                   if f.status == FrameStatus.INVALIDATED]

    if invalidated:
        print("## ⚠️ Invalidated Frames from Prior Session\n")
        print(f"**Session:** `{prior_session}`\n")

        for f in invalidated[:5]:
            desc = f.invalidation_condition.get("description", "Unknown reason")
            print(f"- ✗ `{f.frame_id[:8]}`: {desc}")

        if len(invalidated) > 5:
            print(f"\n... and {len(invalidated) - 5} more.")

        print("\n**Suggestion:** Use `/causal resume` to re-run invalidated frames.\n")
```

### Task 10: Verbose Mode Expansion

**File:** `src/repl/rlaph_loop.py`

```python
# In frame creation, when verbose=True:
if self._verbose:
    print(f"[RLM] Frame created:")
    print(f"[RLM]   id: {frame.frame_id}")
    print(f"[RLM]   depth: {frame.depth}")
    print(f"[RLM]   parent: {frame.parent_id}")
    if frame.canonical_task:
        print(f"[RLM]   canonical_task: {frame.canonical_task}")
    print(f"[RLM]   invalidation_condition: {frame.invalidation_condition}")
    print(f"[RLM]   evidence: {frame.evidence}")
```

---

## File Changes Summary

| File | Change | Lines |
|------|--------|-------|
| `src/repl/rlaph_loop.py` | Add `run_rlaph()` + verbose expansion | +20 |
| `src/agents/sub_agent.py` | NEW: RLMSubAgent class | +50 |
| `src/agents/presets.py` | NEW: Pre-configured agents | +40 |
| `src/agents/__init__.py` | NEW: Package init | +5 |
| `skills/causal.md` | Router skill | +10 |
| `src/skills/causal_router.py` | Command handlers | +150 |
| `hooks/session_start.py` | Enhancement | +20 |
| `src/frame/frame_store.py` | Add `find_most_recent_session()` | +10 |

**Total:** ~305 lines of new code

---

## Testing Plan

1. **Unit tests for flag parsing:**
   ```python
   assert parse_flags("analyze auth.py --depth 3") == {
       "_positional": ["analyze", "auth.py"],
       "depth": "3",
   }
   ```

2. **Unit tests for RLMSubAgent:**
   ```python
   agent = RLMSubAgent(RLMSubAgentConfig(name="test"))
   result = await agent.run("What is 2+2?")
   assert "4" in result.answer
   ```

3. **Integration tests for `/causal`:**
   - `/causal help` → shows all commands
   - `/causal analyze src/frame/causal_frame.py` → returns analysis
   - `/causal status` → shows frames with icons
   - `/causal tree` → shows tree structure

4. **E2E test:**
   - Run `/causal analyze src/auth.py`
   - Modify auth.py
   - `/causal status` shows invalidated
   - `/causal resume` re-runs

---

## Success Criteria

- [ ] `/causal help` shows all commands (dynamically generated)
- [ ] `/causal analyze <file>` runs RLAPH and returns result
- [ ] `/causal status` lists frames with emoji icons
- [ ] `/causal tree` shows frame hierarchy
- [ ] `/causal resume` re-runs invalidated frames
- [ ] `/causal clear-cache` forces fresh ContextMap
- [ ] Skills use RLMSubAgent pattern (no subprocess)
- [ ] `--verbose` shows frame details
- [ ] `--last` targets most recent session
- [ ] SessionStart surfaces invalidated frames + proactive suggestion

---

## Open Questions (Resolved)

| Question | Decision |
|----------|----------|
| Skill file location | Single `skills/causal.md` + Python handler |
| Flag parsing | Simple `shlex` + manual parsing |
| Session targeting | Default to most recent, `--last` alias |
| Output format | Markdown tables + emoji icons |
| Error recovery | Proactive resume suggestion |

---

## Commit Message Template

```
feat: add /causal router + RLMSubAgent pattern for skills

- RLMSubAgent wrapper with persona/prompt overrides
- Pre-configured agents (analyzer, summarizer, debugger, security)
- /causal command dispatcher with flag parsing
- Dynamic /causal help generation
- Emoji icons for frame status (✓ ✗ ⏸)
- SessionStart surfaces invalidated frames + proactive suggestion
- --verbose expansion for frame details
- --last alias for most recent session

Total: ~300 lines of new code
```
