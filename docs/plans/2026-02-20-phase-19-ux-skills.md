# Phase 19: UX & Skills Standardization

**Status:** Draft for Review (v3 - SDK streaming + clarified sub-agent scope)
**Priority:** MEDIUM
**Dependencies:** Phase 18 (Tree + Evidence + Recursion) ✓, Phase 18.5 (Structured Output) ✓

---

## Goal

Make CausalFrame feel like a real plugin with discoverable slash commands, SDK streaming, and clean library interface.

**Key outcomes:**
1. `/causal` prefix for all commands (discoverable, with `/rlm` alias)
2. **SDK streaming** for real-time UX (no more spinner → typing effect)
3. Skills use `RLMSubAgent` pattern for **top-level workflows only**
4. Transparency via `--verbose` / `--depth` / `--last` flags
5. SessionStart surfaces invalidated frames + proactive resume suggestion

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

### 2. RLMSubAgent Pattern (Top-Level Workflows Only)

`RLMSubAgent` wraps **top-level RLM runs** — not every internal LLM call.

**Scope:**
| Call Type | Pattern | Why |
|-----------|---------|-----|
| `/causal analyze` → skill → RLMSubAgent.run() | **Sub-Agent** | Persona override, specialized config |
| Internal `llm(sub_query)` in REPL | **Direct LLM client** | Already managed by RLAPHLoop (depth, frames) |
| Frame creation, evidence tracking | **Direct** | Core loop responsibility |

**Benefits:**
- Specialized agents (analyzer, summarizer, debugger) for user commands
- No subprocess overhead
- Internal recursion stays fast (no extra layer)
- Composable & maintainable

```python
# Skills call specialized agents for top-level commands
analyzer_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="analyzer",
        system_prompt_override="You are a detailed code analyst...",
        default_max_depth=4,
    )
)

result = await analyzer_agent.run(query="Analyze auth.py")

# Internal llm() calls in REPL stay direct — managed by RLAPHLoop
# No sub-agent wrapping needed for recursion
```

### 3. SDK Streaming (Real-Time UX)

Replace HTTP-based `llm_client.py` with **Anthropic SDK** for streaming.

**Benefits:**
- Real-time typing effect (no spinner)
- Better UX for long-running queries
- Applies uniformly to all LLM calls (top-level + internal)

```python
from anthropic import AsyncAnthropic

class LLMClient:
    def __init__(self):
        self.client = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_AUTH_TOKEN"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
        )

    async def call_stream(self, query: str, system: str = "") -> AsyncIterable[str]:
        """Streaming LLM call for real-time UX."""
        async with self.client.messages.stream(
            model=os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": query}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
```

**Note:** Streaming applies to all calls uniformly. Sub-agent pattern is orthogonal — it's a config wrapper, not changing how LLM is called.

### 4. Flag Parsing (Simple & Reliable)

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

### Task 1: SDK Streaming Refactor

**File:** `src/repl/llm_client.py` (refactor)

Replace HTTP-based client with Anthropic SDK for streaming:

```python
from anthropic import AsyncAnthropic
import os
from typing import AsyncIterable

class LLMClient:
    def __init__(self):
        self.client = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_AUTH_TOKEN"),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            timeout=int(os.getenv("API_TIMEOUT_MS", 300000)) / 1000,
        )

    async def call_stream(
        self,
        query: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterable[str]:
        """Streaming LLM call for real-time UX."""
        model = model or os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-20250514")

        async with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": query}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def call(
        self,
        query: str,
        context: dict | None = None,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
        depth: int = 0,
    ) -> str:
        """Sync wrapper for legacy calls (non-streaming)."""
        import asyncio

        async def _collect():
            result = []
            async for chunk in self.call_stream(query, system, model, max_tokens):
                result.append(chunk)
            return "".join(result)

        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # Already in async context — use create_task or return coroutine
            return _collect()
        except RuntimeError:
            # No running loop — safe to use asyncio.run
            return asyncio.run(_collect())

    async def call_async(
        self,
        query: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Async non-streaming call."""
        result = []
        async for chunk in self.call_stream(query, system, model, max_tokens):
            result.append(chunk)
        return "".join(result)
```

**Benefits:**
- Real-time typing effect for better UX
- Uniform interface for all LLM calls
- Backward compatible with existing `call()` method
- UX markers: "Thinking..." at start, "Done" at end

### Task 2: Config File Support

**File:** `src/config.py` (add to existing or create new)

```python
import json
from pathlib import Path
from dataclasses import dataclass, field

CONFIG_PATH = Path.home() / ".claude" / "causalframe-config.json"

DEFAULT_CONFIG = {
    "default_max_depth": 3,
    "default_verbose": False,
    "status_limit": 5,
    "default_model": "sonnet",
    "status_icons": True,
}

@dataclass
class CFConfig:
    """CausalFrame user configuration."""
    default_max_depth: int = 3
    default_verbose: bool = False
    status_limit: int = 5
    default_model: str = "sonnet"
    status_icons: bool = True
    reference_dirs: list[str] = field(default_factory=list)
    auto_resume_on_invalidate: bool = False

    @classmethod
    def load(cls) -> "CFConfig":
        """Load config from file with defaults."""
        config = DEFAULT_CONFIG.copy()
        if CONFIG_PATH.exists():
            try:
                user_config = json.loads(CONFIG_PATH.read_text())
                config.update(user_config)
            except (json.JSONDecodeError, IOError):
                pass  # Use defaults on error
        return cls(
            default_max_depth=config["default_max_depth"],
            default_verbose=config["default_verbose"],
            status_limit=config["status_limit"],
            default_model=config["default_model"],
            status_icons=config["status_icons"],
        )
```

**Config file location:** `~/.claude/causalframe-config.json`

```json
{
  "default_max_depth": 3,
  "default_verbose": false,
  "status_limit": 5,
  "default_model": "sonnet",
  "status_icons": true
}
```

### Task 3: `run_rlaph()` Library Function

**File:** `src/repl/rlaph_loop.py`

```python
async def run_rlaph(
    query: str,
    working_dir: Path | str | None = None,
    session_id: str | None = None,
    max_depth: int = 3,
    verbose: bool = False,
    context: SessionContext | None = None,
    stream: bool = False,
) -> RLPALoopResult | AsyncIterable[str]:
    """
    Library interface for running RLAPH loop.

    Args:
        stream: If True, yield chunks as they arrive (for real-time UX)

    Returns:
        RLPALoopResult if stream=False
        AsyncIterable[str] if stream=True
    """
    loop = RLAPHLoop(max_depth=max_depth, verbose=verbose)
    ctx = context or SessionContext()
    wd = Path(working_dir) if working_dir else Path.cwd()
    return await loop.run(query, ctx, wd, session_id)
```

### Task 4: `RLMSubAgent` Class

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

### Task 5: Pre-defined Sub-Agents

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

### Task 6: `/causal` Router Skill

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

### Task 7: `/causal status` Handler (Dashboard Format)

```python
async def cmd_status(topic: str | None, args: dict) -> str:
    """Show causal awareness dashboard with valid/invalidated frames."""
    from src.config import CFConfig

    config = CFConfig.load()
    limit = config.status_limit
    use_icons = config.status_icons

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

    # Icons
    icon_valid = "✓" if use_icons else "[OK]"
    icon_invalid = "✗" if use_icons else "[X]"
    icon_suspended = "⏸" if use_icons else "[...]"

    # Build dashboard
    output = "## CausalFrame Status\n\n"
    output += f"**Session:** `{session_id}` (most recent; use --session ID for others)\n\n"

    # Summary section
    output += "### Summary\n"
    output += f"- Valid frames: {len(completed)} ({icon_valid})\n"
    if invalidated:
        output += f"- Invalidated frames: {len(invalidated)} ({icon_invalid}) — re-run with `/causal resume`\n"
    else:
        output += f"- Invalidated frames: 0 ({icon_invalid}) — all knowledge is current\n"
    if suspended:
        output += f"- Suspended branches: {len(suspended)} ({icon_suspended}) — ready to resume\n"
    else:
        output += f"- Suspended branches: 0 ({icon_suspended}) — none ready to resume\n"
    output += "\n"

    # Valid frames table
    if completed:
        output += "### Valid Frames\n"
        output += "| Status | Frame | Query | Confidence |\n"
        output += "|--------|-------|-------|------------|\n"

        for f in completed[:limit]:
            query_short = f.query[:40] + "..." if len(f.query) > 40 else f.query
            output += f"| {icon_valid} | `{f.frame_id[:8]}` | {query_short} | {f.confidence:.1f} |\n"

        if len(completed) > limit:
            output += f"\n... and {len(completed) - limit} more. Use `--full` to see all.\n"
        output += "\n"

    # Invalidated frames table
    if invalidated:
        output += "### Invalidated Frames\n"
        output += "| Status | Frame | Reason | Confidence |\n"
        output += "|--------|-------|--------|------------|\n"

        for f in invalidated[:limit]:
            query_short = f.query[:30] + "..." if len(f.query) > 30 else f.query
            reason = f.invalidation_condition.get("description", "Unknown") if f.invalidation_condition else "Unknown"
            reason_short = reason[:30] + "..." if len(reason) > 30 else reason
            output += f"| {icon_invalid} | `{f.frame_id[:8]}` | {reason_short} | {f.confidence:.1f} |\n"

        if len(invalidated) > limit:
            output += f"\n... and {len(invalidated) - limit} more.\n"
        output += "\n"

    # Suspended frames table
    if suspended:
        output += "### Suspended Branches\n"
        output += "| Status | Frame | Query | Confidence |\n"
        output += "|--------|-------|-------|------------|\n"

        for f in suspended[:limit]:
            query_short = f.query[:40] + "..." if len(f.query) > 40 else f.query
            output += f"| {icon_suspended} | `{f.frame_id[:8]}` | {query_short} | {f.confidence:.1f} |\n"
        output += "\n"

    # Suggestions section
    output += "**Suggestions:**\n"
    if invalidated:
        first_invalidated = invalidated[0].frame_id[:8]
        output += f"- Invalidated? Try `/causal resume {first_invalidated}`\n"
    output += "- More details: `/causal status --full` or `/causal tree`\n"

    return output
```

**Example output:**
```
## CausalFrame Status

**Session:** `recursion_test` (most recent; use --session ID for others)

### Summary
- Valid frames: 4 (✓)
- Invalidated frames: 1 (✗) — re-run with `/causal resume`
- Suspended branches: 0 (⏸) — none ready to resume

### Valid Frames
| Status | Frame | Query | Confidence |
|--------|-------|-------|------------|
| ✓ | `99c74b...` | Analyze both src/frame... | 0.8 |
| ✓ | `9a0a68...` | Read the frame_index.py file... | 0.8 |

### Invalidated Frames
| Status | Frame | Reason | Confidence |
|--------|-------|--------|------------|
| ✗ | `f4c81a...` | File src/auth.py changed | 0.8 |

**Suggestions:**
- Invalidated? Try `/causal resume f4c81a...`
- More details: `/causal status --full` or `/causal tree`
```

### Task 8: `/causal tree` Handler

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

### Task 9: `/causal resume` Handler

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

### Task 10: `/causal clear-cache` Handler

```python
def cmd_clear_cache() -> str:
    """Force fresh ContextMap for testing."""
    from src.frame.context_map import ContextMap

    # Clear any cached ContextMap
    ContextMap.clear_cache()

    return "✓ ContextMap cache cleared. Next run will re-scan files."
```

### Task 11: SessionStart Enhancement

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

### Task 12: Verbose Mode Expansion

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
| `src/repl/llm_client.py` | Refactor to SDK + streaming | +40 |
| `src/config.py` | Add CFConfig class for user config | +35 |
| `src/repl/rlaph_loop.py` | Add `run_rlaph()` + stream support + verbose | +30 |
| `src/agents/sub_agent.py` | NEW: RLMSubAgent class | +50 |
| `src/agents/presets.py` | NEW: Pre-configured agents | +40 |
| `src/agents/__init__.py` | NEW: Package init | +5 |
| `skills/causal.md` | Router skill | +10 |
| `src/skills/causal_router.py` | Command handlers (enhanced status) | +170 |
| `hooks/session_start.py` | Enhancement | +20 |
| `src/frame/frame_store.py` | Add `find_most_recent_session()` | +10 |

**Total:** ~410 lines of new/changed code

**Config file:** `~/.claude/causalframe-config.json` (created on first use with defaults)

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
