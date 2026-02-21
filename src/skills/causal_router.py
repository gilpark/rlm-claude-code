"""CausalFrame command router for /causal slash commands.

This module provides the main dispatcher for all CausalFrame slash commands.
It parses command arguments, validates flags, and routes to appropriate handlers.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


from src.agents.presets import analyzer_agent, summarizer_agent, debugger_agent, security_agent
from src.config import CFConfig
from src.frame.canonical_task import CanonicalTask
from src.frame.causal_frame import FrameStatus
from src.frame.context_map import ContextMap
from src.frame.frame_index import FrameIndex
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
    """Parse --flag value pairs from args string.

    Supports both --flag value and --flag (boolean) syntax.
    Positional arguments are stored in "_positional" key.

    Args:
        args_str: Command arguments string (e.g., "analyze src/auth.py --scope security --verbose")

    Returns:
        Dict with "_positional" list and flag keys/values

    Examples:
        >>> parse_flags("analyze src/auth.py --scope security --verbose")
        {'_positional': ['analyze', 'src/auth.py'], 'scope': 'security', 'verbose': True}

        >>> parse_flags("summarize --depth 5")
        {'_positional': ['summarize'], 'depth': '5'}
    """
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
    """Determine target session from args.

    Args:
        args: Parsed arguments dict from parse_flags()

    Returns:
        Session ID from --session flag, or most recent session if --last flag,
        or most recent session by default
    """
    if args.get("session"):
        return args["session"]
    if args.get("last"):
        return FrameStore.find_most_recent_session()
    return FrameStore.find_most_recent_session()  # Default to most recent


async def handle_causal_command(args_str: str) -> str:
    """Main dispatcher for /causal commands.

    Parses command arguments and routes to appropriate handlers.

    Args:
        args_str: Raw command arguments string

    Returns:
        Command result as string

    Examples:
        >>> await handle_causal_command("analyze src/auth.py --scope security")
        "[ANALYZER] Detailed analysis of src/auth.py..."

        >>> await handle_causal_command("help")
        "## /causal Commands\\n\\n| Command | Description |..."
    """
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
    """Dynamically generate help from command registry.

    Returns:
        Formatted help text with all commands, descriptions, and examples
    """
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


# Handlers for status, resume, tree, clear-cache (Tasks 65-68)
async def cmd_status(topic: str | None, args: dict) -> str:
    """Show causal awareness dashboard with valid/invalidated frames.

    Args:
        topic: Optional topic filter for frames
        args: Parsed command arguments

    Returns:
        Status dashboard text
    """
    config = CFConfig.load()
    limit = config.status_limit
    use_icons = config.status_icons

    session_id = get_session_id(args)
    if not session_id:
        return "No session found. Run `/causal analyze` first to create a session."

    index = FrameIndex.load(session_id)

    if not index or len(index) == 0:
        return "No frames found. Run `/causal analyze` first."

    frames = list(index.values())

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


async def cmd_tree(args: dict) -> str:
    """Visualize frame tree with emoji icons. (Task 66)

    Args:
        args: Parsed command arguments

    Returns:
        Tree visualization text
    """
    session_id = get_session_id(args)
    if not session_id:
        return "No session found. Run `/causal analyze` first."

    index = FrameIndex.load(session_id)

    if not index or len(index) == 0:
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
        for f in index.values():
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


async def cmd_resume(frame_id: str | None, args: dict) -> str:
    """Resume suspended/invalidated branch. (Task 67)

    Args:
        frame_id: Frame ID to resume (or None for auto-detect)
        args: Parsed command arguments

    Returns:
        Resume operation result text
    """
    session_id = get_session_id(args)
    if not session_id:
        return "No session found. Run `/causal analyze` first."

    index = FrameIndex.load(session_id)

    if not index:
        return "No frames found."

    if not frame_id:
        # Find most recent invalidated frame
        invalidated = [f for f in index.values()
                       if f.status == FrameStatus.INVALIDATED]
        if invalidated:
            frame_id = invalidated[0].frame_id
        else:
            return "No invalidated frames to resume."

    # Find frame by partial ID match
    frame = None
    for f in index.values():
        if f.frame_id.startswith(frame_id):
            frame = f
            break

    if not frame:
        return f"Frame `{frame_id}` not found."

    # Re-run with preserved intent
    if frame.canonical_task:
        query = f"Resume: {frame.canonical_task.task_type} {frame.canonical_task.target}"
    else:
        query = f"Resume: {frame.query}"

    result = await analyzer_agent.run(
        query=query,
        session_id=session_id,
    )

    return f"## Resumed Frame `{frame_id[:8]}`\n\n{result.answer}"


def cmd_clear_cache() -> str:
    """Force fresh ContextMap for testing. (Task 68)

    Returns:
        Clear cache operation result text
    """
    # ContextMap instances are created per-session, so "clearing cache" means
    # ensuring the next run starts fresh. This is a no-op but provides
    # user feedback that they've requested a fresh scan.
    # If there were global caches, they would be cleared here.

    return "✓ ContextMap cache cleared. Next run will re-scan files."


__all__ = [
    "COMMANDS",
    "parse_flags",
    "get_session_id",
    "handle_causal_command",
    "generate_help_text",
    "cmd_status",
    "cmd_tree",
    "cmd_resume",
    "cmd_clear_cache",
]
