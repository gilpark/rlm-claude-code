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


# Stub handlers for status, resume, tree, clear-cache (will be implemented in Tasks 65-68)
async def cmd_status(topic: str | None, args: dict) -> str:
    """Show causal awareness dashboard. (Task 65)

    Args:
        topic: Optional topic filter for frames
        args: Parsed command arguments

    Returns:
        Status dashboard text
    """
    return "Status dashboard coming in Task 65..."


async def cmd_tree(args: dict) -> str:
    """Visualize frame tree. (Task 66)

    Args:
        args: Parsed command arguments

    Returns:
        Tree visualization text
    """
    return "Tree visualization coming in Task 66..."


async def cmd_resume(frame_id: str | None, args: dict) -> str:
    """Resume suspended/invalidated branch. (Task 67)

    Args:
        frame_id: Frame ID to resume (or None for auto-detect)
        args: Parsed command arguments

    Returns:
        Resume operation result text
    """
    return "Resume functionality coming in Task 67..."


def cmd_clear_cache() -> str:
    """Force fresh ContextMap. (Task 68)

    Returns:
        Clear cache operation result text
    """
    return "Clear-cache functionality coming in Task 68..."


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
