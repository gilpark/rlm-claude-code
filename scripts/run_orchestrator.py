#!/usr/bin/env python3
"""
RLM Orchestrator Entry Point.

This script runs the RLM orchestrator directly, using Claude CLI for LLM calls
when no API keys are available. Context is loaded from the hook-generated JSON file.

Usage:
    uv run python scripts/run_orchestrator.py "analyze this code"
    uv run python scripts/run_orchestrator.py --query "explain the architecture" --depth 2
    uv run python scripts/run_orchestrator.py --init          # Initialize context (for SubagentStart hook)
    uv run python scripts/run_orchestrator.py --validate      # Validate dependencies (for PreToolUse hook)
    uv run python scripts/run_orchestrator.py --status        # Show RLM status (for /rlm command)
    uv run python scripts/run_orchestrator.py --bypass        # Set bypass flag (for /simple command)
    uv run python scripts/run_orchestrator.py --help

Environment:
    Context is loaded from ~/.claude/rlm-state/context.json (written by sync_context hook)

    LLM Provider Selection (automatic):
    - If ANTHROPIC_API_KEY is set → Uses Anthropic API
    - If OPENAI_API_KEY is set → Uses OpenAI API
    - Otherwise → Uses Claude CLI (subscription auth, no API key needed)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def get_state_dir() -> Path:
    """Get RLM state directory."""
    return Path.home() / ".claude" / "rlm-state"


def load_context() -> dict[str, Any]:
    """Load context from state file (written by hooks)."""
    ctx_file = get_state_dir() / "context.json"
    if ctx_file.exists():
        try:
            return json.loads(ctx_file.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "conversation": [],
        "files": {},
        "tool_outputs": [],
        "working_memory": {},
    }


def save_context(data: dict[str, Any]) -> None:
    """Save context to state file."""
    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    ctx_file = state_dir / "context.json"
    ctx_file.write_text(json.dumps(data, indent=2, default=str))


def build_session_context(data: dict[str, Any]):
    """Build SessionContext from raw JSON data."""
    from src.types import Message, MessageRole, SessionContext, ToolOutput

    # Convert messages
    messages = []
    for m in data.get("conversation", []):
        if isinstance(m, Message):
            messages.append(m)
        elif isinstance(m, dict):
            try:
                role = MessageRole(m.get("role", "user"))
            except ValueError:
                role = MessageRole.USER
            messages.append(Message(role=role, content=m.get("content", "")))

    # Convert tool outputs
    tool_outputs = []
    for o in data.get("tool_outputs", []):
        if isinstance(o, ToolOutput):
            tool_outputs.append(o)
        elif isinstance(o, dict):
            tool_outputs.append(
                ToolOutput(
                    tool_name=o.get("tool", o.get("tool_name", "unknown")),
                    content=o.get("content", ""),
                    exit_code=o.get("exit_code"),
                    timestamp=o.get("timestamp"),
                )
            )

    return SessionContext(
        messages=messages,
        files=data.get("files", {}),
        tool_outputs=tool_outputs,
        working_memory=data.get("working_memory", {}),
    )


def do_init() -> dict[str, Any]:
    """Initialize RLM context (for SubagentStart hook)."""
    context = load_context()

    # Return hook-compatible response
    result = {
        "status": "initialized",
        "hookSpecificOutput": {
            "hookEventName": "SubagentStart",
            "additionalContext": f"[RLM context loaded: {len(context.get('files', {}))} files, {len(context.get('tool_outputs', []))} outputs]"
        }
    }
    return result


def do_validate() -> dict[str, Any]:
    """Validate orchestrator dependencies (for PreToolUse hook)."""
    errors = []

    # Check venv exists
    plugin_root = Path(__file__).parent.parent
    venv_python = plugin_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        errors.append("venv not found - run 'uv sync'")

    # Check src modules
    try:
        sys.path.insert(0, str(plugin_root))
        from src.orchestrator import RLMOrchestrator  # noqa: F401
    except ImportError as e:
        errors.append(f"Import error: {e}")

    # Check API client
    try:
        from src.api_client import MultiProviderClient  # noqa: F401
    except ImportError as e:
        errors.append(f"API client error: {e}")

    if errors:
        return {"status": "error", "reasons": errors}
    return {"status": "ok"}


def do_status() -> dict[str, Any]:
    """Show RLM status (for /rlm command)."""
    from src.config import RLMConfig

    config = RLMConfig.load()
    context = load_context()

    result = {
        "status": "active",
        "config": {
            "activation_mode": config.activation.mode,
            "max_depth": config.depth.max,
            "root_model": config.models.root_model,
        },
        "context": {
            "files": len(context.get("files", {})),
            "messages": len(context.get("conversation", [])),
            "tool_outputs": len(context.get("tool_outputs", [])),
        },
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": f"[RLM status: mode={config.activation.mode}, depth={config.depth.max}, files={len(context.get('files', {}))}]"
        }
    }
    return result


def do_bypass() -> dict[str, Any]:
    """Set RLM bypass flag (for /simple command)."""
    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)

    # Set bypass flag
    bypass_file = state_dir / "bypass_rlm.json"
    bypass_file.write_text(json.dumps({"bypass": True, "reason": "simple_mode"}))

    result = {
        "status": "bypass_set",
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": "[RLM bypass enabled for this operation]"
        }
    }
    return result


async def run_orchestrator(query: str, depth: int = 2, verbose: bool = False) -> str:
    """
    Run the RLM orchestrator on a query.

    Args:
        query: User query to process
        depth: Maximum recursion depth (default 2)
        verbose: Print trajectory events

    Returns:
        Final answer from orchestrator
    """
    from src import RLMOrchestrator
    from src.config import DepthConfig, RLMConfig
    from src.trajectory import TrajectoryEventType

    # Load context from hooks
    context_data = load_context()
    context = build_session_context(context_data)

    # Configure depth
    config = RLMConfig(
        depth=DepthConfig(max=depth),
    )

    # Create orchestrator (auto-uses CLI if no API keys)
    orchestrator = RLMOrchestrator(config=config)

    # Run and collect result
    final_answer = None
    async for event in orchestrator.run(query, context):
        if verbose:
            if event.type == TrajectoryEventType.RLM_START:
                print(f"[RLM] {event.content}")
            elif event.type == TrajectoryEventType.REASON:
                print(f"[REASON] depth={event.depth} tokens={event.metadata.get('input_tokens', 0)}")
            elif event.type == TrajectoryEventType.REPL_EXEC:
                print(f"[REPL] {event.content[:100]}...")
            elif event.type == TrajectoryEventType.RECURSE_START:
                print(f"[RECURSE] {event.content}")
            elif event.type == TrajectoryEventType.ERROR:
                print(f"[ERROR] {event.content}")

        if event.type == TrajectoryEventType.FINAL:
            final_answer = event.content

    return final_answer or "No answer produced"


def main():
    parser = argparse.ArgumentParser(
        description="Run RLM orchestrator (uses Claude CLI if no API keys)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    uv run python scripts/run_orchestrator.py "analyze the auth module"

    # With custom depth
    uv run python scripts/run_orchestrator.py --depth 3 "complex analysis task"

    # Verbose output
    uv run python scripts/run_orchestrator.py --verbose "debug this code"

    # Hook operations
    uv run python scripts/run_orchestrator.py --init      # For SubagentStart hook
    uv run python scripts/run_orchestrator.py --validate  # For PreToolUse hook
    uv run python scripts/run_orchestrator.py --status    # For /rlm command
    uv run python scripts/run_orchestrator.py --bypass    # For /simple command
        """,
    )
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--query", "-q", dest="query_flag", help="Query to process (alternative)")
    parser.add_argument("--depth", "-d", type=int, default=2, help="Max recursion depth (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print trajectory events")
    parser.add_argument("--context", action="store_true", help="Print loaded context and exit")

    # Hook flags
    parser.add_argument("--init", action="store_true", help="Initialize context (for SubagentStart hook)")
    parser.add_argument("--validate", action="store_true", help="Validate dependencies (for PreToolUse hook)")
    parser.add_argument("--status", action="store_true", help="Show RLM status (for /rlm command)")
    parser.add_argument("--bypass", action="store_true", help="Set bypass flag (for /simple command)")

    args = parser.parse_args()

    # Handle hook flags
    if args.init:
        result = do_init()
        print(json.dumps(result))
        return

    if args.validate:
        result = do_validate()
        print(json.dumps(result))
        if result.get("status") != "ok":
            sys.exit(1)
        return

    if args.status:
        result = do_status()
        print(json.dumps(result, indent=2))
        return

    if args.bypass:
        result = do_bypass()
        print(json.dumps(result))
        return

    # Handle --context flag
    if args.context:
        context = load_context()
        print(json.dumps(context, indent=2, default=str))
        return

    # Get query
    query = args.query or args.query_flag
    if not query:
        parser.error("Query required. Provide as argument or use --query")

    # Run orchestrator
    try:
        result = asyncio.run(run_orchestrator(query, depth=args.depth, verbose=args.verbose))
        print(result)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
