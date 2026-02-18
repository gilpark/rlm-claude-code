#!/usr/bin/env python3
"""
RLM Orchestrator Entry Point.

This script runs the RLM orchestrator with a composed prompt.
The prompt should include task + file paths (composed by /rlm-orchestrator command).

Usage:
    # Basic usage - pass composed prompt
    uv run python scripts/run_orchestrator.py "Task: explain auth flow

    Files to read:
    - /path/to/src/auth.py
    - /path/to/src/login.py"

    # With verbose output
    uv run python scripts/run_orchestrator.py --verbose "<prompt>"

    # Utility commands
    uv run python scripts/run_orchestrator.py --validate  # Check dependencies
    uv run python scripts/run_orchestrator.py --status    # Show RLM status

Flow:
    /rlm-orchestrator <task>
            ↓
    Main Claude (Glob/Grep to find paths)
            ↓
    uv run python scripts/run_orchestrator.py "<task + paths>"
            ↓
    RLM orchestrator runs (REPL can read files as needed)
            ↓
    Returns answer

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
import warnings
from pathlib import Path
from typing import Any

# Suppress third-party deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cpmpy")


def get_state_dir() -> Path:
    """Get RLM state directory."""
    return Path.home() / ".claude" / "rlm-state"


def load_context_from_disk() -> dict[str, Any]:
    """Load context from state file (written by hooks). Fallback only."""
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


def build_context(
    files: dict[str, str] | None = None,
    working_memory: dict[str, Any] | None = None,
    use_disk_fallback: bool = False,
) -> dict[str, Any]:
    """Build context from files or disk fallback."""
    context = {
        "conversation": [],
        "files": files or {},
        "tool_outputs": [],
        "working_memory": working_memory or {},
    }

    # If no files provided and fallback enabled, try disk
    if not context["files"] and use_disk_fallback:
        disk_context = load_context_from_disk()
        if disk_context.get("files"):
            context["files"] = disk_context["files"]
            context["working_memory"] = disk_context.get("working_memory", {})

    return context


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
    context = load_context_from_disk()

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
    }
    return result


def do_bypass() -> dict[str, Any]:
    """Set RLM bypass flag (for /simple command)."""
    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)

    # Set bypass flag
    bypass_file = state_dir / "bypass_rlm.json"
    bypass_file.write_text(json.dumps({"bypass": True, "reason": "simple_mode"}))

    return {"status": "bypass_set"}


async def run_orchestrator(
    query: str,
    depth: int = 2,
    verbose: bool = False,
    stream: bool = False,
) -> str:
    """
    Run the RLM orchestrator on a query.

    Args:
        query: User query (includes task + file paths)
        depth: Maximum recursion depth (default 2)
        verbose: Print trajectory events
        stream: Use streaming mode for real-time tokens

    Returns:
        Final answer from orchestrator
    """
    from src import RLMOrchestrator
    from src.config import ActivationConfig, DepthConfig, RLMConfig
    from src.trajectory import TrajectoryEventType

    # Build empty context - files are read by REPL as needed
    context_data = build_context(files={}, use_disk_fallback=False)
    context = build_session_context(context_data)

    if verbose:
        print(f"[RLM] Starting orchestrator with query ({len(query)} chars)")
        if stream:
            print("[RLM] Streaming mode enabled")

    # Configure depth and force always mode
    config = RLMConfig(
        depth=DepthConfig(max=depth),
        activation=ActivationConfig(mode="always"),
    )

    # Create orchestrator
    orchestrator = RLMOrchestrator(config=config)

    # Run and collect result
    final_answer = None
    async for event in orchestrator.run(query, context):
        if verbose:
            if event.type == TrajectoryEventType.RLM_START:
                print(f"[RLM] {event.content}")
            elif event.type == TrajectoryEventType.REASON:
                tokens = event.metadata.get('input_tokens', 0)
                print(f"[REASON] depth={event.depth} tokens={tokens}")
            elif event.type == TrajectoryEventType.STREAM:
                # Print streaming tokens in real-time
                print(event.content, end="", flush=True)
            elif event.type == TrajectoryEventType.REPL_EXEC:
                print(f"[REPL] {event.content[:100]}...")
            elif event.type == TrajectoryEventType.RECURSE_START:
                print(f"[RECURSE] {event.content}")
            elif event.type == TrajectoryEventType.ERROR:
                print(f"[ERROR] {event.content}")

        if event.type == TrajectoryEventType.FINAL:
            final_answer = event.content

    # Add newline after streaming
    if stream and verbose:
        print()

    return final_answer or "No answer produced"


def main():
    parser = argparse.ArgumentParser(
        description="Run RLM orchestrator (uses Claude CLI if no API keys)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pass composed prompt (from /rlm-orchestrator command)
    uv run python scripts/run_orchestrator.py "Task: explain auth

    Files:
    - /path/to/src/auth.py"

    # With verbose output
    uv run python scripts/run_orchestrator.py --verbose "<prompt>"

    # Utility commands
    uv run python scripts/run_orchestrator.py --validate
    uv run python scripts/run_orchestrator.py --status
        """,
    )
    parser.add_argument("query", nargs="?", help="Composed prompt (task + file paths)")
    parser.add_argument("--query", "-q", dest="query_flag", help="Query to process (alternative)")
    parser.add_argument("--depth", "-d", type=int, default=2, help="Max recursion depth (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print trajectory events")
    parser.add_argument("--stream", "-s", action="store_true", help="Stream tokens in real-time")

    # Utility commands
    parser.add_argument("--validate", action="store_true", help="Validate dependencies")
    parser.add_argument("--status", action="store_true", help="Show RLM status")
    parser.add_argument("--bypass", action="store_true", help="Set bypass flag")

    args = parser.parse_args()

    # Handle utility commands
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

    # Get query
    query = args.query or args.query_flag
    if not query:
        parser.error("Query required. Provide as argument or use --query")

    # Run orchestrator
    try:
        result = asyncio.run(
            run_orchestrator(query, depth=args.depth, verbose=args.verbose, stream=args.stream)
        )
        print(result)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
