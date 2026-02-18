#!/usr/bin/env python3
"""
RLM Orchestrator Entry Point.

This script runs the RLM orchestrator directly, using Claude CLI for LLM calls
when no API keys are available. Context can be passed via files or loaded from disk.

Usage:
    # Basic usage - pass files directly
    uv run python scripts/run_orchestrator.py "analyze this code" --files src/auth.py src/api.py

    # With directory (recursively finds .py files)
    uv run python scripts/run_orchestrator.py "explain the architecture" --dir src/

    # With custom depth
    uv run python scripts/run_orchestrator.py --depth 3 --dir src/ "complex analysis task"

    # Verbose output
    uv run python scripts/run_orchestrator.py --verbose --files src/main.py "debug this code"

    # Utility commands
    uv run python scripts/run_orchestrator.py --validate  # Check dependencies
    uv run python scripts/run_orchestrator.py --status    # Show RLM status
    uv run python scripts/run_orchestrator.py --help

Context Sources (in order of priority):
    1. --files: Explicit file paths passed as arguments
    2. --dir: Directory to scan for files (respects --ext filter)
    3. Disk: ~/.claude/rlm-state/context.json (fallback from hooks)

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


def load_files_from_paths(file_paths: list[str]) -> dict[str, str]:
    """Load file contents from explicit paths."""
    files = {}
    for path_str in file_paths:
        path = Path(path_str)
        if path.exists() and path.is_file():
            try:
                files[str(path)] = path.read_text()
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}", file=sys.stderr)
    return files


def load_files_from_directory(
    directory: str,
    extensions: list[str] | None = None,
    max_files: int = 50,
    max_size_kb: int = 100,
) -> dict[str, str]:
    """Load files from a directory recursively."""
    if extensions is None:
        extensions = [".py", ".md", ".json", ".yaml", ".yml", ".toml"]

    files = {}
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Warning: Directory not found: {directory}", file=sys.stderr)
        return files

    for ext in extensions:
        for path in dir_path.rglob(f"*{ext}"):
            if len(files) >= max_files:
                print(f"Warning: Max files ({max_files}) reached", file=sys.stderr)
                break

            # Skip hidden dirs and common exclusions
            if any(part.startswith(".") or part in {"node_modules", "__pycache__", ".venv", "venv"} for part in path.parts):
                continue

            # Skip large files
            if path.stat().st_size > max_size_kb * 1024:
                continue

            try:
                files[str(path)] = path.read_text()
            except Exception:
                pass

    return files


def build_context(
    files: dict[str, str] | None = None,
    working_memory: dict[str, Any] | None = None,
    use_disk_fallback: bool = True,
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


def do_init() -> dict[str, Any]:
    """Initialize RLM context - kept for backward compatibility."""
    context = load_context_from_disk()
    result = {
        "status": "initialized",
        "files_loaded": len(context.get("files", {})),
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
    files: dict[str, str] | None = None,
) -> str:
    """
    Run the RLM orchestrator on a query.

    Args:
        query: User query to process
        depth: Maximum recursion depth (default 2)
        verbose: Print trajectory events
        files: Dict of filename -> content for context

    Returns:
        Final answer from orchestrator
    """
    from src import RLMOrchestrator
    from src.config import DepthConfig, RLMConfig
    from src.trajectory import TrajectoryEventType

    # Build context from files (with disk fallback)
    context_data = build_context(files=files, use_disk_fallback=True)
    context = build_session_context(context_data)

    if verbose:
        print(f"[RLM] Loaded {len(context.files)} files into context")

    # Configure depth and force always mode
    from src.config import ActivationConfig
    config = RLMConfig(
        depth=DepthConfig(max=depth),
        activation=ActivationConfig(mode="always"),
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
    # Pass files directly
    uv run python scripts/run_orchestrator.py "analyze auth" --files src/auth.py src/api.py

    # Scan a directory
    uv run python scripts/run_orchestrator.py "explain architecture" --dir src/

    # With custom depth and verbose
    uv run python scripts/run_orchestrator.py --depth 3 --verbose --dir src/ "complex analysis"

    # Utility commands
    uv run python scripts/run_orchestrator.py --validate  # Check dependencies
    uv run python scripts/run_orchestrator.py --status    # Show RLM status
        """,
    )
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--query", "-q", dest="query_flag", help="Query to process (alternative)")
    parser.add_argument("--depth", "-d", type=int, default=2, help="Max recursion depth (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print trajectory events")

    # Context source arguments
    parser.add_argument(
        "--files", "-f", nargs="+", metavar="FILE",
        help="Files to include in context (space-separated)"
    )
    parser.add_argument(
        "--dir", "-D", metavar="DIR",
        help="Directory to scan for files"
    )
    parser.add_argument(
        "--ext", "-e", nargs="+", default=[".py", ".md"],
        help="File extensions to include when scanning directory (default: .py .md)"
    )
    parser.add_argument(
        "--max-files", type=int, default=50,
        help="Max files to load from directory (default: 50)"
    )

    # Utility commands
    parser.add_argument("--validate", action="store_true", help="Validate dependencies")
    parser.add_argument("--status", action="store_true", help="Show RLM status")
    parser.add_argument("--bypass", action="store_true", help="Set bypass flag")
    parser.add_argument("--init", action="store_true", help="Initialize (backward compat)")

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

    if args.init:
        result = do_init()
        print(json.dumps(result))
        return

    # Get query
    query = args.query or args.query_flag
    if not query:
        parser.error("Query required. Provide as argument or use --query")

    # Load files from specified sources
    files = {}

    # 1. Load from explicit --files
    if args.files:
        files.update(load_files_from_paths(args.files))

    # 2. Load from --dir
    if args.dir:
        files.update(load_files_from_directory(
            args.dir,
            extensions=args.ext,
            max_files=args.max_files,
        ))

    # Run orchestrator
    try:
        result = asyncio.run(
            run_orchestrator(query, depth=args.depth, verbose=args.verbose, files=files)
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
