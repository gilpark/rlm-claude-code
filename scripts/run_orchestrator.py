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
import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

# Suppress third-party deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cpmpy")

# Import for ContextMap (re-exported for test access)
from src.frame.context_map import ContextMap, get_current_commit_hash as _get_current_commit_hash


def get_current_commit_hash(root_dir: Path) -> str | None:
    """
    Get current git HEAD commit hash.

    Re-exported from context_map for orchestrator use.

    Args:
        root_dir: Repository root directory

    Returns:
        8-char commit hash or None if not in git repo
    """
    return _get_current_commit_hash(root_dir)


def get_or_create_session_id(explicit_id: str | None = None) -> str:
    """
    Get session ID from various sources in priority order:
    1. Explicit argument (--session-id)
    2. Environment variable (CLAUDE_SESSION_ID)
    3. Coordination file (~/.claude/rlm-frames/.current_session)
    4. Generate new UUID[:8]
    """
    # 1. Explicit argument
    if explicit_id:
        return explicit_id

    # 2. Environment variable
    env_id = os.environ.get("CLAUDE_SESSION_ID")
    if env_id:
        return env_id

    # 3. Coordination file
    session_file = Path.home() / ".claude" / "rlm-frames" / ".current_session"
    if session_file.exists():
        try:
            data = json.loads(session_file.read_text())
            if data.get("session_id"):
                return data["session_id"]
        except (json.JSONDecodeError, KeyError):
            pass

    # 4. Generate new
    return str(uuid.uuid4())[:8]


def get_state_dir() -> Path:
    """Get RLM state directory."""
    return Path.home() / ".claude" / "rlm-state"


def load_context_from_disk() -> dict[str, Any]:
    """Load context from transcript file (preferred) or state file (fallback)."""
    # Try transcript parser first (more complete)
    try:
        from src.transcript_parser import load_session_context
        transcript_context = load_session_context()
        if transcript_context and not transcript_context.get("error"):
            return {
                "conversation": transcript_context.get("messages", []),
                "files": transcript_context.get("files", {}),
                "tool_outputs": list(transcript_context.get("tool_outputs", {}).values()),
                "working_memory": transcript_context.get("metadata", {}),
            }
    except ImportError:
        pass

    # Fallback to state file
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
    use_disk_fallback: bool = True,  # Changed default to True
) -> dict[str, Any]:
    """Build context from files or disk fallback (transcript)."""
    context = {
        "conversation": [],
        "files": files or {},
        "tool_outputs": [],
        "working_memory": working_memory or {},
    }

    # If no files provided and fallback enabled, try disk (transcript)
    if not context["files"] and use_disk_fallback:
        disk_context = load_context_from_disk()
        if disk_context.get("files"):
            context["files"] = disk_context["files"]
            context["working_memory"] = disk_context.get("working_memory", {})
        if disk_context.get("conversation"):
            context["conversation"] = disk_context["conversation"]

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

    # Check v2 src modules
    try:
        sys.path.insert(0, str(plugin_root))
        from src.repl.rlaph_loop import RLAPHLoop  # noqa: F401
    except ImportError as e:
        errors.append(f"Import error: {e}")

    # Check LLM client
    try:
        from src.repl.llm_client import LLMClient  # noqa: F401
    except ImportError as e:
        errors.append(f"LLM client error: {e}")

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


async def run_rlaph(
    query: str,
    depth: int = 2,
    verbose: bool = False,
    working_dir: Path | None = None,
    session_id: str | None = None,
) -> str:
    """
    Run the RLAPH loop (clean synchronous llm() mode).

    Key difference from legacy orchestrator:
    - llm() returns actual result immediately (not DeferredOperation)
    - Single clean loop, easier to debug
    - No deferred operation processing

    Args:
        query: User query (includes task + file paths)
        depth: Maximum recursion depth (default 2)
        verbose: Print trajectory events
        working_dir: Working directory for file operations
        session_id: Session ID for frame persistence (default: auto-generate)

    Returns:
        Final answer from RLAPH loop
    """
    from src.repl.rlaph_loop import RLAPHLoop

    print("[RLM:START] Initializing RLAPH loop")

    # Build empty context - files are read by REPL as needed
    context_data = build_context(files={}, use_disk_fallback=False)
    context = build_session_context(context_data)

    work_dir = working_dir or Path.cwd()

    # Create ContextMap for session
    context_map = ContextMap(work_dir)
    commit_hash = context_map.commit_hash

    if verbose:
        print(f"[RLM:CONFIG] Max depth: {depth}")
        print(f"[RLM:CONFIG] Working dir: {work_dir}")
        if commit_hash:
            print(f"[RLM:CONFIG] Git commit: {commit_hash}")

    print(f"[RLM:QUERY] Processing ({len(query)} chars)")
    print(f"[RLM:CONTEXT] {len(context_map.paths)} files in scope")

    # Create RLAPH loop (no renderer in v2 - verbose handled internally)
    # NOTE: context_map parameter will be added in Task 5
    loop = RLAPHLoop(
        max_iterations=20,
        max_depth=depth,
    )

    # Set commit_hash on frame index
    loop.frame_index.commit_hash = commit_hash

    # Set initial query on frame index
    loop.frame_index.initial_query = query
    loop.frame_index.query_summary = query[:50] + ("..." if len(query) > 50 else "")

    # Run loop
    result = await loop.run(query, context, working_dir=work_dir, session_id=session_id)

    print(f"[RLM:DONE] Completed in {result.iterations} iterations")
    print(f"[RLM:DONE] Tokens: {result.tokens_used}, Time: {result.execution_time_ms:.0f}ms")

    return result.answer


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
    parser.add_argument(
        "--session-id",
        dest="session_id",
        help="Session ID for frame persistence (default: auto-detect or generate)",
    )

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

    # Determine working directory (project root)
    plugin_root = Path(__file__).parent.parent

    # Get session ID from coordination file or explicit argument
    session_id = get_or_create_session_id(args.session_id)

    if args.verbose:
        print(f"[RLM:SESSION] Using session_id: {session_id}")

    # Run RLAPH loop (clean synchronous llm() mode)
    try:
        result = asyncio.run(
            run_rlaph(
                query,
                depth=args.depth,
                verbose=args.verbose,
                working_dir=plugin_root,
                session_id=session_id,
            )
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
