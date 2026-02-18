#!/usr/bin/env python3
"""
Initialize RLM environment on session start.

Called by: hooks/hooks.json SessionStart

Hook input (via stdin JSON):
{
    "session_id": "...",
    "transcript_path": "...",
    "hook_event_name": "SessionStart",
    "cwd": "...",
    ...
}

This script:
1. Captures session_id and transcript_path from hook input
2. Saves to ~/.claude/rlm-state/current-session.json for later use
3. Exports CLAUDE_SESSION_ID and CLAUDE_TRANSCRIPT_PATH via CLAUDE_ENV_FILE
4. Initializes the RLM state persistence layer
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def init_rlm():
    """Initialize RLM configuration and state from SessionStart hook."""
    config_dir = Path.home() / ".claude"
    config_dir.mkdir(exist_ok=True)
    state_dir = config_dir / "rlm-state"
    state_dir.mkdir(exist_ok=True)

    # Read hook input from stdin
    hook_data = {}
    try:
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            hook_data = json.loads(stdin_data)
    except json.JSONDecodeError:
        pass

    # Extract session info from hook data
    session_id = hook_data.get("session_id", os.environ.get("CLAUDE_SESSION_ID", "default"))
    transcript_path = hook_data.get("transcript_path", os.environ.get("CLAUDE_TRANSCRIPT_PATH", ""))
    cwd = hook_data.get("cwd", os.environ.get("CLAUDE_CWD", os.getcwd()))
    hook_event = hook_data.get("hook_event_name", "unknown")

    # Save session info to file for later use by other scripts
    session_info = {
        "session_id": session_id,
        "transcript_path": transcript_path,
        "cwd": cwd,
        "hook_event": hook_event,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }

    session_file = state_dir / "current-session.json"
    with open(session_file, "w") as f:
        json.dump(session_info, f, indent=2)

    # Export to CLAUDE_ENV_FILE if available (makes vars available to subsequent commands)
    env_file = os.environ.get("CLAUDE_ENV_FILE")
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"export CLAUDE_SESSION_ID='{session_id}'\n")
            f.write(f"export CLAUDE_TRANSCRIPT_PATH='{transcript_path}'\n")

    # Create symlink to transcript for easy access
    if transcript_path and Path(transcript_path).exists():
        symlink_path = state_dir / "current-session.jsonl"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(transcript_path)

    # Create default RLM config if not exists
    config_file = config_dir / "rlm-config.json"
    if not config_file.exists():
        default_config = {
            "activation": {
                "mode": "micro",
                "fallback_token_threshold": 80000,
                "complexity_score_threshold": 2
            },
            "depth": {
                "default": 2,
                "max": 3,
                "spawn_repl_at_depth_1": True
            },
            "hybrid": {
                "enabled": True,
                "simple_query_bypass": True,
                "simple_confidence_threshold": 0.95
            },
            "trajectory": {
                "verbosity": "normal",
                "streaming": True,
                "colors": True,
                "export_enabled": True,
                "export_path": "~/.claude/rlm-trajectories/"
            },
            "models": {
                "root_model": "opus",
                "recursive_depth_1": "sonnet",
                "recursive_depth_2": "haiku",
                "openai_root": "gpt-5.2-codex",
                "openai_recursive": "gpt-4o-mini"
            },
            "cost_controls": {
                "max_recursive_calls_per_turn": 10,
                "max_tokens_per_recursive_call": 8000,
                "abort_on_cost_threshold": 50000
            }
        }

        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=2)

    # Create trajectories directory
    trajectories_dir = config_dir / "rlm-trajectories"
    trajectories_dir.mkdir(exist_ok=True)

    # Initialize StatePersistence session
    try:
        from src.state_persistence import get_persistence

        persistence = get_persistence()
        persistence.init_session(session_id)

        # Store session info in working memory
        persistence.update_working_memory("session_info", session_info)
        persistence.save_state()

    except ImportError as e:
        print(f"StatePersistence not available: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to initialize session: {e}", file=sys.stderr)

    # Output success (stdout is added to context for SessionStart hooks)
    result = {
        "status": "initialized",
        "session_id": session_id,
        "transcript_path": transcript_path,
        "config_file": str(config_file),
        "message": f"RLM session initialized for {session_id}",
    }
    print(json.dumps(result))


if __name__ == "__main__":
    init_rlm()
