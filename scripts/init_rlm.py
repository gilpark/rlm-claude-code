#!/usr/bin/env python3
"""
Initialize RLM environment on session start.

Called by: hooks/hooks.json SessionStart
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def init_rlm():
    """Initialize RLM configuration and state."""
    config_dir = Path.home() / ".claude"
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "rlm-config.json"
    
    # Create default config if not exists
    if not config_file.exists():
        default_config = {
            "activation": {
                "mode": "complexity",
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
        
        print(f"Created RLM config at {config_file}", file=sys.stderr)
    
    # Create trajectories directory
    trajectories_dir = config_dir / "rlm-trajectories"
    trajectories_dir.mkdir(exist_ok=True)

    # Initialize StatePersistence session
    try:
        from src.state_persistence import get_persistence

        persistence = get_persistence()
        session_id = os.environ.get("CLAUDE_SESSION_ID", "default")
        persistence.init_session(session_id)
        print(f"RLM session initialized: {session_id}", file=sys.stderr)
    except ImportError as e:
        print(f"StatePersistence not available: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to initialize session: {e}", file=sys.stderr)

    print("RLM initialized", file=sys.stderr)


if __name__ == "__main__":
    init_rlm()
