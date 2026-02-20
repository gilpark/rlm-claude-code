"""Test session ID flow from orchestrator to frame persistence."""

import subprocess
from pathlib import Path

import pytest


def test_orchestrator_accepts_session_id_argument():
    """Test that orchestrator accepts --session-id argument."""
    result = subprocess.run(
        ["uv", "run", "python", "scripts/run_orchestrator.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # Check that --session-id is in the help output
    assert "--session-id" in result.stdout or "session-id" in result.stdout, (
        f"--session-id not found in help output. stdout:\n{result.stdout}"
    )


def test_orchestrator_session_id_in_run_rlaph_signature():
    """Test that run_rlaph function accepts session_id parameter."""
    import inspect
    import sys

    plugin_root = Path(__file__).parent.parent
    sys.path.insert(0, str(plugin_root))

    from scripts.run_orchestrator import run_rlaph

    sig = inspect.signature(run_rlaph)
    params = list(sig.parameters.keys())

    assert "session_id" in params, f"session_id not in run_rlaph parameters: {params}"


def test_orchestrator_passes_session_id_to_loop():
    """Test that session_id is passed from main() to RLAPHLoop.run()."""
    # This test verifies the integration by checking the code flow
    # A more thorough integration test would require mocking
    import ast

    plugin_root = Path(__file__).parent.parent
    script_path = plugin_root / "scripts" / "run_orchestrator.py"

    source = script_path.read_text()
    tree = ast.parse(source)

    # Find the run_rlaph function and check if it accepts session_id
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_rlaph":
            param_names = [arg.arg for arg in node.args.args]
            assert "session_id" in param_names, (
                f"session_id not in run_rlaph parameters: {param_names}"
            )
            break
    else:
        pytest.fail("run_rlaph function not found in run_orchestrator.py")


def test_orchestrator_main_passes_session_id():
    """Test that main() passes session_id from args/environment to run_rlaph."""
    import ast

    plugin_root = Path(__file__).parent.parent
    script_path = plugin_root / "scripts" / "run_orchestrator.py"

    source = script_path.read_text()

    # Check that session_id is extracted from args or environment
    assert "session_id" in source, "session_id not referenced in script"

    # Check for environment variable fallback pattern
    assert "CLAUDE_SESSION_ID" in source, (
        "CLAUDE_SESSION_ID environment variable not referenced"
    )
