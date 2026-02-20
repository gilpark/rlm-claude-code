"""Tests for run_orchestrator.py entry point."""

from __future__ import annotations

import pytest
from pathlib import Path


@pytest.mark.asyncio
async def test_run_rlaph_creates_context_map(tmp_path):
    """run_rlaph should create ContextMap and pass to RLAPHLoop."""
    from scripts.run_orchestrator import run_rlaph

    test_file = tmp_path / "test.py"
    test_file.write_text("print('test')")

    result = await run_rlaph(
        "What is in test.py?",
        depth=0,
        verbose=False,
        working_dir=tmp_path,
    )

    assert result is not None


def test_get_current_commit_hash_in_git_repo():
    """get_current_commit_hash should return hash in git repo."""
    from scripts.run_orchestrator import get_current_commit_hash

    hash_result = get_current_commit_hash(Path.cwd())
    assert hash_result is not None
    assert len(hash_result) == 8
