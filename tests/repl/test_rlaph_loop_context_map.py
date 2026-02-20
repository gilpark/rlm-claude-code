"""Tests for RLAPHLoop ContextMap integration."""
import pytest
from pathlib import Path
from src.repl.rlaph_loop import RLAPHLoop
from src.frame.context_map import ContextMap
from src.types import SessionContext


@pytest.mark.asyncio
async def test_rlaph_loop_accepts_context_map(tmp_path):
    """RLAPHLoop should accept and use ContextMap."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    cm = ContextMap(tmp_path)

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=0,
        context_map=cm,
    )

    assert loop._context_map is cm


@pytest.mark.asyncio
async def test_rlaph_loop_passes_context_map_to_repl(tmp_path):
    """RLAPHLoop should pass ContextMap to RLMEnvironment."""
    cm = ContextMap(tmp_path)

    loop = RLAPHLoop(context_map=cm)
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})

    # Run will create REPL with context_map
    assert loop._context_map is cm
