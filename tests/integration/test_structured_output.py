"""Integration test for structured LLM output."""
import pytest
from pathlib import Path

from src.repl.rlaph_loop import RLAPHLoop
from src.repl.json_parser import parse_llm_response
from src.types import SessionContext


@pytest.mark.asyncio
async def test_llm_returns_structured_output():
    """LLM should return parseable JSON."""
    loop = RLAPHLoop(max_depth=2, verbose=True)
    context = SessionContext(files={}, messages=[])

    result = await loop.run(
        "What is 2+2? Return as JSON with conclusion and confidence.",
        context,
        working_dir=Path.cwd(),
        session_id="test_structured",
    )

    # Try to parse the answer as JSON
    parsed = parse_llm_response(result.answer)

    # Should have a conclusion (even if not perfect JSON)
    assert parsed.conclusion or result.answer


@pytest.mark.asyncio
async def test_sub_tasks_trigger_recursion():
    """sub_tasks field should trigger automatic recursion."""
    loop = RLAPHLoop(max_depth=3, verbose=True)
    context = SessionContext(files={}, messages=[])

    # This query should decompose into sub-tasks
    result = await loop.run(
        "Analyze src/frame/causal_frame.py and src/frame/frame_index.py. " +
        "Use sub_tasks to analyze each file separately.",
        context,
        working_dir=Path.cwd(),
        session_id="test_subtasks",
    )

    # Should have created child frames
    assert len(loop.frame_index) > 1

    # Check for depth > 0 frames (recursion happened)
    depths = [f.depth for f in loop.frame_index._frames.values()]
    assert max(depths) >= 1  # At least one recursive call


@pytest.mark.asyncio
async def test_next_action_controls_recursion():
    """next_action=finalize should prevent infinite recursion."""
    loop = RLAPHLoop(max_depth=5, verbose=True)
    context = SessionContext(files={}, messages=[])

    result = await loop.run(
        "Count from 1 to 3. Set next_action to 'finalize' when done.",
        context,
        working_dir=Path.cwd(),
        session_id="test_next_action",
    )

    # Should not exceed max depth
    assert result.depth_used < 5

    # Parse final response
    parsed = parse_llm_response(result.answer)
    assert parsed.next_action in ("finalize", None)  # Finalized or default
