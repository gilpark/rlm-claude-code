"""
Tests for llm_sync recursion with child frame creation.

Implements: Phase 18 Task A1 - Add Synchronous llm(sub_query) Recursion
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.repl.rlaph_loop import RLAPHLoop
from src.frame.causal_frame import FrameStatus
from src.types import RecursionDepthError


@pytest.mark.asyncio
async def test_llm_sync_returns_result():
    """llm_sync returns actual LLM result."""
    # Create loop with mocked LLM client
    mock_client = Mock()
    mock_client.call.return_value = "Test response"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
    )

    # Initialize frame tracking
    loop._current_frame_id = "test_parent_id"

    # Call llm_sync
    result = loop.llm_sync("What is 2+2?")

    # Verify result
    assert result == "Test response"
    # Verify LLM client was called
    mock_client.call.assert_called_once()


@pytest.mark.asyncio
async def test_llm_sync_with_explicit_depth():
    """llm_sync accepts explicit depth parameter."""
    mock_client = Mock()
    mock_client.call.return_value = "Response"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
    )
    loop._current_frame_id = "test_parent_id"

    # Call with explicit depth
    result = loop.llm_sync("Query", depth=2)

    assert result == "Response"
    # Verify depth was passed correctly
    call_kwargs = mock_client.call.call_args.kwargs
    assert call_kwargs["depth"] == 2


@pytest.mark.asyncio
async def test_llm_sync_respects_max_depth():
    """llm_sync raises RecursionDepthError if exceeded."""
    mock_client = Mock()
    mock_client.call.return_value = "Response"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=2,  # Max depth is 2
        llm_client=mock_client,
    )
    loop._current_frame_id = "test_parent_id"

    # Calling at depth 3 should raise
    with pytest.raises(RecursionDepthError) as exc_info:
        loop.llm_sync("Query", depth=3)

    assert exc_info.value.depth == 3
    assert exc_info.value.max_depth == 2


@pytest.mark.asyncio
async def test_llm_sync_depth_default_increments():
    """Without explicit depth, llm_sync uses current+1."""
    mock_client = Mock()
    mock_client.call.return_value = "Response"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
    )
    loop._current_frame_id = "test_parent_id"
    loop._depth = 1  # Set current depth

    # Call without explicit depth - should use 2 (1+1)
    result = loop.llm_sync("Query")

    assert result == "Response"
    call_kwargs = mock_client.call.call_args.kwargs
    assert call_kwargs["depth"] == 2


@pytest.mark.asyncio
async def test_llm_sync_creates_child_frame():
    """llm_sync creates child frame with correct properties."""
    mock_client = Mock()
    mock_client.call.return_value = "LLM result"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
    )
    loop._current_frame_id = "parent_frame_id"

    # Call llm_sync
    result = loop.llm_sync("Test query")

    # Verify child frame was created
    assert len(loop.frame_index) == 1

    # Find child frame by parent
    children = loop.frame_index.find_by_parent("parent_frame_id")
    assert len(children) == 1

    child_frame = children[0]
    assert child_frame.depth == 1  # Default depth is _depth + 1
    assert child_frame.parent_id == "parent_frame_id"
    assert "Test query" in child_frame.query
    assert "LLM result" in child_frame.conclusion
    assert child_frame.status == FrameStatus.COMPLETED


@pytest.mark.asyncio
async def test_llm_sync_verbose_logging():
    """llm_sync logs recursion decisions when verbose=True."""
    mock_client = Mock()
    mock_client.call.return_value = "Response"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
        verbose=True,
    )
    loop._current_frame_id = "test_parent_id"

    # Call and capture print output
    with patch("builtins.print") as mock_print:
        result = loop.llm_sync("Test query with verbose logging")

        # Verify logging occurred
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "[RLM] Recursion at depth" in call_args
        assert "Test query with verbose logging" in call_args


@pytest.mark.asyncio
async def test_llm_sync_does_not_mutate_internal_depth():
    """llm_sync uses explicit depth without mutating self._depth."""
    mock_client = Mock()
    mock_client.call.return_value = "Response"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
    )
    loop._current_frame_id = "test_parent_id"
    initial_depth = loop._depth

    # Call at different depth
    loop.llm_sync("Query", depth=2)

    # Verify internal depth unchanged
    assert loop._depth == initial_depth


@pytest.mark.asyncio
async def test_llm_sync_at_boundary():
    """llm_sync works correctly at max_depth boundary."""
    mock_client = Mock()
    mock_client.call.return_value = "Response"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
    )
    loop._current_frame_id = "test_parent_id"

    # Call exactly at max_depth - should succeed
    result = loop.llm_sync("Query", depth=3)

    assert result == "Response"
    mock_client.call.assert_called_once()
