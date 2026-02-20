"""
Integration tests for REPL llm() function with RLAPHLoop.llm_sync.

Implements: Phase 18 Task A1 - REPL llm() integration with child frame creation
"""

import pytest
from unittest.mock import Mock

from src.repl.rlaph_loop import RLAPHLoop
from src.repl.repl_environment import RLMEnvironment
from src.frame.causal_frame import FrameStatus
from src.types import SessionContext, Message, MessageRole


@pytest.mark.asyncio
async def test_repl_llm_creates_child_frame_via_loop():
    """
    Test that calling llm() from REPL creates child frames via loop.llm_sync.

    This is the core integration test for Phase 18 Task A1.
    """
    # Create loop with mocked LLM client
    mock_client = Mock()
    mock_client.call.return_value = "Sub-query result"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
    )
    loop._current_frame_id = "parent_frame_id"

    # Create REPL with loop reference (as RLAPHLoop.run() does)
    context = SessionContext(
        messages=[Message(role=MessageRole.USER, content="Test")],
        files={},
        tool_outputs=[],
        working_memory={},
    )
    repl = RLMEnvironment(
        context,
        llm_client=mock_client,
        loop=loop,  # Pass loop for llm_sync integration
    )

    # Call llm() from REPL (as Python code would)
    result = repl._recursive_query("What is 2+2?")

    # Verify result
    assert result == "Sub-query result"

    # Verify child frame was created (this is the key integration test)
    assert len(loop.frame_index) == 1
    children = loop.frame_index.find_by_parent("parent_frame_id")
    assert len(children) == 1

    child_frame = children[0]
    assert child_frame.depth == 1
    assert child_frame.parent_id == "parent_frame_id"
    assert "What is 2+2?" in child_frame.query
    assert "Sub-query result" in child_frame.conclusion
    assert child_frame.status == FrameStatus.COMPLETED


@pytest.mark.asyncio
async def test_repl_llm_fallback_without_loop():
    """
    Test that REPL llm() falls back to direct LLMClient.call when no loop.

    This ensures backward compatibility when REPL is used standalone.
    """
    # Create REPL WITHOUT loop reference
    mock_client = Mock()
    mock_client.call.return_value = "Direct LLM result"

    context = SessionContext(
        messages=[Message(role=MessageRole.USER, content="Test")],
        files={},
        tool_outputs=[],
        working_memory={},
    )
    repl = RLMEnvironment(
        context,
        llm_client=mock_client,
        loop=None,  # No loop - should fall back to direct call
    )

    # Call llm() from REPL
    result = repl._recursive_query("What is 2+2?")

    # Verify result
    assert result == "Direct LLM result"

    # Verify direct LLMClient.call was used (no frame tracking)
    mock_client.call.assert_called_once()


@pytest.mark.asyncio
async def test_repl_globals_llm_aliases():
    """
    Test that all llm function aliases in REPL globals work correctly.

    Ensures llm(), recursive_llm(), and recursive_query() all create frames.
    """
    mock_client = Mock()
    mock_client.call.return_value = "Result"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
    )
    loop._current_frame_id = "test_parent_id"

    context = SessionContext(
        messages=[Message(role=MessageRole.USER, content="Test")],
        files={},
        tool_outputs=[],
        working_memory={},
    )
    repl = RLMEnvironment(
        context,
        llm_client=mock_client,
        loop=loop,
    )

    # All three aliases should map to _recursive_query
    assert "llm" in repl.globals
    assert "recursive_llm" in repl.globals
    assert "recursive_query" in repl.globals

    # All should reference the same function
    assert repl.globals["llm"] == repl._recursive_query
    assert repl.globals["recursive_llm"] == repl._recursive_query
    assert repl.globals["recursive_query"] == repl._recursive_query


@pytest.mark.asyncio
async def test_repl_llm_passes_context_correctly():
    """
    Test that REPL llm() passes context string to loop.llm_sync.
    """
    mock_client = Mock()
    mock_client.call.return_value = "Context-aware result"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=3,
        llm_client=mock_client,
    )
    loop._current_frame_id = "parent_id"

    context = SessionContext(
        messages=[Message(role=MessageRole.USER, content="Test")],
        files={},
        tool_outputs=[],
        working_memory={},
    )
    repl = RLMEnvironment(
        context,
        llm_client=mock_client,
        loop=loop,
    )

    # Call with context
    result = repl._recursive_query(
        "Summarize this",
        context="Prior context from parent frame",
    )

    assert result == "Context-aware result"
    mock_client.call.assert_called_once()

    # Verify context was passed correctly
    call_kwargs = mock_client.call.call_args.kwargs
    assert "prior" in call_kwargs.get("context", {})
    assert "Prior context from parent frame" in call_kwargs["context"]["prior"]


@pytest.mark.asyncio
async def test_repl_llm_respects_max_depth():
    """
    Test that REPL llm() raises RecursionDepthError when exceeded.
    """
    from src.types import RecursionDepthError

    mock_client = Mock()
    mock_client.call.return_value = "Result"

    loop = RLAPHLoop(
        max_iterations=5,
        max_depth=2,  # Max depth is 2
        llm_client=mock_client,
    )
    loop._current_frame_id = "parent_id"

    context = SessionContext(
        messages=[Message(role=MessageRole.USER, content="Test")],
        files={},
        tool_outputs=[],
        working_memory={},
    )
    repl = RLMEnvironment(
        context,
        llm_client=mock_client,
        loop=loop,
    )

    # Calling at depth 3 should raise (exceeds max_depth=2)
    with pytest.raises(RecursionDepthError) as exc_info:
        repl._recursive_query("Query", depth=3)

    assert exc_info.value.depth == 3
    assert exc_info.value.max_depth == 2
