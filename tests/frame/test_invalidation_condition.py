"""Tests for generate_invalidation_condition helper."""

from src.frame.context_slice import ContextSlice
from src.frame.causal_frame import generate_invalidation_condition


def test_single_file() -> None:
    """Test invalidation condition with single file."""
    context_slice = ContextSlice(
        files={"/path/to/file.py": "hash123"},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert result == "file.py changes or is deleted"


def test_multiple_files() -> None:
    """Test invalidation condition with multiple files (<=3)."""
    context_slice = ContextSlice(
        files={
            "/path/to/file.py": "hash123",
            "/another/file.py": "hash456",
            "/third/file.py": "hash789",
        },
        memory_refs=[],
        tool_outputs={},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert result == "any of 3 files (file.py, file.py, file.py) change"


def test_multiple_files_with_more() -> None:
    """Test invalidation condition with many files (>3)."""
    context_slice = ContextSlice(
        files={
            "/path/to/file1.py": "hash1",
            "/path/to/file2.py": "hash2",
            "/path/to/file3.py": "hash3",
            "/path/to/file4.py": "hash4",
            "/path/to/file5.py": "hash5",
        },
        memory_refs=[],
        tool_outputs={},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert result == "any of 5 files (file1.py, file2.py, file3.py (+2 more)) change"


def test_empty_context() -> None:
    """Test invalidation condition with empty context."""
    context_slice = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert result == "No automatic invalidation condition"


def test_tool_outputs() -> None:
    """Test invalidation condition with tool outputs."""
    context_slice = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={"grep": "hash123"},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert result == "tool results from grep change"


def test_multiple_tool_outputs() -> None:
    """Test invalidation condition with multiple tool outputs."""
    context_slice = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={"grep": "hash123", "find": "hash456"},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert result == "tool results from grep, find change"


def test_memory_refs() -> None:
    """Test invalidation condition with memory references."""
    context_slice = ContextSlice(
        files={},
        memory_refs=["mem1", "mem2"],
        tool_outputs={},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert result == "memory entries change"


def test_combined_files_and_tools() -> None:
    """Test invalidation condition with both files and tools."""
    context_slice = ContextSlice(
        files={"/path/to/file.py": "hash123"},
        memory_refs=[],
        tool_outputs={"grep": "hash456"},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert result == "file.py changes or is deleted; or tool results from grep change"


def test_all_components() -> None:
    """Test invalidation condition with all components."""
    context_slice = ContextSlice(
        files={"/path/to/file.py": "hash123"},
        memory_refs=["mem1"],
        tool_outputs={"grep": "hash456"},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    expected = "file.py changes or is deleted; or tool results from grep change; or memory entries change"
    assert result == expected
