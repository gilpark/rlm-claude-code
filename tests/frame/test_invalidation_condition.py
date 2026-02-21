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

    assert isinstance(result, dict)
    assert "files" in result
    assert "tools" in result
    assert "memory_refs" in result
    assert "description" in result
    assert result["files"] == ["/path/to/file.py"]
    assert result["description"] == "file.py changes or is deleted"


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

    assert isinstance(result, dict)
    assert len(result["files"]) == 3
    assert "any of 3 files" in result["description"]
    assert "file.py" in result["description"]


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

    assert isinstance(result, dict)
    assert len(result["files"]) == 5
    assert "any of 5 files" in result["description"]
    assert "(+2 more)" in result["description"]


def test_empty_context() -> None:
    """Test invalidation condition with empty context."""
    context_slice = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert isinstance(result, dict)
    assert result["files"] == []
    assert result["tools"] == []
    assert result["memory_refs"] == []
    assert result["description"] == "No automatic invalidation condition"


def test_tool_outputs() -> None:
    """Test invalidation condition with tool outputs."""
    context_slice = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={"grep": "hash123"},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert isinstance(result, dict)
    assert result["tools"] == ["grep"]
    assert result["description"] == "tool results from grep change"


def test_multiple_tool_outputs() -> None:
    """Test invalidation condition with multiple tool outputs."""
    context_slice = ContextSlice(
        files={},
        memory_refs=[],
        tool_outputs={"grep": "hash123", "find": "hash456"},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert isinstance(result, dict)
    assert set(result["tools"]) == {"grep", "find"}
    assert "grep" in result["description"]
    assert "find" in result["description"]


def test_memory_refs() -> None:
    """Test invalidation condition with memory references."""
    context_slice = ContextSlice(
        files={},
        memory_refs=["mem1", "mem2"],
        tool_outputs={},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert isinstance(result, dict)
    assert result["memory_refs"] == ["mem1", "mem2"]
    assert result["description"] == "memory entries change"


def test_combined_files_and_tools() -> None:
    """Test invalidation condition with both files and tools."""
    context_slice = ContextSlice(
        files={"/path/to/file.py": "hash123"},
        memory_refs=[],
        tool_outputs={"grep": "hash456"},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert isinstance(result, dict)
    assert result["files"] == ["/path/to/file.py"]
    assert result["tools"] == ["grep"]
    assert "file.py changes or is deleted" in result["description"]
    assert "tool results from grep change" in result["description"]


def test_all_components() -> None:
    """Test invalidation condition with all components."""
    context_slice = ContextSlice(
        files={"/path/to/file.py": "hash123"},
        memory_refs=["mem1"],
        tool_outputs={"grep": "hash456"},
        token_budget=1000,
    )

    result = generate_invalidation_condition(context_slice)

    assert isinstance(result, dict)
    assert result["files"] == ["/path/to/file.py"]
    assert result["tools"] == ["grep"]
    assert result["memory_refs"] == ["mem1"]
    assert "file.py changes or is deleted" in result["description"]
    assert "tool results from grep change" in result["description"]
    assert "memory entries change" in result["description"]
