"""Tests for ContextSlice dataclass."""

from src.context_slice import ContextSlice


def test_context_slice_hash_is_deterministic():
    """Same content should produce same hash."""
    slice1 = ContextSlice(
        files={"a.py": "hash1"},
        memory_refs=["mem1"],
        tool_outputs={"grep": "out1"},
        token_budget=1000
    )
    slice2 = ContextSlice(
        files={"a.py": "hash1"},
        memory_refs=["mem1"],
        tool_outputs={"grep": "out1"},
        token_budget=1000
    )
    assert slice1.hash() == slice2.hash()


def test_context_slice_hash_changes_with_content():
    """Different content should produce different hash."""
    slice1 = ContextSlice(
        files={"a.py": "hash1"},
        memory_refs=["mem1"],
        tool_outputs={"grep": "out1"},
        token_budget=1000
    )
    slice2 = ContextSlice(
        files={"a.py": "hash2"},  # different hash
        memory_refs=["mem1"],
        tool_outputs={"grep": "out1"},
        token_budget=1000
    )
    assert slice1.hash() != slice2.hash()


def test_context_slice_hash_ignores_token_budget():
    """Token budget should not affect hash (not part of content identity)."""
    slice1 = ContextSlice(
        files={"a.py": "hash1"},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000
    )
    slice2 = ContextSlice(
        files={"a.py": "hash1"},
        memory_refs=[],
        tool_outputs={},
        token_budget=2000  # different budget
    )
    assert slice1.hash() == slice2.hash()  # budget not in hash


def test_context_slice_hash_is_16_chars():
    """Hash should be 16 hex characters (64 bits)."""
    slice1 = ContextSlice(
        files={"a.py": "hash1"},
        memory_refs=[],
        tool_outputs={},
        token_budget=1000
    )
    h = slice1.hash()
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
