"""Tests for REPL ContextMap integration."""
import pytest
from pathlib import Path
from src.repl.repl_environment import RLMEnvironment
from src.frame.context_map import ContextMap
from src.types import SessionContext, Message, MessageRole


def test_repl_uses_context_map_for_read_file(tmp_path):
    """RLMEnvironment should use ContextMap for file reads."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    env = RLMEnvironment(context, context_map=cm)
    env.enable_file_access(tmp_path)

    # Read file should use ContextMap
    result = env._read_file("test.py")
    assert "hello" in result


def test_repl_rejects_file_outside_root(tmp_path):
    """RLMEnvironment should reject files outside working directory (security guard)."""
    # Create a file outside the root
    outside_file = tmp_path.parent / "outside_repl_test.py"
    outside_file.write_text("outside content")

    try:
        cm = ContextMap(tmp_path)

        context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
        env = RLMEnvironment(context, context_map=cm)
        env.enable_file_access(tmp_path)

        # Should raise error for file outside root
        with pytest.raises(FileNotFoundError) as exc_info:
            env._read_file(str(outside_file))

        assert "outside" in str(exc_info.value).lower()
    finally:
        outside_file.unlink()


def test_repl_auto_discovers_new_files(tmp_path):
    """RLMEnvironment should auto-discover new files within root (dynamic discovery)."""
    # Create a new file after ContextMap was created
    new_file = tmp_path / "auto_discovered.py"
    new_file.write_text("auto discovered content")

    cm = ContextMap(tmp_path)
    # Don't manually add to paths - should be auto-discovered

    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    env = RLMEnvironment(context, context_map=cm)
    env.enable_file_access(tmp_path)

    # Should work without manual add_path (dynamic discovery)
    result = env._read_file("auto_discovered.py")
    assert "auto discovered content" in result


def test_repl_context_map_tracks_files_read(tmp_path):
    """RLMEnvironment should track files read via ContextMap."""
    test_file = tmp_path / "test.py"
    test_file.write_text("test content")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    env = RLMEnvironment(context, context_map=cm)
    env.enable_file_access(tmp_path)

    env._read_file("test.py")

    # Should track in files_read (for frame context_slice)
    assert str(test_file.resolve()) in env.files_read


def test_repl_without_context_map_uses_fallback(tmp_path):
    """RLMEnvironment without ContextMap should auto-create one and still work.

    When enable_file_access is called without a pre-provided ContextMap,
    it auto-creates one from git files. Files in that context should be readable.
    """
    test_file = tmp_path / "fallback.py"
    test_file.write_text("fallback content")

    # No ContextMap provided - enable_file_access will create one
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    env = RLMEnvironment(context, context_map=None)
    env.enable_file_access(tmp_path)

    # Auto-created ContextMap should have added this file from git
    # But if it's not in git, we need to add it manually for the test
    if env._context_map is not None:
        env._context_map.add_path(test_file)

    # Should work with the auto-created ContextMap
    result = env._read_file("fallback.py")
    assert "fallback content" in result


def test_repl_context_map_caches_content(tmp_path):
    """ContextMap should cache content and not re-read from disk."""
    test_file = tmp_path / "cached.py"
    test_file.write_text("original")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
    env = RLMEnvironment(context, context_map=cm)
    env.enable_file_access(tmp_path)

    # First read
    result1 = env._read_file("cached.py")
    assert "original" in result1

    # Modify on disk
    test_file.write_text("modified")

    # Should return cached version
    result2 = env._read_file("cached.py")
    assert "original" in result2
