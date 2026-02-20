"""Tests for ContextMap."""
import pytest
from pathlib import Path
from src.frame.context_map import ContextMap


def test_context_map_initializes_with_root_dir(tmp_path):
    """ContextMap should initialize with root directory."""
    (tmp_path / "test.py").write_text("print('hello')")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.py").write_text("# nested")

    cm = ContextMap(tmp_path)

    assert cm.root == tmp_path.resolve()
    assert isinstance(cm.paths, set)
    assert isinstance(cm.hashes, dict)
    assert isinstance(cm.contents, dict)


def test_get_content_loads_lazily(tmp_path):
    """get_content should load file content on first access."""
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    # Content not loaded yet
    assert test_file not in cm.contents

    # Load it
    content = cm.get_content(test_file)
    assert content == "print('hello')"
    assert test_file in cm.contents


def test_get_content_caches_result(tmp_path):
    """get_content should cache and reuse content."""
    test_file = tmp_path / "test.py"
    test_file.write_text("original")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    content1 = cm.get_content(test_file)

    # Modify file on disk
    test_file.write_text("modified")

    # Should return cached version
    content2 = cm.get_content(test_file)
    assert content2 == "original"


def test_get_content_rejects_unknown_path(tmp_path):
    """get_content should raise for paths not in context."""
    cm = ContextMap(tmp_path)

    unknown_file = tmp_path / "unknown.py"
    unknown_file.write_text("unknown")

    with pytest.raises(ValueError, match="not in context"):
        cm.get_content(unknown_file)


def test_get_hash_computes_and_caches(tmp_path):
    """get_hash should compute hash and cache it."""
    test_file = tmp_path / "test.py"
    test_file.write_text("test content")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    hash1 = cm.get_hash(test_file)
    assert len(hash1) == 16  # blake2b truncated

    # Should be cached
    hash2 = cm.get_hash(test_file)
    assert hash1 == hash2


def test_refresh_from_diff_updates_paths(tmp_path):
    """refresh_from_diff should update paths and hashes."""
    test_file = tmp_path / "test.py"
    test_file.write_text("original")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)
    old_hash = cm.get_hash(test_file)

    # Modify file
    test_file.write_text("modified")

    # Refresh
    cm.refresh_from_diff({test_file})

    # Hash should be updated
    new_hash = cm.get_hash(test_file)
    assert new_hash != old_hash

    # Content cache should be cleared
    assert test_file not in cm.contents


def test_refresh_from_diff_handles_deleted_files(tmp_path):
    """refresh_from_diff should remove deleted files."""
    test_file = tmp_path / "deleted.py"
    test_file.write_text("will be deleted")

    cm = ContextMap(tmp_path)
    cm.paths.add(test_file)

    # Delete file
    test_file.unlink()

    # Refresh
    cm.refresh_from_diff({test_file})

    # Should be removed from paths
    assert test_file not in cm.paths
    assert test_file not in cm.hashes
