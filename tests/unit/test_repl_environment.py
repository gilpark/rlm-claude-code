"""
Tests for REPL environment async file I/O operations.

Implements: Task 8 - Async File I/O for parallel file reads
"""

import asyncio
from pathlib import Path

import pytest

from src.repl.repl_environment import RLMEnvironment
from src.types import SessionContext


class TestAsyncFileIO:
    """Tests for async file I/O operations."""

    @pytest.fixture
    def env_with_files(self, tmp_path):
        """Create environment with test files."""
        # Create test files
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file3 = tmp_path / "file3.py"
        file1.write_text("print('file1')")
        file2.write_text("print('file2')")
        file3.write_text("print('file3')")

        # Create environment
        context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})
        env = RLMEnvironment(context)
        env.enable_file_access(working_dir=tmp_path)

        # Add files to ContextMap (since tmp_path isn't a git repo)
        if env._context_map is not None:
            env._context_map.add_path(file1)
            env._context_map.add_path(file2)
            env._context_map.add_path(file3)

        return env, tmp_path

    @pytest.mark.asyncio
    async def test_read_file_async_single(self, env_with_files):
        """read_file_async returns file content."""
        env, tmp_path = env_with_files

        content = await env.read_file_async("file1.py")
        assert "file1" in content

    @pytest.mark.asyncio
    async def test_read_files_async_parallel(self, env_with_files):
        """read_files_async reads multiple files in parallel."""
        env, tmp_path = env_with_files

        files = ["file1.py", "file2.py", "file3.py"]
        results = await env.read_files_async(files)

        assert len(results) == 3
        assert "file1" in results["file1.py"]
        assert "file2" in results["file2.py"]
        assert "file3" in results["file3.py"]

    @pytest.mark.asyncio
    async def test_read_files_async_handles_missing(self, env_with_files):
        """read_files_async handles missing files gracefully."""
        env, tmp_path = env_with_files

        files = ["file1.py", "nonexistent.py"]
        results = await env.read_files_async(files)

        assert "file1.py" in results
        assert results.get("nonexistent.py") is None or "error" in str(results.get("nonexistent.py", "")).lower()
