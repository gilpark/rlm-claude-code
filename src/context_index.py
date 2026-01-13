"""
Hierarchical file index for massive context handling.

Implements: Phase 4 Massive Context (SPEC-01.04)

Indexes 10K+ files with summaries (imports, exports, structure).
Lazy-loads full content only when needed.
Budget-aware context retrieval using FTS5 search.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tokenization import count_tokens

if TYPE_CHECKING:
    from collections.abc import Iterator


# Patterns for extracting imports and exports
# Match: import os, sys OR from pathlib import Path
PYTHON_PLAIN_IMPORT = re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE)
PYTHON_FROM_IMPORT = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+", re.MULTILINE)
PYTHON_EXPORT_PATTERN = re.compile(r"^(?:def|class|async\s+def)\s+(\w+)", re.MULTILINE)

JS_TS_IMPORT_PATTERN = re.compile(r"^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)
JS_TS_EXPORT_PATTERN = re.compile(
    r"^\s*export\s+(?:default\s+)?(?:function|class|const|let|var|async\s+function)\s+(\w+)",
    re.MULTILINE,
)


@dataclass
class FileIndex:
    """
    Index entry for a single file.

    Stores metadata and summary without full content.
    """

    path: str
    token_count: int
    summary: str  # First ~100 tokens + structure info
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    content_hash: str = ""  # For change detection
    mtime: float = 0.0  # File modification time


@dataclass
class IndexStats:
    """Statistics about the file index."""

    total_files: int
    total_tokens: int
    indexed_extensions: dict[str, int]
    avg_tokens_per_file: float


class ContextIndex:
    """
    Hierarchical file index with FTS5 search and budget-aware retrieval.

    Implements: Phase 4 Massive Context (SPEC-01.04)

    Example:
        >>> index = ContextIndex("/path/to/db")
        >>> index.index_directory("/path/to/codebase")
        >>> files = index.get_relevant_context("authentication", budget_tokens=4000)
    """

    # File extensions to index
    INDEXABLE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".html",
        ".css",
        ".scss",
        ".vue",
        ".svelte",
    }

    # Directories to skip
    SKIP_DIRS = {
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "venv",
        ".venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        "coverage",
        ".coverage",
    }

    def __init__(self, db_path: str | None = None):
        """
        Initialize the context index.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory database.
        """
        self.db_path = db_path or ":memory:"
        self._is_memory = self.db_path == ":memory:"
        self._persistent_conn: sqlite3.Connection | None = None
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema with FTS5."""
        conn = self._get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS file_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    token_count INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    imports TEXT,  -- JSON array
                    exports TEXT,  -- JSON array
                    content_hash TEXT,
                    mtime REAL,
                    indexed_at INTEGER DEFAULT (strftime('%s', 'now') * 1000)
                );

                CREATE INDEX IF NOT EXISTS idx_file_path ON file_index(path);
                CREATE INDEX IF NOT EXISTS idx_file_tokens ON file_index(token_count);

                -- FTS5 virtual table for file search
                CREATE VIRTUAL TABLE IF NOT EXISTS file_fts USING fts5(
                    path,
                    summary,
                    imports,
                    exports,
                    content='file_index',
                    content_rowid='id',
                    tokenize='porter'
                );

                -- Triggers to keep FTS in sync
                CREATE TRIGGER IF NOT EXISTS file_fts_insert AFTER INSERT ON file_index BEGIN
                    INSERT INTO file_fts(rowid, path, summary, imports, exports)
                    VALUES (NEW.id, NEW.path, NEW.summary, NEW.imports, NEW.exports);
                END;

                CREATE TRIGGER IF NOT EXISTS file_fts_update AFTER UPDATE ON file_index BEGIN
                    INSERT INTO file_fts(file_fts, rowid, path, summary, imports, exports)
                    VALUES ('delete', OLD.id, OLD.path, OLD.summary, OLD.imports, OLD.exports);
                    INSERT INTO file_fts(rowid, path, summary, imports, exports)
                    VALUES (NEW.id, NEW.path, NEW.summary, NEW.imports, NEW.exports);
                END;

                CREATE TRIGGER IF NOT EXISTS file_fts_delete AFTER DELETE ON file_index BEGIN
                    INSERT INTO file_fts(file_fts, rowid, path, summary, imports, exports)
                    VALUES ('delete', OLD.id, OLD.path, OLD.summary, OLD.imports, OLD.exports);
                END;
            """
            )
            conn.commit()
        finally:
            self._close_connection(conn)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if self._is_memory:
            # For in-memory databases, reuse persistent connection
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(self.db_path)
                self._persistent_conn.row_factory = sqlite3.Row
            return self._persistent_conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

    def _close_connection(self, conn: sqlite3.Connection) -> None:
        """Close a connection (no-op for in-memory persistent connection)."""
        if not self._is_memory:
            self._close_connection(conn)

    def index_file(self, file_path: str | Path, base_path: str | Path = "") -> FileIndex | None:
        """
        Index a single file.

        Args:
            file_path: Path to file
            base_path: Base path for relative path calculation

        Returns:
            FileIndex if indexed, None if skipped
        """
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            return None

        # Check extension
        if file_path.suffix.lower() not in self.INDEXABLE_EXTENSIONS:
            return None

        # Calculate relative path
        if base_path:
            try:
                rel_path = str(file_path.relative_to(base_path))
            except ValueError:
                rel_path = str(file_path)
        else:
            rel_path = str(file_path)

        # Read content
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return None

        # Get file metadata
        stat = file_path.stat()
        content_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

        # Check if already indexed and unchanged
        existing = self._get_file_entry(rel_path)
        if existing and existing["content_hash"] == content_hash:
            return FileIndex(
                path=rel_path,
                token_count=existing["token_count"],
                summary=existing["summary"],
                imports=self._parse_json_list(existing["imports"]),
                exports=self._parse_json_list(existing["exports"]),
                content_hash=content_hash,
                mtime=existing["mtime"],
            )

        # Extract metadata
        token_count = count_tokens(content)
        summary = self._create_summary(content, file_path.suffix)
        imports = self._extract_imports(content, file_path.suffix)
        exports = self._extract_exports(content, file_path.suffix)

        # Store in database
        import json

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_index
                (path, token_count, summary, imports, exports, content_hash, mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rel_path,
                    token_count,
                    summary,
                    json.dumps(imports),
                    json.dumps(exports),
                    content_hash,
                    stat.st_mtime,
                ),
            )
            conn.commit()
        finally:
            self._close_connection(conn)

        return FileIndex(
            path=rel_path,
            token_count=token_count,
            summary=summary,
            imports=imports,
            exports=exports,
            content_hash=content_hash,
            mtime=stat.st_mtime,
        )

    def index_directory(
        self,
        directory: str | Path,
        max_files: int | None = None,
        progress_callback: Any | None = None,
    ) -> int:
        """
        Index all files in a directory recursively.

        Args:
            directory: Directory to index
            max_files: Maximum files to index (None for unlimited)
            progress_callback: Optional callback(indexed, total) for progress

        Returns:
            Number of files indexed
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            return 0

        indexed = 0
        files_to_index = list(self._walk_files(directory))
        total = len(files_to_index) if max_files is None else min(len(files_to_index), max_files)

        for file_path in files_to_index:
            if max_files is not None and indexed >= max_files:
                break

            result = self.index_file(file_path, base_path=directory)
            if result:
                indexed += 1
                if progress_callback:
                    progress_callback(indexed, total)

        return indexed

    def _walk_files(self, directory: Path) -> Iterator[Path]:
        """Walk directory yielding indexable files."""
        for root, dirs, files in os.walk(directory):
            # Filter out skip directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.INDEXABLE_EXTENSIONS:
                    yield file_path

    def _get_file_entry(self, path: str) -> dict[str, Any] | None:
        """Get existing file entry from database."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM file_index WHERE path = ?", (path,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            self._close_connection(conn)

    def _parse_json_list(self, json_str: str | None) -> list[str]:
        """Parse JSON list string."""
        if not json_str:
            return []
        import json

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return []

    def _create_summary(self, content: str, suffix: str) -> str:
        """
        Create a summary of file content.

        Includes first ~100 tokens plus structure information.
        """
        lines = content.split("\n")

        # Get first meaningful lines (skip empty and comments for summary)
        summary_lines = []
        token_count = 0
        max_summary_tokens = 100

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            line_tokens = count_tokens(line)
            if token_count + line_tokens > max_summary_tokens:
                break

            summary_lines.append(line)
            token_count += line_tokens

        summary = "\n".join(summary_lines)

        # Add structure info
        if suffix in {".py"}:
            classes = len(re.findall(r"^class\s+\w+", content, re.MULTILINE))
            functions = len(re.findall(r"^(?:def|async\s+def)\s+\w+", content, re.MULTILINE))
            if classes or functions:
                summary += f"\n[Structure: {classes} classes, {functions} functions]"

        elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
            classes = len(re.findall(r"^\s*class\s+\w+", content, re.MULTILINE))
            functions = len(
                re.findall(r"^\s*(?:export\s+)?(?:async\s+)?function\s+\w+", content, re.MULTILINE)
            )
            if classes or functions:
                summary += f"\n[Structure: {classes} classes, {functions} functions]"

        return summary

    def _extract_imports(self, content: str, suffix: str) -> list[str]:
        """Extract import statements from file content."""
        imports = []

        if suffix == ".py":
            # Match plain imports: import os
            for match in PYTHON_PLAIN_IMPORT.finditer(content):
                imports.append(match.group(1))
            # Match from imports: from pathlib import Path
            for match in PYTHON_FROM_IMPORT.finditer(content):
                imports.append(match.group(1))

        elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
            for match in JS_TS_IMPORT_PATTERN.finditer(content):
                imports.append(match.group(1))

        return list(set(imports))[:20]  # Limit to 20 unique imports

    def _extract_exports(self, content: str, suffix: str) -> list[str]:
        """Extract exported symbols from file content."""
        exports = []

        if suffix == ".py":
            for match in PYTHON_EXPORT_PATTERN.finditer(content):
                exports.append(match.group(1))

        elif suffix in {".js", ".ts", ".jsx", ".tsx"}:
            for match in JS_TS_EXPORT_PATTERN.finditer(content):
                exports.append(match.group(1))

        return exports[:30]  # Limit to 30 exports

    def search(self, query: str, limit: int = 20) -> list[FileIndex]:
        """
        Search files by content/path/imports/exports.

        Args:
            query: FTS5 search query
            limit: Maximum results

        Returns:
            List of FileIndex entries sorted by relevance
        """
        if not query or not query.strip():
            return []

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT f.*, bm25(file_fts) as score
                FROM file_index f
                JOIN file_fts ON f.id = file_fts.rowid
                WHERE file_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """,
                (query, limit),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    FileIndex(
                        path=row["path"],
                        token_count=row["token_count"],
                        summary=row["summary"],
                        imports=self._parse_json_list(row["imports"]),
                        exports=self._parse_json_list(row["exports"]),
                        content_hash=row["content_hash"] or "",
                        mtime=row["mtime"] or 0.0,
                    )
                )
            return results

        except sqlite3.OperationalError:
            # Invalid FTS query syntax
            return []
        finally:
            self._close_connection(conn)

    def get_relevant_context(
        self,
        query: str,
        budget_tokens: int,
        base_path: str | Path | None = None,
    ) -> dict[str, str]:
        """
        Get relevant file contents that fit within token budget.

        Args:
            query: Search query
            budget_tokens: Maximum total tokens to return
            base_path: Base path for reading file contents

        Returns:
            Dict mapping file paths to contents, prioritized by relevance
        """
        # Search for relevant files
        results = self.search(query, limit=100)

        if not results:
            return {}

        # Fit files into budget by relevance order
        context: dict[str, str] = {}
        remaining_budget = budget_tokens

        for file_entry in results:
            if file_entry.token_count > remaining_budget:
                continue

            # Read full content
            if base_path:
                full_path = Path(base_path) / file_entry.path
            else:
                full_path = Path(file_entry.path)

            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                context[file_entry.path] = content
                remaining_budget -= file_entry.token_count
            except (OSError, UnicodeDecodeError):
                # Use summary as fallback
                context[file_entry.path] = file_entry.summary
                remaining_budget -= count_tokens(file_entry.summary)

            if remaining_budget <= 0:
                break

        return context

    def get_file(self, path: str) -> FileIndex | None:
        """
        Get a file entry by path.

        Args:
            path: File path

        Returns:
            FileIndex if found, None otherwise
        """
        entry = self._get_file_entry(path)
        if not entry:
            return None

        return FileIndex(
            path=entry["path"],
            token_count=entry["token_count"],
            summary=entry["summary"],
            imports=self._parse_json_list(entry["imports"]),
            exports=self._parse_json_list(entry["exports"]),
            content_hash=entry["content_hash"] or "",
            mtime=entry["mtime"] or 0.0,
        )

    def get_stats(self) -> IndexStats:
        """Get statistics about the index."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total_files,
                    COALESCE(SUM(token_count), 0) as total_tokens
                FROM file_index
            """
            )
            row = cursor.fetchone()
            total_files = row["total_files"]
            total_tokens = row["total_tokens"]

            # Get extension breakdown
            cursor = conn.execute(
                """
                SELECT
                    SUBSTR(path, INSTR(path, '.')) as ext,
                    COUNT(*) as count
                FROM file_index
                WHERE INSTR(path, '.') > 0
                GROUP BY ext
                ORDER BY count DESC
            """
            )
            extensions = {row["ext"]: row["count"] for row in cursor.fetchall()}

            return IndexStats(
                total_files=total_files,
                total_tokens=total_tokens,
                indexed_extensions=extensions,
                avg_tokens_per_file=total_tokens / total_files if total_files > 0 else 0,
            )

        finally:
            self._close_connection(conn)

    def clear(self) -> int:
        """
        Clear all indexed files.

        Returns:
            Number of files removed
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) as count FROM file_index")
            count = cursor.fetchone()["count"]

            conn.execute("DELETE FROM file_index")
            conn.commit()
            return count
        finally:
            self._close_connection(conn)

    def remove_file(self, path: str) -> bool:
        """
        Remove a file from the index.

        Args:
            path: File path to remove

        Returns:
            True if removed, False if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("DELETE FROM file_index WHERE path = ?", (path,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            self._close_connection(conn)

    def rebuild_fts_index(self) -> int:
        """
        Rebuild the FTS index from scratch.

        Returns:
            Number of files re-indexed
        """
        conn = self._get_connection()
        try:
            # Use FTS5 rebuild command for external content tables
            conn.execute("INSERT INTO file_fts(file_fts) VALUES('rebuild')")
            conn.commit()

            # Count indexed files
            cursor = conn.execute("SELECT COUNT(*) as count FROM file_index")
            count = cursor.fetchone()["count"]

            return count
        finally:
            self._close_connection(conn)


__all__ = [
    "FileIndex",
    "IndexStats",
    "ContextIndex",
]
