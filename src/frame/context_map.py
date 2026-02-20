"""ContextMap - minimal context externalization for REPL navigation."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ContextMap:
    """
    Minimal context externalization for REPL navigation.

    REPL navigates this map instead of reading files directly.
    This enables:
    - Spatial externalization (REPL doesn't touch disk)
    - Hash-based invalidation
    - Lazy content loading with caching
    - Git-aware change detection

    Usage:
        cm = ContextMap(Path.cwd())
        content = cm.get_content("src/main.py")  # lazy load
        cm.refresh_from_diff({Path("src/main.py")})  # after edit
    """

    root: Path
    paths: set[Path] = field(default_factory=set)
    hashes: dict[Path, str] = field(default_factory=dict)
    contents: dict[Path, str] = field(default_factory=dict)
    commit_hash: str | None = field(default=None, init=False)

    def __post_init__(self):
        """Initialize paths from git ls-files or empty."""
        self.root = Path(self.root).resolve()
        self._populate_from_git()

    def _populate_from_git(self) -> None:
        """Populate paths from git ls-files if in git repo."""
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    if line:
                        p = self.root / line
                        if p.is_file():
                            self.paths.add(p)

                # Also get commit hash
                hash_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if hash_result.returncode == 0:
                    self.commit_hash = hash_result.stdout.strip()[:8]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Not a git repo or git not available

    def _compute_hash(self, path: Path) -> str:
        """Compute blake2b hash of file content (truncated to 16 chars)."""
        return hashlib.blake2b(path.read_bytes()).hexdigest()[:16]

    def get_content(self, path: str | Path) -> str:
        """
        Get file content, loading lazily if needed.

        Args:
            path: File path (relative to root or absolute)

        Returns:
            File content as string

        Raises:
            ValueError: If path not in context
        """
        p = Path(path) if not isinstance(path, Path) else path
        if not p.is_absolute():
            p = self.root / p

        p = p.resolve()

        if p not in self.paths:
            raise ValueError(f"Path not in context: {path}")

        if p not in self.contents:
            self.contents[p] = p.read_text(encoding="utf-8", errors="replace")
            # Update hash on first load
            if p not in self.hashes:
                self.hashes[p] = self._compute_hash(p)

        return self.contents[p]

    def get_hash(self, path: Path) -> str:
        """
        Get content hash, computing if needed.

        Args:
            path: Absolute file path

        Returns:
            16-char blake2b hash
        """
        if path not in self.hashes:
            self.hashes[path] = self._compute_hash(path)
        return self.hashes[path]

    def refresh_from_diff(self, changed_paths: set[Path]) -> None:
        """
        Refresh paths after files changed.

        Called by PostToolUse hook after Write/Edit operations.

        Args:
            changed_paths: Set of paths that may have changed
        """
        for p in changed_paths:
            p = p.resolve()

            if p.exists() and p.is_file():
                # Add to paths if new
                self.paths.add(p)
                # Recompute hash
                self.hashes[p] = self._compute_hash(p)
                # Drop cached content (will reload on next access)
                self.contents.pop(p, None)
            else:
                # File deleted - remove from all caches
                self.paths.discard(p)
                self.hashes.pop(p, None)
                self.contents.pop(p, None)

    def add_path(self, path: str | Path) -> None:
        """
        Add a path to the context (e.g., discovered via glob).

        Args:
            path: File path to add
        """
        p = Path(path) if not isinstance(path, Path) else path
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()

        if p.exists() and p.is_file():
            self.paths.add(p)


def detect_changed_files(old_commit: str, root_dir: Path) -> set[Path]:
    """
    Detect files changed between old commit and HEAD.

    Args:
        old_commit: Git commit hash (can be short)
        root_dir: Repository root directory

    Returns:
        Set of changed file paths (absolute)
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{old_commit}..HEAD"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return set()

        changed = set()
        for line in result.stdout.strip().splitlines():
            if line:
                p = root_dir / line
                if p.exists():
                    changed.add(p.resolve())
        return changed
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()


def get_current_commit_hash(root_dir: Path) -> str | None:
    """
    Get current git HEAD commit hash.

    Args:
        root_dir: Repository root directory

    Returns:
        8-char commit hash or None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


__all__ = ["ContextMap", "detect_changed_files", "get_current_commit_hash"]
