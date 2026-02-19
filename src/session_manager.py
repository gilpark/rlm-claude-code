"""
Session Manager for RLM-Claude-Code.

Implements: Session State Consolidation Plan Phase 1

Manages session-centric state in ~/.claude/rlm-sessions/{session_id}/
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .session_schema import (
    CellIndex,
    CellIndexEntry,
    CellType,
    SessionActivation,
    SessionBudget,
    SessionContextData,
    SessionMetadata,
    SessionState,
)

if TYPE_CHECKING:
    from .causal_frame import CausalFrame


class SessionManager:
    """
    Manages session-centric state in ~/.claude/rlm-sessions/{session_id}/

    Implements: Session State Consolidation Plan Phase 1

    This class extends the patterns from StatePersistence but organizes
    state into a session-centric directory structure.
    """

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize session manager.

        Args:
            base_dir: Base directory for sessions (default: ~/.claude/rlm-sessions)
        """
        self.base_dir = base_dir or (Path.home() / ".claude" / "rlm-sessions")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._current_session: SessionState | None = None
        self._current_session_dir: Path | None = None

    @property
    def current_session(self) -> SessionState | None:
        """Get current session state."""
        return self._current_session

    @property
    def current_session_dir(self) -> Path | None:
        """Get current session directory."""
        return self._current_session_dir

    def get_session_dir(self, session_id: str) -> Path:
        """Get session directory path."""
        return self.base_dir / session_id

    def get_session_file(self, session_id: str) -> Path:
        """Get session.json path."""
        return self.get_session_dir(session_id) / "session.json"

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return self.get_session_file(session_id).exists()

    def create_session(
        self,
        session_id: str,
        cwd: str,
        transcript_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionState:
        """
        Create a new session directory and initialize session.json.

        Args:
            session_id: Unique session identifier
            cwd: Current working directory
            transcript_path: Path to Claude's native transcript
            metadata: Additional metadata to include

        Returns:
            Newly created session state
        """
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (session_dir / "cells").mkdir(exist_ok=True)
        (session_dir / "reasoning").mkdir(exist_ok=True)

        now = time.time()
        state = SessionState(
            metadata=SessionMetadata(
                session_id=session_id,
                created_at=now,
                updated_at=now,
                cwd=cwd,
                claude_transcript_path=transcript_path,
                **(metadata or {}),
            ),
            activation=SessionActivation(),
            context=SessionContextData(),
            budget=SessionBudget(),
        )

        # Write initial session.json
        self._write_session_json(session_dir, state)

        # Update current symlink
        self._update_current_symlink(session_id)

        # Initialize empty cell index
        self._init_cell_index(session_dir, session_id)

        self._current_session = state
        self._current_session_dir = session_dir
        return state

    def load_session(self, session_id: str) -> SessionState:
        """
        Load session state from disk.

        Args:
            session_id: Session to load

        Returns:
            Loaded session state

        Raises:
            FileNotFoundError: If session not found
        """
        session_dir = self.get_session_dir(session_id)
        session_file = self.get_session_file(session_id)

        if not session_file.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(session_file) as f:
            data = json.load(f)

        state = SessionState(**data)
        self._current_session = state
        self._current_session_dir = session_dir
        return state

    def save_session(self, state: SessionState | None = None) -> Path:
        """
        Save current session state to disk.

        Uses atomic write to prevent corruption.

        Args:
            state: State to save (uses current if not provided)

        Returns:
            Path to saved session.json

        Raises:
            ValueError: If no session to save
        """
        state = state or self._current_session
        if state is None:
            raise ValueError("No session to save")

        session_dir = self._current_session_dir or self.get_session_dir(
            state.metadata.session_id
        )

        state.metadata.updated_at = time.time()
        self._write_session_json(session_dir, state)

        return session_dir / "session.json"

    def _write_session_json(self, session_dir: Path, state: SessionState) -> None:
        """
        Atomically write session.json.

        Uses write-to-temp-then-rename pattern to prevent corruption.
        """
        target_path = session_dir / "session.json"

        # Write to temp file in same directory
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="session_",
            dir=session_dir,
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(state.model_dump_json(indent=2))
            # Atomic rename
            os.rename(temp_path, target_path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _update_current_symlink(self, session_id: str) -> None:
        """Update 'current' symlink to point to active session."""
        current_link = self.base_dir / "current"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(session_id)

    def _init_cell_index(self, session_dir: Path, session_id: str) -> None:
        """Initialize empty cell index."""
        index = CellIndex(session_id=session_id)
        index_file = session_dir / "cells" / "index.json"
        index_file.write_text(index.model_dump_json(indent=2))

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session directory.

        Args:
            session_id: Session to delete
        """
        session_dir = self.get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)

        # Clear current session if deleted
        if self._current_session and self._current_session.metadata.session_id == session_id:
            self._current_session = None
            self._current_session_dir = None

    def list_sessions(
        self,
        session_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """
        List all session IDs with optional filtering.

        Args:
            session_type: Filter by session type
            tags: Filter by tags (sessions must have ALL specified tags)

        Returns:
            List of matching session IDs
        """
        sessions = []
        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            if session_dir.name in ("current", "archive"):
                continue

            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue

            try:
                with open(session_file) as f:
                    data = json.load(f)

                # Apply filters
                if session_type and data.get("metadata", {}).get("session_type") != session_type:
                    continue

                if tags:
                    session_tags = set(data.get("metadata", {}).get("tags", []))
                    if not all(t in session_tags for t in tags):
                        continue

                sessions.append(session_dir.name)
            except (json.JSONDecodeError, KeyError):
                # Skip invalid sessions
                continue

        return sorted(sessions)

    def get_current_session_id(self) -> str | None:
        """Get current session ID from symlink."""
        current_link = self.base_dir / "current"
        if current_link.is_symlink():
            return current_link.resolve().name
        return None

    def archive_session(self, session_id: str) -> Path:
        """
        Archive a session to the archive directory.

        Args:
            session_id: Session to archive

        Returns:
            Path to archived session
        """
        session_dir = self.get_session_dir(session_id)
        archive_dir = self.base_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        archived_path = archive_dir / session_id
        if session_dir.exists():
            shutil.move(str(session_dir), str(archived_path))

        return archived_path

    def fork_session(self, session_id: str, new_session_id: str) -> SessionState:
        """
        Fork a session to a new session ID.

        Args:
            session_id: Session to fork
            new_session_id: New session ID

        Returns:
            New forked session state
        """
        # Load original
        original = self.load_session(session_id)

        # Create fork
        now = time.time()
        forked = SessionState(
            metadata=SessionMetadata(
                session_id=new_session_id,
                created_at=now,
                updated_at=now,
                cwd=original.metadata.cwd,
                claude_transcript_path=None,  # New transcript for fork
                parent_session_id=session_id,
                session_type="fork",
                tags=original.metadata.tags.copy(),
                description=f"Forked from {session_id}",
            ),
            activation=SessionActivation(),
            context=SessionContextData(
                files=original.context.files.copy(),
                working_memory=original.context.working_memory.copy(),
            ),
            budget=SessionBudget(),
        )

        # Create new session directory
        new_session_dir = self.get_session_dir(new_session_id)
        new_session_dir.mkdir(parents=True, exist_ok=True)
        (new_session_dir / "cells").mkdir(exist_ok=True)
        (new_session_dir / "reasoning").mkdir(exist_ok=True)

        # Write forked state
        self._write_session_json(new_session_dir, forked)
        self._init_cell_index(new_session_dir, new_session_id)

        return forked

    def update_activation(self, active: bool, reason: str | None = None) -> None:
        """Update RLM activation status."""
        if self._current_session:
            self._current_session.activation.rlm_active = active
            self._current_session.activation.activation_reason = reason
            self._current_session.metadata.updated_at = time.time()

    def update_depth(self, depth: int) -> None:
        """Update current recursion depth."""
        if self._current_session:
            self._current_session.activation.current_depth = depth
            self._current_session.metadata.updated_at = time.time()

    def add_tokens(self, tokens: int, model: str | None = None, cost: float = 0.0) -> None:
        """Add to token and cost tracking."""
        if self._current_session:
            self._current_session.budget.total_tokens_used += tokens
            self._current_session.budget.cost_usd += cost
            if model:
                if model not in self._current_session.budget.by_model:
                    self._current_session.budget.by_model[model] = {
                        "calls": 0,
                        "tokens": 0,
                        "cost_usd": 0.0,
                    }
                self._current_session.budget.by_model[model]["calls"] += 1
                self._current_session.budget.by_model[model]["tokens"] += tokens
                self._current_session.budget.by_model[model]["cost_usd"] += cost
            self._current_session.metadata.updated_at = time.time()

    def increment_recursive_calls(self, count: int = 1) -> None:
        """Increment recursive call counter."""
        if self._current_session:
            self._current_session.budget.total_recursive_calls += count
            self._current_session.metadata.updated_at = time.time()

    def update_working_memory(self, key: str, value: Any) -> None:
        """Update working memory."""
        if self._current_session:
            self._current_session.context.working_memory[key] = value
            self._current_session.metadata.updated_at = time.time()

    # =========================================================================
    # Causal Frame Storage Methods
    # =========================================================================

    def _get_frames_dir(self) -> Path | None:
        """Get frames directory for current session."""
        if not self._current_session_dir:
            return None
        frames_dir = self._current_session_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        return frames_dir

    def _save_frame_to_disk(self, frame: "CausalFrame") -> None:
        """Persist frame to disk."""
        from .frame_serialization import serialize_frame

        frames_dir = self._get_frames_dir()
        if frames_dir:
            frame_file = frames_dir / f"{frame.frame_id}.json"
            frame_file.write_text(json.dumps(serialize_frame(frame), indent=2))

    def save_frame(self, frame: "CausalFrame") -> str:
        """
        Save a CausalFrame to the current session.

        Updates parent's children list if parent_id is set.

        Args:
            frame: CausalFrame to save

        Returns:
            frame_id of saved frame
        """
        if not self._current_session:
            raise ValueError("No active session")

        # Update parent's children list
        if frame.parent_id:
            parent = self.load_frame(frame.parent_id)
            if parent and frame.frame_id not in parent.children:
                parent.children.append(frame.frame_id)
                self._save_frame_to_disk(parent)
                # Update in-memory parent
                for i, f in enumerate(self._current_session.context.causal_frames):
                    if f.frame_id == frame.parent_id:
                        self._current_session.context.causal_frames[i] = parent
                        break

        # Add to session's causal_frames list (or update existing)
        existing_idx = None
        for i, f in enumerate(self._current_session.context.causal_frames):
            if f.frame_id == frame.frame_id:
                existing_idx = i
                break

        if existing_idx is not None:
            self._current_session.context.causal_frames[existing_idx] = frame
        else:
            self._current_session.context.causal_frames.append(frame)

        # Persist to disk
        self._save_frame_to_disk(frame)

        return frame.frame_id

    def load_frame(self, frame_id: str) -> "CausalFrame | None":
        """
        Load a CausalFrame by ID.

        Args:
            frame_id: Frame identifier

        Returns:
            CausalFrame or None if not found
        """
        from .frame_serialization import deserialize_frame

        if not self._current_session:
            return None

        # Check in-memory first
        for frame in self._current_session.context.causal_frames:
            if frame.frame_id == frame_id:
                return frame

        # Try loading from disk
        frames_dir = self._get_frames_dir()
        if frames_dir:
            frame_file = frames_dir / f"{frame_id}.json"
            if frame_file.exists():
                return deserialize_frame(json.loads(frame_file.read_text()))

        return None

    def get_session_frames(self) -> list["CausalFrame"]:
        """
        Get all CausalFrames for current session.

        Returns:
            List of CausalFrames
        """
        if not self._current_session:
            return []

        return list(self._current_session.context.causal_frames)

    def end_session(self) -> None:
        """Mark session as ended."""
        if self._current_session:
            self._current_session.metadata.ended_at = time.time()
            self.save_session()


# Global instance for convenience
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


__all__ = [
    "SessionManager",
    "get_session_manager",
]
