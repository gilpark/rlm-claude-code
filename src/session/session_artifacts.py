"""SessionArtifacts - session metadata for cross-session comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class FileRecord:
    """Record of a file's state in a session."""

    path: str
    hash: str
    role: str  # "read" | "modified" | "created"


@dataclass
class SessionArtifacts:
    """
    Session metadata for cross-session comparison.

    Captures what was in scope at session start for later comparison.
    The raw Claude Code transcript is the source of truth; this is
    structure extracted from it.
    """

    session_id: str
    initial_prompt: str           # Why this session started
    files: dict[str, FileRecord]  # What was in scope
    root_frame_id: str            # Entry point to call tree
    conversation_log: str         # Path to Claude Code transcript

    def save(self, base_dir: Path | None = None) -> Path:
        """
        Save session artifacts to JSON file.

        Args:
            base_dir: Optional base directory (default: ~/.claude/rlm-frames/{session_id}/)

        Returns:
            Path to the saved file
        """
        if base_dir is None:
            base_dir = Path.home() / ".claude" / "rlm-frames" / self.session_id
        else:
            base_dir = Path(base_dir) / self.session_id

        base_dir.mkdir(parents=True, exist_ok=True)
        save_path = base_dir / "artifacts.json"

        # Convert to dict with FileRecord serialization
        files_dict = {
            path: {"path": rec.path, "hash": rec.hash, "role": rec.role}
            for path, rec in self.files.items()
        }

        data = {
            "session_id": self.session_id,
            "initial_prompt": self.initial_prompt,
            "files": files_dict,
            "root_frame_id": self.root_frame_id,
            "conversation_log": self.conversation_log,
            "saved_at": datetime.now().isoformat(),
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        return save_path

    @classmethod
    def load(cls, session_id: str, base_dir: Path | None = None) -> "SessionArtifacts | None":
        """
        Load session artifacts from JSON file.

        Args:
            session_id: Session identifier to load
            base_dir: Optional base directory (default: ~/.claude/rlm-frames/)

        Returns:
            SessionArtifacts if found, None otherwise
        """
        if base_dir is None:
            base_dir = Path.home() / ".claude" / "rlm-frames"
        else:
            base_dir = Path(base_dir)

        load_path = base_dir / session_id / "artifacts.json"

        if not load_path.exists():
            return None

        with open(load_path) as f:
            data = json.load(f)

        # Reconstruct FileRecord dict
        files = {
            path: FileRecord(path=rec["path"], hash=rec["hash"], role=rec["role"])
            for path, rec in data.get("files", {}).items()
        }

        return cls(
            session_id=data["session_id"],
            initial_prompt=data.get("initial_prompt", ""),
            files=files,
            root_frame_id=data.get("root_frame_id", ""),
            conversation_log=data.get("conversation_log", ""),
        )
