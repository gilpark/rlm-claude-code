"""SessionArtifacts - session metadata for cross-session comparison."""

from __future__ import annotations

from dataclasses import dataclass


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
