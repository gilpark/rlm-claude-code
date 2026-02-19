"""Session comparison for cross-session analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session_artifacts import SessionArtifacts


@dataclass
class SessionDiff:
    """Result of comparing two sessions."""

    same_task: bool
    changed_files: list[str]
    invalidated_frame_ids: list[str]
    resumable_frames: list[str]  # suspended frames that might be relevant


def compare_sessions(
    current: "SessionArtifacts",
    prior: "SessionArtifacts"
) -> SessionDiff:
    """
    Compare two sessions to detect what changed.

    Comparison logic:
    1. Compare initial_prompt → same task or new?
    2. Compare file hashes → what changed?
    3. (Future) Find frames referencing changed files → invalidated

    Note: invalidated_frame_ids and resumable_frames are populated
    when a FrameIndex is provided (future enhancement).
    """
    # Check if same task (same initial prompt)
    same_task = current.initial_prompt == prior.initial_prompt

    # Find changed files
    changed_files = []

    # New files
    for path in current.files:
        if path not in prior.files:
            changed_files.append(path)

    # Modified files (same path, different hash)
    for path, current_record in current.files.items():
        if path in prior.files:
            prior_record = prior.files[path]
            if current_record.hash != prior_record.hash:
                changed_files.append(path)

    return SessionDiff(
        same_task=same_task,
        changed_files=changed_files,
        invalidated_frame_ids=[],  # Populated with FrameIndex
        resumable_frames=[]
    )
