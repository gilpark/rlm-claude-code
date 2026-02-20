"""Session comparison for cross-session analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frame_index import FrameIndex
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
    prior: "SessionArtifacts",
    index: "FrameIndex | None" = None
) -> SessionDiff:
    """
    Compare two sessions to detect what changed.

    Comparison logic:
    1. Compare initial_prompt → same task or new?
    2. Compare file hashes → what changed?
    3. With FrameIndex: find frames referencing changed files → invalidated
    4. With FrameIndex: find suspended frames that might be relevant

    Args:
        current: Current session artifacts
        prior: Prior session artifacts
        index: Optional FrameIndex for frame-level analysis
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

    # If no FrameIndex, return basic diff
    if index is None:
        return SessionDiff(
            same_task=same_task,
            changed_files=changed_files,
            invalidated_frame_ids=[],
            resumable_frames=[]
        )

    # With FrameIndex: find invalidated frames
    invalidated_frame_ids = _find_invalidated_frames(index, changed_files)

    # With FrameIndex: find resumable suspended frames
    resumable_frames = _find_resumable_frames(index, same_task)

    return SessionDiff(
        same_task=same_task,
        changed_files=changed_files,
        invalidated_frame_ids=invalidated_frame_ids,
        resumable_frames=resumable_frames
    )


def _find_invalidated_frames(index: "FrameIndex", changed_files: list[str]) -> list[str]:
    """Find frames whose context_slice includes changed files."""
    invalidated = []

    for frame in index.values():
        # Check if any changed file is in this frame's context
        for file_path in frame.context_slice.files:
            if file_path in changed_files:
                invalidated.append(frame.frame_id)
                break

    return invalidated


def _find_resumable_frames(index: "FrameIndex", same_task: bool) -> list[str]:
    """Find suspended frames that might be worth resuming."""
    if not same_task:
        return []  # Different task, don't resume old work

    suspended = index.get_suspended_frames()
    return [f.frame_id for f in suspended]
