"""FrameIndex - flat index for O(n) frame lookup and branch queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .causal_frame import CausalFrame, FrameStatus


class FrameIndex:
    """
    Flat index for O(n) frame lookup and branch queries.

    At 10-20 frames, O(n) scan is instant. No DAG structure needed.
    """

    def __init__(self):
        self._frames: dict[str, "CausalFrame"] = {}

    def add(self, frame: "CausalFrame") -> None:
        """Add a frame to the index."""
        self._frames[frame.frame_id] = frame

    def get(self, frame_id: str) -> "CausalFrame | None":
        """Get a frame by ID, or None if not found."""
        return self._frames.get(frame_id)

    def get_active_frames(self) -> list["CausalFrame"]:
        """Get all frames with RUNNING status."""
        from .causal_frame import FrameStatus
        return [f for f in self._frames.values() if f.status == FrameStatus.RUNNING]

    def get_suspended_frames(self) -> list["CausalFrame"]:
        """Get all frames with SUSPENDED status."""
        from .causal_frame import FrameStatus
        return [f for f in self._frames.values() if f.status == FrameStatus.SUSPENDED]

    def get_pivots(self) -> list["CausalFrame"]:
        """Get all frames that branched from another frame."""
        return [f for f in self._frames.values() if f.branched_from is not None]

    def __contains__(self, frame_id: str) -> bool:
        return frame_id in self._frames

    def __len__(self) -> int:
        return len(self._frames)

    def items(self) -> list[tuple[str, "CausalFrame"]]:
        """Return all (frame_id, frame) pairs."""
        return list(self._frames.items())

    def values(self) -> list["CausalFrame"]:
        """Return all frames."""
        return list(self._frames.values())
