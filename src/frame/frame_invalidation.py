"""Frame invalidation with cascade propagation."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .frame_index import FrameIndex


def propagate_invalidation(
    frame_id: str,
    reason: str,
    index: "FrameIndex"
) -> list[str]:
    """
    Invalidate a frame and all its dependents.

    Propagation direction:
    - DOWN: to all children (tree walk)
    - SIDEWAYS: to frames using this as evidence

    Returns list of all invalidated frame IDs.
    """
    from .causal_frame import FrameStatus

    invalidated = set()

    def _invalidate(fid: str, current_reason: str):
        # Cycle detection
        if fid in invalidated:
            return
        invalidated.add(fid)

        frame = index.get(fid)
        if frame is None:
            return

        # Update status and reason
        frame.status = FrameStatus.INVALIDATED
        frame.escalation_reason = current_reason

        # CASCADE DOWN to children
        for child_id in frame.children:
            _invalidate(child_id, f"Parent invalidated: {reason}")

        # CASCADE SIDEWAYS to evidence consumers
        dependents = index.find_dependent_frames(fid)
        for dep_id in dependents:
            _invalidate(dep_id, f"Evidence invalidated: {fid}")

    _invalidate(frame_id, reason)
    return list(invalidated)
