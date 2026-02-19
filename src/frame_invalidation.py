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
    - UP + SIDEWAYS: to frames using this as evidence (scan)

    At 10-20 frames, O(n) scan is instant. No DAG structure needed.
    """
    from .causal_frame import FrameStatus

    invalidated = set()

    def _invalidate(fid: str):
        if fid in invalidated:
            return
        invalidated.add(fid)

        frame = index.get(fid)
        if frame is None:
            return

        frame.status = FrameStatus.INVALIDATED
        frame.escalation_reason = reason

        # CASCADE DOWN to children
        for child_id in frame.children:
            _invalidate(child_id)

        # SCAN for UP + SIDEWAYS (frames using this as evidence)
        for other_id, other in index.items():
            if fid in other.evidence:
                _invalidate(other_id)

    _invalidate(frame_id)
    return list(invalidated)
