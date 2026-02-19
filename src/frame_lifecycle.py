"""FrameLifecycle - manage frame state transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .causal_frame import CausalFrame, FrameStatus


class FrameLifecycle:
    """
    Manage frame state transitions.

    Valid transitions follow the frame lifecycle:
    - CREATED → RUNNING (start execution)
    - RUNNING → COMPLETED | INVALIDATED | SUSPENDED
    - COMPLETED → VERIFIED | INVALIDATED
    - VERIFIED → PROMOTED | INVALIDATED | UNCERTAIN
    - SUSPENDED → RUNNING | INVALIDATED (resume or discard)
    - UNCERTAIN → VERIFIED | INVALIDATED
    - PROMOTED → (terminal)
    - INVALIDATED → (terminal)
    """

    VALID_TRANSITIONS: dict["FrameStatus", set["FrameStatus"]]

    def __init__(self):
        from .causal_frame import FrameStatus
        self.VALID_TRANSITIONS = {
            FrameStatus.CREATED: {FrameStatus.RUNNING},
            FrameStatus.RUNNING: {
                FrameStatus.COMPLETED,
                FrameStatus.INVALIDATED,
                FrameStatus.SUSPENDED,
            },
            FrameStatus.COMPLETED: {FrameStatus.VERIFIED, FrameStatus.INVALIDATED},
            FrameStatus.VERIFIED: {
                FrameStatus.PROMOTED,
                FrameStatus.INVALIDATED,
                FrameStatus.UNCERTAIN,
            },
            FrameStatus.SUSPENDED: {FrameStatus.RUNNING, FrameStatus.INVALIDATED},
            FrameStatus.UNCERTAIN: {FrameStatus.VERIFIED, FrameStatus.INVALIDATED},
            FrameStatus.PROMOTED: set(),  # Terminal
            FrameStatus.INVALIDATED: set(),  # Terminal
        }

    def can_transition(
        self, from_status: "FrameStatus", to_status: "FrameStatus"
    ) -> bool:
        """Check if a transition is valid."""
        return to_status in self.VALID_TRANSITIONS.get(from_status, set())

    def transition(self, frame: "CausalFrame", to_status: "FrameStatus") -> bool:
        """
        Attempt to transition a frame to a new status.

        Returns True if successful, False if transition is invalid.
        """
        if not self.can_transition(frame.status, to_status):
            return False
        frame.status = to_status
        return True
