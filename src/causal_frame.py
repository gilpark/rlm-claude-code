"""CausalFrame - tree-structured frame for RLM execution."""

from __future__ import annotations

from enum import Enum


class FrameStatus(Enum):
    """Lifecycle states for a CausalFrame."""

    CREATED = "created"        # Frame created, not yet running
    RUNNING = "running"        # Frame is actively executing
    COMPLETED = "completed"    # Frame finished execution
    VERIFIED = "verified"      # Frame output verified
    PROMOTED = "promoted"      # Promoted to memory_store
    INVALIDATED = "invalidated"  # Cascade invalidation
    SUSPENDED = "suspended"    # Low confidence / needs human
    UNCERTAIN = "uncertain"    # Propagated uncertainty
