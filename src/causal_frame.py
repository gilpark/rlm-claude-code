"""CausalFrame - tree-structured frame for RLM execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context_slice import ContextSlice


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


@dataclass
class CausalFrame:
    """
    A single frame in the RLM execution tree.

    Design decisions:
    - Core records data (confidence, status)
    - Root LM decides policy (escalation, context slice)
    - frame_id is deterministic for caching
    """

    # Identity — DETERMINISTIC
    frame_id: str                 # hash(parent_id + query + context_slice.hash())
    depth: int                    # 0 = root
    parent_id: str | None
    children: list[str]

    # Reasoning
    query: str
    context_slice: "ContextSlice"
    evidence: list[str]           # Frame IDs + raw observations
    conclusion: str | None
    confidence: float             # 0.0-1.0, Core records, Root LM decides policy
    invalidation_condition: str   # What would make this wrong

    # Branch + propagation control
    status: FrameStatus
    branched_from: str | None     # If pivot: which frame this branched from
    escalation_reason: str | None # Why this frame was escalated/suspended

    created_at: datetime
    completed_at: datetime | None


def compute_frame_id(
    parent_id: str | None,
    query: str,
    context_slice: "ContextSlice"
) -> str:
    """
    Compute deterministic frame ID for caching and deduplication.

    Same (parent, query, context) = same frame_id = cache hit.
    """
    content = f"{parent_id}:{query}:{context_slice.hash()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
