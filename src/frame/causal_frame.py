"""CausalFrame - tree-structured frame for RLM execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context_slice import ContextSlice
    from .canonical_task import CanonicalTask


class FrameStatus(Enum):
    """Lifecycle states for a CausalFrame."""

    RUNNING = "running"          # Frame is actively executing
    COMPLETED = "completed"      # Frame finished execution
    SUSPENDED = "suspended"      # Pivot — preserved, not deleted
    INVALIDATED = "invalidated"  # Cascade invalidation
    PROMOTED = "promoted"        # Persisted as long-term knowledge


@dataclass
class CausalFrame:
    """
    A single frame in the RLM execution tree.

    Design decisions:
    - Core records data (confidence, status)
    - Root LM decides policy (escalation, context slice)
    - frame_id is deterministic for caching
    """

    # Identity — DETERMINISTIC (no defaults)
    frame_id: str                 # hash(parent_id + query + context_slice.hash())
    depth: int                    # 0 = root
    parent_id: str | None
    children: list[str]

    # Reasoning (no defaults)
    query: str
    context_slice: "ContextSlice"
    evidence: list[str]           # Frame IDs + raw observations
    conclusion: str | None
    confidence: float             # 0.0-1.0, Core records, Root LM decides policy
    invalidation_condition: dict  # Structured: {"files": [...], "tools": [...], "memory_refs": [...], "description": "..."}

    # Branch + propagation control
    status: FrameStatus
    created_at: datetime

    # Optional fields with defaults
    canonical_task: "CanonicalTask | None" = None  # Normalized intent for frame deduplication
    branched_from: str | None = None     # If pivot: which frame this branched from
    escalation_reason: str | None = None # Why this frame was escalated/suspended
    completed_at: datetime | None = None


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


def generate_invalidation_condition(context_slice: "ContextSlice") -> dict:
    """
    Generate structured invalidation condition from context_slice.

    Returns a dict that can be programmatically checked, not just a string.

    Structure:
    {
        "files": ["/path/to/file1.py", ...],
        "tools": ["Read", "Glob", ...],
        "memory_refs": ["ref1", ...],
        "description": "Human-readable summary"
    }
    """
    files = list(context_slice.files.keys()) if context_slice.files else []
    tools = list(context_slice.tool_outputs.keys()) if context_slice.tool_outputs else []
    memory_refs = list(context_slice.memory_refs) if context_slice.memory_refs else []
    description = _generate_description(context_slice)

    return {
        "files": files,
        "tools": tools,
        "memory_refs": memory_refs,
        "description": description,
    }


def _generate_description(context_slice: "ContextSlice") -> str:
    """Generate human-readable description for debugging."""
    parts = []

    if context_slice.files:
        filenames = [Path(p).name for p in context_slice.files.keys()]
        if len(filenames) == 1:
            parts.append(f"{filenames[0]} changes or is deleted")
        else:
            shown = filenames[:3]
            more = f" (+{len(filenames) - 3} more)" if len(filenames) > 3 else ""
            parts.append(f"any of {len(filenames)} files ({', '.join(shown)}{more}) change")

    if context_slice.tool_outputs:
        tool_names = list(context_slice.tool_outputs.keys())
        parts.append(f"tool results from {', '.join(tool_names)} change")

    if context_slice.memory_refs:
        parts.append("memory entries change")

    if not parts:
        return "No automatic invalidation condition"

    return "; or ".join(parts)
