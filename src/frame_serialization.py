"""Frame serialization to JSON format."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .causal_frame import CausalFrame


def serialize_frame(frame: "CausalFrame") -> dict[str, Any]:
    """
    Serialize a CausalFrame to a JSON-compatible dict.

    Format: JSON with ISO 8601 timestamps.
    """
    return {
        "frame_id": frame.frame_id,
        "depth": frame.depth,
        "parent_id": frame.parent_id,
        "children": frame.children,
        "query": frame.query,
        "context_slice": {
            "files": frame.context_slice.files,
            "memory_refs": frame.context_slice.memory_refs,
            "tool_outputs": frame.context_slice.tool_outputs,
            "token_budget": frame.context_slice.token_budget,
        },
        "evidence": frame.evidence,
        "conclusion": frame.conclusion,
        "confidence": frame.confidence,
        "invalidation_condition": frame.invalidation_condition,
        "status": frame.status.value,
        "branched_from": frame.branched_from,
        "escalation_reason": frame.escalation_reason,
        "created_at": frame.created_at.isoformat(),
        "completed_at": frame.completed_at.isoformat() if frame.completed_at else None,
    }


def deserialize_frame(data: dict[str, Any]) -> "CausalFrame":
    """
    Deserialize a dict back to a CausalFrame.

    Inverse of serialize_frame.
    """
    from .causal_frame import CausalFrame, FrameStatus
    from .context_slice import ContextSlice

    return CausalFrame(
        frame_id=data["frame_id"],
        depth=data["depth"],
        parent_id=data["parent_id"],
        children=data["children"],
        query=data["query"],
        context_slice=ContextSlice(
            files=data["context_slice"]["files"],
            memory_refs=data["context_slice"]["memory_refs"],
            tool_outputs=data["context_slice"]["tool_outputs"],
            token_budget=data["context_slice"]["token_budget"],
        ),
        evidence=data["evidence"],
        conclusion=data["conclusion"],
        confidence=data["confidence"],
        invalidation_condition=data["invalidation_condition"],
        status=FrameStatus(data["status"]),
        branched_from=data.get("branched_from"),
        escalation_reason=data.get("escalation_reason"),
        created_at=datetime.fromisoformat(data["created_at"]),
        completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
    )
