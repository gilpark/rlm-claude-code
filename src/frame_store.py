"""FrameStore - JSONL persistence for CausalFrames.

Design doc reference: src/frame_store.py in target architecture.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .causal_frame import CausalFrame, FrameStatus


class FrameStore:
    """
    JSONL-based persistence for CausalFrames.

    One file per session: ~/.claude/rlm-frames/{root_session_id}.jsonl
    Zero dependencies. Human-readable. Append-only.

    Each line is a JSON object representing one CausalFrame.
    """

    def __init__(self, path: Path | str):
        """
        Initialize FrameStore.

        Args:
            path: Path to the JSONL file for this session
        """
        self.path = Path(path)
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, frame: "CausalFrame") -> None:
        """
        Save a frame to the store.

        Appends to the JSONL file. Does not check for duplicates.

        Args:
            frame: CausalFrame to persist
        """
        data = self._serialize(frame)
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def load(self, frame_id: str) -> "CausalFrame | None":
        """
        Load a frame by ID.

        Scans the file linearly. At 10-20 frames, this is instant.

        Args:
            frame_id: ID of the frame to load

        Returns:
            CausalFrame if found, None otherwise
        """
        if not self.path.exists():
            return None

        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("frame_id") == frame_id:
                    return self._deserialize(data)

        return None

    def list(self) -> list["CausalFrame"]:
        """
        List all frames in the store.

        Returns:
            List of all CausalFrames
        """
        if not self.path.exists():
            return []

        frames = []
        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                frames.append(self._deserialize(data))

        return frames

    def find_by_status(self, status: "FrameStatus") -> list["CausalFrame"]:
        """
        Find all frames with a given status.

        Args:
            status: FrameStatus to filter by

        Returns:
            List of matching CausalFrames
        """
        from .causal_frame import FrameStatus

        frames = self.list()
        return [f for f in frames if f.status == status]

    def _serialize(self, frame: "CausalFrame") -> dict:
        """Serialize CausalFrame to JSON-compatible dict."""
        data = {
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
            "created_at": frame.created_at.isoformat() if frame.created_at else None,
            "completed_at": frame.completed_at.isoformat() if frame.completed_at else None,
        }
        return data

    def _deserialize(self, data: dict) -> "CausalFrame":
        """Deserialize dict to CausalFrame."""
        from .causal_frame import CausalFrame, FrameStatus
        from .context_slice import ContextSlice

        context_slice = ContextSlice(
            files=data["context_slice"]["files"],
            memory_refs=data["context_slice"]["memory_refs"],
            tool_outputs=data["context_slice"]["tool_outputs"],
            token_budget=data["context_slice"]["token_budget"],
        )

        return CausalFrame(
            frame_id=data["frame_id"],
            depth=data["depth"],
            parent_id=data["parent_id"],
            children=data["children"],
            query=data["query"],
            context_slice=context_slice,
            evidence=data["evidence"],
            conclusion=data["conclusion"],
            confidence=data["confidence"],
            invalidation_condition=data["invalidation_condition"],
            status=FrameStatus(data["status"]),
            branched_from=data.get("branched_from"),
            escalation_reason=data.get("escalation_reason"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )


__all__ = ["FrameStore"]
