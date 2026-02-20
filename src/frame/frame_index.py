"""FrameIndex - flat index for O(n) frame lookup and branch queries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
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
        self.commit_hash: str | None = None  # Git commit hash for change detection

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

    def find_by_parent(self, parent_id: str) -> list["CausalFrame"]:
        """Find all frames with a given parent."""
        return [f for f in self._frames.values() if f.parent_id == parent_id]

    def find_promoted(self) -> list["CausalFrame"]:
        """Find all PROMOTED frames (persisted facts)."""
        from .causal_frame import FrameStatus
        return [f for f in self._frames.values() if f.status == FrameStatus.PROMOTED]

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

    def save(self, session_id: str, base_dir: Path | None = None) -> Path:
        """
        Save frame index to JSON file.

        Args:
            session_id: Session identifier for the file name
            base_dir: Optional base directory (default: ~/.claude/rlm-frames/)

        Returns:
            Path to the saved file
        """
        if base_dir is None:
            base_dir = Path.home() / ".claude" / "rlm-frames" / session_id
        else:
            base_dir = base_dir / session_id

        base_dir.mkdir(parents=True, exist_ok=True)
        save_path = base_dir / "index.json"

        # Serialize frames
        frames_data = []
        for frame in self._frames.values():
            frame_dict = {
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
            frames_data.append(frame_dict)

        data = {
            "session_id": session_id,
            "commit_hash": self.commit_hash,
            "frames": frames_data,
            "saved_at": datetime.now().isoformat(),
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        return save_path

    @classmethod
    def load(cls, session_id: str, base_dir: Path | None = None) -> "FrameIndex":
        """
        Load frame index from JSON file.

        Args:
            session_id: Session identifier to load
            base_dir: Optional base directory (default: ~/.claude/rlm-frames/)

        Returns:
            FrameIndex with loaded frames (empty if file doesn't exist)
        """
        from .causal_frame import CausalFrame, FrameStatus
        from .context_slice import ContextSlice

        if base_dir is None:
            base_dir = Path.home() / ".claude" / "rlm-frames"

        load_path = base_dir / session_id / "index.json"

        if not load_path.exists():
            return cls()

        with open(load_path) as f:
            data = json.load(f)

        index = cls()
        index.commit_hash = data.get("commit_hash")
        for frame_dict in data.get("frames", []):
            context_slice = ContextSlice(
                files=frame_dict["context_slice"]["files"],
                memory_refs=frame_dict["context_slice"]["memory_refs"],
                tool_outputs=frame_dict["context_slice"]["tool_outputs"],
                token_budget=frame_dict["context_slice"]["token_budget"],
            )

            frame = CausalFrame(
                frame_id=frame_dict["frame_id"],
                depth=frame_dict["depth"],
                parent_id=frame_dict["parent_id"],
                children=frame_dict["children"],
                query=frame_dict["query"],
                context_slice=context_slice,
                evidence=frame_dict["evidence"],
                conclusion=frame_dict["conclusion"],
                confidence=frame_dict["confidence"],
                invalidation_condition=frame_dict["invalidation_condition"],
                status=FrameStatus(frame_dict["status"]),
                branched_from=frame_dict.get("branched_from"),
                escalation_reason=frame_dict.get("escalation_reason"),
                created_at=datetime.fromisoformat(frame_dict["created_at"]) if frame_dict.get("created_at") else None,
                completed_at=datetime.fromisoformat(frame_dict["completed_at"]) if frame_dict.get("completed_at") else None,
            )
            index.add(frame)

        return index
