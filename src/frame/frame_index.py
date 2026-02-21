"""FrameIndex - flat index for O(n) frame lookup and branch queries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .causal_frame import CausalFrame, FrameStatus

from .causal_frame import FrameStatus


@dataclass
class FrameIndex:
    """
    Flat index for O(n) frame lookup and branch queries.

    At 10-20 frames, O(n) scan is instant. No DAG structure needed.
    """

    initial_query: str = ""
    query_summary: str = ""
    commit_hash: str | None = None  # Git commit hash for change detection
    _frames: dict[str, "CausalFrame"] = field(default_factory=dict)
    _dependent_cache: dict[str, set[str]] = field(default_factory=dict)  # For B4

    def add(self, frame: "CausalFrame") -> None:
        """Add a frame to the index, update parent's children and evidence."""
        self._frames[frame.frame_id] = frame

        # Invalidate dependent cache when frames change
        self._dependent_cache.clear()

        # Update parent's children list and evidence
        if frame.parent_id and frame.parent_id in self._frames:
            parent = self._frames[frame.parent_id]

            # Add to children (defensive)
            if frame.frame_id not in parent.children:
                parent.children.append(frame.frame_id)

            # Add to evidence only if child completed successfully
            if frame.status == FrameStatus.COMPLETED:
                if frame.frame_id not in parent.evidence:
                    parent.evidence.append(frame.frame_id)

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

    def find_dependent_frames(self, frame_id: str) -> set[str]:
        """
        Find all frames that depend on a given frame.

        Dependents include:
        - Children (frames with this frame as parent_id)
        - Evidence consumers (frames citing this frame in evidence list)

        Results are cached for O(1) lookup until frames change.
        """
        # Check cache
        if frame_id in self._dependent_cache:
            return self._dependent_cache[frame_id].copy()

        dependents = set()

        for fid, frame in self._frames.items():
            # Check if child
            if frame.parent_id == frame_id:
                dependents.add(fid)

            # Check if cites as evidence
            if frame_id in frame.evidence:
                dependents.add(fid)

        # Cache result
        self._dependent_cache[frame_id] = dependents

        return dependents.copy()

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
                "canonical_task": frame.canonical_task.to_dict() if frame.canonical_task else None,
            }
            frames_data.append(frame_dict)

        data = {
            "session_id": session_id,
            "initial_query": self.initial_query,
            "query_summary": self.query_summary,
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
        from .canonical_task import CanonicalTask

        if base_dir is None:
            base_dir = Path.home() / ".claude" / "rlm-frames"

        load_path = base_dir / session_id / "index.json"

        if not load_path.exists():
            return cls()

        with open(load_path) as f:
            data = json.load(f)

        index = cls()
        index.initial_query = data.get("initial_query", "")
        index.query_summary = data.get("query_summary", "")
        index.commit_hash = data.get("commit_hash")
        for frame_dict in data.get("frames", []):
            context_slice = ContextSlice(
                files=frame_dict["context_slice"]["files"],
                memory_refs=frame_dict["context_slice"]["memory_refs"],
                tool_outputs=frame_dict["context_slice"]["tool_outputs"],
                token_budget=frame_dict["context_slice"]["token_budget"],
            )

            # Load canonical_task if present
            canonical_task = None
            if frame_dict.get("canonical_task"):
                canonical_task = CanonicalTask.from_dict(frame_dict["canonical_task"])

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
                canonical_task=canonical_task,
            )
            index.add(frame)

        return index

    @classmethod
    def load_with_validation(
        cls,
        session_id: str,
        base_dir: Path | None = None,
        current_root: Path | None = None,
    ) -> "FrameIndex":
        """
        Load frame index and validate frames against current file state.

        Uses git diff to detect changed files, then invalidates frames
        whose context_slice files have different hashes.

        Args:
            session_id: Session identifier to load
            base_dir: Optional base directory for frame files
            current_root: Current repo root (for git diff)

        Returns:
            FrameIndex with validated/invalidated frames
        """
        from .causal_frame import FrameStatus
        from .context_map import detect_changed_files
        from .frame_invalidation import propagate_invalidation

        index = cls.load(session_id, base_dir)

        if index.commit_hash and current_root:
            # Detect changed files via git diff
            changed_paths = detect_changed_files(index.commit_hash, current_root)
            changed_strs = {str(p) for p in changed_paths}

            # Check each frame for hash mismatches
            for frame in index.values():
                if frame.status != FrameStatus.COMPLETED:
                    continue

                for file_path, stored_hash in frame.context_slice.files.items():
                    if file_path in changed_strs:
                        # File changed since frame was created
                        propagate_invalidation(
                            frame.frame_id,
                            f"File changed: {file_path}",
                            index,
                        )
                        break

        return index
