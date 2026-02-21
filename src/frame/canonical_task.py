"""Canonical task representation for frame deduplication."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.llm_client import LLMClient


@dataclass(frozen=True)
class CanonicalTask:
    """
    Normalized representation of a task intent.

    Used for frame identity to prevent duplicate frames from similar queries.
    Frame identity = hash(canonical_task + context_slice.hash())

    Examples:
        "Analyze the auth module" -> CanonicalTask("analyze", ["**/*auth*"])
        "Debug login error" -> CanonicalTask("debug", ["**/*login*"])
        "Summarize API" -> CanonicalTask("summarize", ["**/*api*"])
    """

    task_type: str  # "analyze", "debug", "summarize", "implement", "verify", "explain"
    target: str | list[str]  # "auth.py" or ["src/auth/*"] or ["**/*auth*"]
    analysis_scope: str | None = None  # "correctness", "architecture", "security"
    params: dict = field(default_factory=dict)

    def to_hash(self) -> str:
        """Generate stable hash for this canonical task."""
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.blake2b(data.encode(), digest_size=8).hexdigest()

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CanonicalTask":
        """Deserialize from dict."""
        return cls(
            task_type=data.get("task_type", "unknown"),
            target=data.get("target", "unknown"),
            analysis_scope=data.get("analysis_scope"),
            params=data.get("params", {}),
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        scope = f" [{self.analysis_scope}]" if self.analysis_scope else ""
        target = self.target if isinstance(self.target, str) else ", ".join(self.target[:3])
        return f"{self.task_type}({target}){scope}"
