"""ContextSlice - partitioned context per frame."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass
class ContextSlice:
    """
    A frame's view of context — not the whole, just its slice.

    Core enforces token_budget. Root LM chooses what files/references to include
    via tool calls (peek, grep, spawn_child).
    """

    files: dict[str, str]         # file_path → content_hash
    memory_refs: list[str]        # memory entry IDs
    tool_outputs: dict[str, str]  # tool_name → output_hash
    token_budget: int

    def hash(self) -> str:
        """
        Deterministic hash for cache key.

        Note: token_budget is NOT included in hash since it's a constraint,
        not part of content identity.
        """
        content = json.dumps({
            "files": sorted(self.files.items()),
            "memory_refs": sorted(self.memory_refs),
            "tool_outputs": sorted(self.tool_outputs.items()),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
