"""
Session schema models for RLM-Claude-Code.

Implements: Session State Consolidation Plan Phase 1

Pydantic models for the unified session.json schema.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .causal_frame import CausalFrame


class ActivationMode(str, Enum):
    """RLM activation mode."""

    COMPLEXITY = "complexity"
    MANUAL = "manual"
    ALWAYS = "always"
    NEVER = "never"


class SessionMetadata(BaseModel):
    """Session metadata for session.json."""

    session_id: str
    created_at: float
    updated_at: float
    ended_at: float | None = None
    cwd: str
    claude_transcript_path: str | None = None
    rlm_version: str = "0.1.0"
    parent_session_id: str | None = None
    session_type: str = "default"  # default, migrated, fork, checkpoint
    tags: list[str] = Field(default_factory=list)
    description: str | None = None


class SessionActivation(BaseModel):
    """RLM activation state for session.json."""

    rlm_active: bool = False
    activation_mode: ActivationMode = ActivationMode.COMPLEXITY
    activation_reason: str | None = None
    complexity_score: float | None = None
    current_depth: int = 0
    max_depth: int = 2


class SessionBudget(BaseModel):
    """Token and cost budget tracking for session.json."""

    total_tokens_used: int = 0
    total_recursive_calls: int = 0
    max_recursive_calls: int = 10
    cost_usd: float = 0.0
    by_model: dict[str, dict[str, float]] = Field(default_factory=dict)


class FileInfo(BaseModel):
    """Information about a file in session context."""

    hash: str | None = None
    size_bytes: int | None = None
    first_access: float | None = None
    last_access: float | None = None


class ToolOutputEntry(BaseModel):
    """A tool output entry in session context."""

    tool_name: str
    content_preview: str = Field(default="", max_length=1000)
    exit_code: int | None = None
    timestamp: float
    cell_id: str | None = None


class SessionContextData(BaseModel):
    """
    Context data for session.json.

    Note: Conversation history is NOT stored here - it's in the symlinked transcript.
    """

    files: dict[str, FileInfo] = Field(default_factory=dict)
    tool_outputs: list[ToolOutputEntry] = Field(default_factory=list, max_length=100)
    working_memory: dict[str, Any] = Field(default_factory=dict)
    causal_frames: list[Any] = Field(default_factory=list)  # CausalFrame objects (dataclass)
    frame_transitions: list[dict[str, Any]] = Field(default_factory=list)


class CellIndexInfo(BaseModel):
    """Cell index information for session.json."""

    count: int = 0
    index_path: str = "cells/index.json"


class TrajectoryInfo(BaseModel):
    """Trajectory information for session.json."""

    events_count: int = 0
    export_path: str | None = None


class SessionState(BaseModel):
    """
    Unified session state for session.json.

    Implements: Session State Consolidation Plan Phase 1

    This is the main schema for the session-centric state architecture.
    Conversation history is NOT stored here - it's in the symlinked transcript.
    """

    metadata: SessionMetadata
    activation: SessionActivation = Field(default_factory=SessionActivation)
    context: SessionContextData = Field(default_factory=SessionContextData)
    budget: SessionBudget = Field(default_factory=SessionBudget)
    cells: CellIndexInfo = Field(default_factory=CellIndexInfo)
    trajectory: TrajectoryInfo = Field(default_factory=TrajectoryInfo)

    model_config = {
        "use_enum_values": True,
        "extra": "forbid",
    }


# Cell schema for Phase 2


class CellType(str, Enum):
    """Type of cell."""

    REPL = "repl"
    TOOL = "tool"
    LLM_CALL = "llm_call"
    MAP_REDUCE = "map_reduce"
    VERIFICATION = "verification"


class CellInput(BaseModel):
    """Input for a cell."""

    source: str  # "repl", "tool", "llm_call", etc.
    operation: str
    args: dict[str, Any] = Field(default_factory=dict)
    context_snapshot: dict[str, Any] | None = None


class CellOutput(BaseModel):
    """Output from a cell."""

    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    cached_response: str | None = None


class CellMetadata(BaseModel):
    """Metadata for a cell."""

    depth: int = 0
    model: str | None = None
    tokens_used: int = 0
    temperature: float | None = None
    promoted_to_db: bool = False


class CellTrace(BaseModel):
    """Decision trace for a cell."""

    goal: str | None = None
    decision: str | None = None
    rationale: str | None = None


class Cell(BaseModel):
    """
    A replayable computation unit.

    Implements: Session State Consolidation Plan Phase 2
    """

    cell_id: str = Field(pattern=r"^cell_[a-z0-9]{8}$")
    created_at: float
    type: CellType
    input: CellInput
    output: CellOutput = Field(default_factory=CellOutput)
    dependencies: list[str] = Field(default_factory=list)
    dependents: list[str] = Field(default_factory=list)
    metadata: CellMetadata = Field(default_factory=CellMetadata)
    trace: CellTrace | None = None

    model_config = {
        "use_enum_values": True,
    }


class CellIndexEntry(BaseModel):
    """Entry in the cell DAG index."""

    type: CellType
    dependencies: list[str] = Field(default_factory=list)
    dependents: list[str] = Field(default_factory=list)

    model_config = {
        "use_enum_values": True,
    }


class CellIndex(BaseModel):
    """
    Cell DAG index for cells/index.json.

    Implements: Session State Consolidation Plan Phase 2
    """

    version: str = "1.0"
    session_id: str
    cells: dict[str, CellIndexEntry] = Field(default_factory=dict)
    roots: list[str] = Field(default_factory=list)
    leaves: list[str] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    cycles_detected: list[str] | None = None


__all__ = [
    "ActivationMode",
    "SessionMetadata",
    "SessionActivation",
    "SessionBudget",
    "FileInfo",
    "ToolOutputEntry",
    "SessionContextData",
    "CellIndexInfo",
    "TrajectoryInfo",
    "SessionState",
    "CellType",
    "CellInput",
    "CellOutput",
    "CellMetadata",
    "CellTrace",
    "Cell",
    "CellIndexEntry",
    "CellIndex",
]
