"""Plugin interface for extensible RLM reasoning."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from .causal_frame import CausalFrame
    from .session_artifacts import SessionArtifacts


class PluginError(Exception):
    """Plugin operation failed."""

    recoverable: bool = False
    frame_id: str | None = None
    reason: str | None = None


class CoreContext(TypedDict):
    """
    Context provided to plugins — read-only view of current state.

    Core provides data. Plugin decides what to do with it.
    """

    current_frame: "CausalFrame | None"
    index: dict[str, "CausalFrame"]
    artifacts: "SessionArtifacts | None"
    changed_files: list[str]
    invalidated_frames: list[str]
    suspended_frames: list[str]
    confidence_threshold: float


class RLMPlugin(Protocol):
    """
    Plugin interface for extensible RLM reasoning.

    Design principle: Core owns causality, plugins observe.
    Plugins cannot modify the call tree.
    """

    def transform_input(
        self,
        raw_input: Any,
        ctx: CoreContext
    ) -> dict | PluginError:
        """
        What does the model see?

        Returns: context_slice for this frame or PluginError
        """
        ...

    def parse_output(
        self,
        raw_output: str,
        frame: "CausalFrame"
    ) -> Any | PluginError:
        """
        What do we keep from the model's response?

        Returns: structured data — must preserve invalidation_condition
        """
        ...

    def store(self, parsed: Any, frame: "CausalFrame") -> None:
        """
        Where does it go?

        Side effect only. No decisions here — those belong in parse_output.
        """
        ...
