"""Default RLM plugin for standard reasoning."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..causal_frame import CausalFrame
    from ..plugin_interface import CoreContext


class DefaultRLMPlugin:
    """
    Default RLM plugin for standard reasoning.

    This plugin provides basic transform_input, parse_output, and store
    implementations. It can be extended or replaced for custom behavior.
    """

    def transform_input(
        self,
        raw_input: Any,
        ctx: "CoreContext"
    ) -> dict:
        """
        Transform input for the model.

        Default: pass through query with context info.
        """
        result = dict(raw_input) if isinstance(raw_input, dict) else {"input": raw_input}

        # Add suspended frame info if available
        if ctx.get("suspended_frames"):
            result["suspended_frames"] = ctx["suspended_frames"]

        return result

    def parse_output(
        self,
        raw_output: str,
        frame: "CausalFrame"
    ) -> Any:
        """
        Parse output from the model.

        Default: extract conclusion from output.
        """
        return {
            "raw_output": raw_output,
            "conclusion": raw_output,
            "confidence": frame.confidence,
        }

    def store(self, parsed: Any, frame: "CausalFrame") -> None:
        """
        Store parsed output.

        Default: update frame conclusion.
        Side effect only.
        """
        if isinstance(parsed, dict) and "conclusion" in parsed:
            frame.conclusion = parsed["conclusion"]
