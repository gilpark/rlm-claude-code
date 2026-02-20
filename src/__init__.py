"""CausalFrame v2: REPL + CausalFrame persistence.

Causal awareness for Claude Code - externalize reasoning, evolve with
your environment through REPL-based context decomposition.

v2 Architecture:
- REPL: Spatial externalization within session
- CausalFrame: Temporal persistence across sessions

Based on: Zhang et al., "Recursive Language Models" (2025)
"""

__version__ = "0.0.1"

# REPL Layer
from .repl import (
    LLMClient,
    LLMError,
    RLMEnvironment,
    RLAPHLoop,
    RLPALoopResult,
    ToolBridge,
    ToolPermissions,
    ToolResult,
)

# Frame Layer
from .frame import (
    CausalFrame,
    ContextSlice,
    FrameIndex,
    FrameStore,
    FrameStatus,
    compute_frame_id,
    propagate_invalidation,
)

# Session Layer
from .session import FileRecord, SessionArtifacts, SessionDiff, compare_sessions

# Plugin
from .plugin_interface import CoreContext, PluginError, RLMPlugin

# Types & Config
from .config import RLMConfig, default_config
from .repl.prompts import build_rlm_system_prompt
from .types import SessionContext

__all__ = [
    # REPL
    "RLAPHLoop",
    "RLPALoopResult",
    "RLMEnvironment",
    "LLMClient",
    "LLMError",
    "ToolBridge",
    "ToolPermissions",
    "ToolResult",
    # Frame
    "CausalFrame",
    "FrameStatus",
    "compute_frame_id",
    "ContextSlice",
    "FrameIndex",
    "propagate_invalidation",
    "FrameStore",
    # Session
    "SessionArtifacts",
    "FileRecord",
    "SessionDiff",
    "compare_sessions",
    # Plugin
    "CoreContext",
    "RLMPlugin",
    "PluginError",
    # Types & Config
    "SessionContext",
    "RLMConfig",
    "default_config",
    "build_rlm_system_prompt",
]

