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
from .rlaph_loop import RLAPHLoop, RLPALoopResult
from .repl_environment import RLMEnvironment
from .llm_client import LLMClient, LLMError
from .tool_bridge import ToolBridge, ToolPermissions, ToolResult

# Causal Layer
from .causal_frame import CausalFrame, FrameStatus, compute_frame_id
from .context_slice import ContextSlice
from .frame_index import FrameIndex
from .frame_invalidation import propagate_invalidation
from .frame_store import FrameStore
from .session_artifacts import SessionArtifacts, FileRecord
from .session_comparison import SessionDiff, compare_sessions
from .plugin_interface import CoreContext, RLMPlugin, PluginError

# Types & Config
from .types import SessionContext
from .config import RLMConfig, default_config
from .prompts import build_rlm_system_prompt

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
    # Causal
    "CausalFrame",
    "FrameStatus",
    "compute_frame_id",
    "ContextSlice",
    "FrameIndex",
    "propagate_invalidation",
    "FrameStore",
    "SessionArtifacts",
    "FileRecord",
    "SessionDiff",
    "compare_sessions",
    "CoreContext",
    "RLMPlugin",
    "PluginError",
    # Types & Config
    "SessionContext",
    "RLMConfig",
    "default_config",
    "build_rlm_system_prompt",
]
