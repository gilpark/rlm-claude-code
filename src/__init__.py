"""
RLM-Claude-Code: Recursive Language Model integration for Claude Code.

Transform Claude Code into an RLM agent for unbounded context handling
and improved reasoning via REPL-based context decomposition.

Implements intelligent orchestration with:
- Automatic complexity-based RLM activation
- Claude-powered orchestration decisions
- Strategy learning from successful trajectories
- User-configurable preferences and budgets
"""

__version__ = "0.2.0"

# Core orchestration
from .orchestrator import RLMOrchestrator
from .intelligent_orchestrator import IntelligentOrchestrator
from .auto_activation import AutoActivator, check_auto_activation

# Context and REPL
from .context_manager import externalize_context
from .repl_environment import RLMEnvironment
from .types import SessionContext

# Complexity and routing
from .complexity_classifier import should_activate_rlm
from .orchestration_schema import ExecutionMode, OrchestrationPlan, ToolAccessLevel
from .local_orchestrator import LocalOrchestrator, LocalModelConfig, RECOMMENDED_CONFIGS
from .orchestration_logger import (
    OrchestrationLogger,
    LoggerConfig,
    OrchestrationDecisionLog,
    TrainingDataExporter,
    get_logger,
    set_logger,
)

# Trajectory and analysis
from .trajectory import TrajectoryEvent, TrajectoryRenderer
from .trajectory_analysis import TrajectoryAnalyzer, StrategyType

# User preferences and tools
from .user_preferences import UserPreferences, PreferencesManager
from .tool_bridge import ToolBridge, ToolPermissions
from .strategy_cache import StrategyCache

# Memory system (SPEC-02, SPEC-03)
from .memory_store import MemoryStore, Node, Hyperedge
from .memory_evolution import MemoryEvolution, ConsolidationResult, PromotionResult, DecayResult

# Reasoning traces (SPEC-04)
from .reasoning_traces import ReasoningTraces, DecisionNode, DecisionTree

# Enhanced budget tracking (SPEC-05)
from .enhanced_budget import EnhancedBudgetTracker, EnhancedBudgetMetrics, BudgetLimits, BudgetAlert

__all__ = [
    # Core
    "RLMOrchestrator",
    "IntelligentOrchestrator",
    "AutoActivator",
    "check_auto_activation",
    # Context
    "SessionContext",
    "externalize_context",
    "RLMEnvironment",
    # Complexity
    "should_activate_rlm",
    "ExecutionMode",
    "OrchestrationPlan",
    "ToolAccessLevel",
    # Local orchestration
    "LocalOrchestrator",
    "LocalModelConfig",
    "RECOMMENDED_CONFIGS",
    # Orchestration logging
    "OrchestrationLogger",
    "LoggerConfig",
    "OrchestrationDecisionLog",
    "TrainingDataExporter",
    "get_logger",
    "set_logger",
    # Trajectory
    "TrajectoryEvent",
    "TrajectoryRenderer",
    "TrajectoryAnalyzer",
    "StrategyType",
    # Preferences and tools
    "UserPreferences",
    "PreferencesManager",
    "ToolBridge",
    "ToolPermissions",
    "StrategyCache",
    # Memory system (SPEC-02, SPEC-03)
    "MemoryStore",
    "Node",
    "Hyperedge",
    "MemoryEvolution",
    "ConsolidationResult",
    "PromotionResult",
    "DecayResult",
    # Reasoning traces (SPEC-04)
    "ReasoningTraces",
    "DecisionNode",
    "DecisionTree",
    # Enhanced budget (SPEC-05)
    "EnhancedBudgetTracker",
    "EnhancedBudgetMetrics",
    "BudgetLimits",
    "BudgetAlert",
]
