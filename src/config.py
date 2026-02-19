"""
Configuration management for RLM-Claude-Code.

Implements: Spec §5.3 Router Configuration
"""

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal


def _filter_dataclass_fields(data: dict[str, Any], cls: type) -> dict[str, Any]:
    """Filter dict to only include fields that exist in the dataclass."""
    valid_fields = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid_fields}


def _get_use_rlm_core() -> bool:
    """
    Determine whether to use rlm-core Rust bindings.

    Priority order:
    1. RLM_USE_CORE environment variable (if set)
    2. use_rlm_core setting in ~/.claude/rlm-config.json
    3. Default: False

    Returns:
        True if rlm-core should be used, False otherwise.
    """
    # Check environment variable first (highest priority)
    env_value = os.getenv("RLM_USE_CORE")
    if env_value is not None:
        return env_value.lower() == "true"

    # Check config file
    config_path = Path.home() / ".claude" / "rlm-config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
            return data.get("use_rlm_core", False)
        except (json.JSONDecodeError, OSError):
            pass

    return False


# Feature flag for rlm-core migration
# Set RLM_USE_CORE=true env var or use_rlm_core=true in config to enable
USE_RLM_CORE = _get_use_rlm_core()

@dataclass
class ActivationConfig:
    """
    Configuration for RLM activation.

    Implements: SPEC-14.10-14.15 for always-on micro mode.

    Modes:
    - "micro": Always-on with minimal cost, starts at micro level (SPEC-14.12 default)
    - "complexity": Original heuristic-based activation
    - "always": Always full RLM
    - "manual": Only when explicitly enabled
    - "token": Activate above token threshold
    """

    mode: Literal["micro", "complexity", "always", "manual", "token"] = "micro"
    fallback_token_threshold: int = 80000
    complexity_score_threshold: int = 2
    # SPEC-14.30: Fast-path bypass configuration
    fast_path_enabled: bool = True
    # SPEC-14.20: Escalation configuration
    escalation_enabled: bool = True
    # SPEC-14.62: Session token budget
    session_budget_tokens: int = 500_000


@dataclass
class DepthConfig:
    """Configuration for recursive depth."""

    default: int = 2
    max: int = 3
    spawn_repl_at_depth_1: bool = True


@dataclass
class HybridConfig:
    """Configuration for hybrid mode."""

    enabled: bool = True
    simple_query_bypass: bool = True
    simple_confidence_threshold: float = 0.95


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory output."""

    verbosity: Literal["minimal", "normal", "verbose", "debug"] = "normal"
    streaming: bool = True
    colors: bool = True
    export_enabled: bool = True
    export_path: str = "~/.claude/rlm-trajectories/"


@dataclass
class ModelConfig:
    """Configuration for model selection by depth."""

    root_model: str = "opus"  # Default to Opus
    recursive_depth_1: str = "sonnet"
    recursive_depth_2: str = "haiku"
    # Alternative models for OpenAI routing
    openai_root: str = "gpt-5.2-codex"
    openai_recursive: str = "gpt-4o-mini"


@dataclass
class CostConfig:
    """Configuration for cost controls."""

    max_recursive_calls_per_turn: int = 10
    max_tokens_per_recursive_call: int = 8000
    abort_on_cost_threshold: int = 50000  # tokens


@dataclass
class RLMConfig:
    """
    Complete RLM configuration.

    Implements: Spec §5.3 Router Configuration
    """

    activation: ActivationConfig = field(default_factory=ActivationConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    cost_controls: CostConfig = field(default_factory=CostConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "RLMConfig":
        """Load configuration from file."""
        if path is None:
            path = Path.home() / ".claude" / "rlm-config.json"

        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        # Backward compatibility: migrate old field names
        models_data = data.get("models", {})
        if "root" in models_data and "root_model" not in models_data:
            models_data["root_model"] = models_data.pop("root")

        return cls(
            activation=ActivationConfig(**_filter_dataclass_fields(data.get("activation", {}), ActivationConfig)),
            depth=DepthConfig(**_filter_dataclass_fields(data.get("depth", {}), DepthConfig)),
            hybrid=HybridConfig(**_filter_dataclass_fields(data.get("hybrid", {}), HybridConfig)),
            trajectory=TrajectoryConfig(**_filter_dataclass_fields(data.get("trajectory", {}), TrajectoryConfig)),
            models=ModelConfig(**_filter_dataclass_fields(models_data, ModelConfig)),
            cost_controls=CostConfig(**_filter_dataclass_fields(data.get("cost_controls", {}), CostConfig)),
        )

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = Path.home() / ".claude" / "rlm-config.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                {
                    "activation": self.activation.__dict__,
                    "depth": self.depth.__dict__,
                    "hybrid": self.hybrid.__dict__,
                    "trajectory": self.trajectory.__dict__,
                    "models": self.models.__dict__,
                    "cost_controls": self.cost_controls.__dict__,
                },
                f,
                indent=2,
            )


# Default configuration instance
default_config = RLMConfig()
