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

    # Primary models (GLM via Anthropic-compatible API)
    root_model: str = "glm-4.6"  # Best accuracy + speed (100%, 3.1s avg)
    recursive_depth_1: str = "glm-4.6"  # Same as root for consistency
    recursive_depth_2: str = "glm-4.7"  # Cheaper for deep recursion
    recursive_depth_3: str = "glm-4.7"  # Cheapest for deepest

    # Temperature for REPL (low = deterministic, high = creative)
    # 0.0-0.3 = deterministic, 0.4-0.7 = balanced, 0.8-1.0 = creative
    temperature: float = 0.1  # Low for accurate REPL output

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


# -----------------------------------------------------------------------------
# CausalFrame Configuration
# -----------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".claude" / "causalframe-config.json"

DEFAULT_CF_CONFIG = {
    "default_max_depth": 3,
    "default_verbose": False,
    "status_limit": 5,
    "default_model": "sonnet",
    "status_icons": True,
    "reference_dirs": [],
    "auto_resume_on_invalidate": False,
}


@dataclass
class CFConfig:
    """
    CausalFrame user configuration.

    Provides UX settings for CausalFrame operations like depth limits,
    verbosity, status display, and auto-resume behavior.

    Separate from RLMConfig which handles model/routing settings.
    """

    default_max_depth: int = 3
    default_verbose: bool = False
    status_limit: int = 5
    default_model: str = "sonnet"
    status_icons: bool = True
    reference_dirs: list[str] = field(default_factory=list)
    auto_resume_on_invalidate: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> "CFConfig":
        """
        Load config from file with defaults.

        Args:
            path: Optional config file path. Defaults to ~/.claude/causalframe-config.json

        Returns:
            CFConfig instance with user settings merged with defaults
        """
        if path is None:
            path = CONFIG_PATH

        config = DEFAULT_CF_CONFIG.copy()

        if path.exists():
            try:
                user_config = json.loads(path.read_text())
                config.update(user_config)
            except (json.JSONDecodeError, IOError):
                # Use defaults on error
                pass

        # Filter to only valid fields for the dataclass
        filtered = _filter_dataclass_fields(config, cls)

        return cls(**filtered)

    def save(self, path: Path | None = None) -> None:
        """
        Save configuration to file.

        Args:
            path: Optional config file path. Defaults to ~/.claude/causalframe-config.json
        """
        if path is None:
            path = CONFIG_PATH

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(
                {
                    "default_max_depth": self.default_max_depth,
                    "default_verbose": self.default_verbose,
                    "status_limit": self.status_limit,
                    "default_model": self.default_model,
                    "status_icons": self.status_icons,
                    "reference_dirs": self.reference_dirs,
                    "auto_resume_on_invalidate": self.auto_resume_on_invalidate,
                },
                f,
                indent=2,
            )


# Default CausalFrame configuration instance
default_cf_config = CFConfig()
