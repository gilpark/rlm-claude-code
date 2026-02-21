"""
Unit tests for CFConfig class.

Tests CausalFrame user configuration loading, saving, and defaults.
"""

import json
import pytest
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import CFConfig, DEFAULT_CF_CONFIG, CONFIG_PATH


class TestCFConfigDefaults:
    """Tests for default configuration values."""

    def test_default_values_match_spec(self):
        """Default values match the specification."""
        config = CFConfig()

        assert config.default_max_depth == 3
        assert config.default_verbose is False
        assert config.status_limit == 5
        assert config.default_model == "sonnet"
        assert config.status_icons is True
        assert config.reference_dirs == []
        assert config.auto_resume_on_invalidate is False

    def test_default_cf_config_dict(self):
        """DEFAULT_CF_CONFIG has all required fields."""
        assert "default_max_depth" in DEFAULT_CF_CONFIG
        assert "default_verbose" in DEFAULT_CF_CONFIG
        assert "status_limit" in DEFAULT_CF_CONFIG
        assert "default_model" in DEFAULT_CF_CONFIG
        assert "status_icons" in DEFAULT_CF_CONFIG
        assert "reference_dirs" in DEFAULT_CF_CONFIG
        assert "auto_resume_on_invalidate" in DEFAULT_CF_CONFIG

    def test_config_path_is_correct(self):
        """CONFIG_PATH points to the correct location."""
        assert CONFIG_PATH == Path.home() / ".claude" / "causalframe-config.json"


class TestCFConfigLoad:
    """Tests for CFConfig.load() method."""

    def test_load_without_file_returns_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Loading without a config file returns default values."""
        # Use a temporary path that doesn't exist
        nonexistent_path = tmp_path / "nonexistent-config.json"

        config = CFConfig.load(path=nonexistent_path)

        assert config.default_max_depth == 3
        assert config.default_verbose is False
        assert config.status_limit == 5
        assert config.default_model == "sonnet"
        assert config.status_icons is True
        assert config.reference_dirs == []
        assert config.auto_resume_on_invalidate is False

    def test_load_with_valid_config(self, tmp_path: Path):
        """Loading a valid config file overrides defaults."""
        config_path = tmp_path / "causalframe-config.json"
        user_config = {
            "default_max_depth": 5,
            "default_verbose": True,
            "status_limit": 10,
            "default_model": "opus",
            "status_icons": False,
            "reference_dirs": ["/path/to/ref1", "/path/to/ref2"],
            "auto_resume_on_invalidate": True,
        }
        config_path.write_text(json.dumps(user_config))

        config = CFConfig.load(path=config_path)

        assert config.default_max_depth == 5
        assert config.default_verbose is True
        assert config.status_limit == 10
        assert config.default_model == "opus"
        assert config.status_icons is False
        assert config.reference_dirs == ["/path/to/ref1", "/path/to/ref2"]
        assert config.auto_resume_on_invalidate is True

    def test_load_partial_config_uses_defaults(self, tmp_path: Path):
        """Partial config file uses defaults for unspecified fields."""
        config_path = tmp_path / "causalframe-config.json"
        user_config = {
            "default_max_depth": 7,
            "default_verbose": True,
        }
        config_path.write_text(json.dumps(user_config))

        config = CFConfig.load(path=config_path)

        # Specified values
        assert config.default_max_depth == 7
        assert config.default_verbose is True

        # Default values for unspecified fields
        assert config.status_limit == 5
        assert config.default_model == "sonnet"
        assert config.status_icons is True
        assert config.reference_dirs == []
        assert config.auto_resume_on_invalidate is False

    def test_load_with_invalid_json_returns_defaults(self, tmp_path: Path):
        """Invalid JSON in config file falls back to defaults."""
        config_path = tmp_path / "causalframe-config.json"
        config_path.write_text("{ invalid json }")

        config = CFConfig.load(path=config_path)

        assert config.default_max_depth == 3
        assert config.default_verbose is False
        assert config.status_limit == 5
        assert config.default_model == "sonnet"

    def test_load_filters_unknown_fields(self, tmp_path: Path):
        """Unknown fields in config are filtered out."""
        config_path = tmp_path / "causalframe-config.json"
        user_config = {
            "default_max_depth": 4,
            "unknown_field": "should_be_ignored",
            "another_unknown": 123,
        }
        config_path.write_text(json.dumps(user_config))

        config = CFConfig.load(path=config_path)

        assert config.default_max_depth == 4
        # Unknown fields should not cause errors

    def test_load_with_reference_dirs(self, tmp_path: Path):
        """Can load reference_dirs list from config."""
        config_path = tmp_path / "causalframe-config.json"
        user_config = {
            "reference_dirs": [
                "/Users/test/project1",
                "/Users/test/project2",
                "~/Documents/references",
            ],
        }
        config_path.write_text(json.dumps(user_config))

        config = CFConfig.load(path=config_path)

        assert len(config.reference_dirs) == 3
        assert config.reference_dirs[0] == "/Users/test/project1"
        assert config.reference_dirs[1] == "/Users/test/project2"
        assert config.reference_dirs[2] == "~/Documents/references"

    def test_load_with_empty_reference_dirs(self, tmp_path: Path):
        """Empty reference_dirs list is handled correctly."""
        config_path = tmp_path / "causalframe-config.json"
        user_config = {"reference_dirs": []}
        config_path.write_text(json.dumps(user_config))

        config = CFConfig.load(path=config_path)

        assert config.reference_dirs == []


class TestCFConfigSave:
    """Tests for CFConfig.save() method."""

    def test_save_creates_parent_directories(self, tmp_path: Path):
        """Saving creates parent directories if they don't exist."""
        config_path = tmp_path / "nested" / "dirs" / "causalframe-config.json"

        config = CFConfig(default_max_depth=10, default_verbose=True)
        config.save(path=config_path)

        assert config_path.exists()
        assert config_path.parent.is_dir()

    def test_save_writes_all_fields(self, tmp_path: Path):
        """Saving writes all configuration fields to file."""
        config_path = tmp_path / "causalframe-config.json"

        config = CFConfig(
            default_max_depth=8,
            default_verbose=True,
            status_limit=15,
            default_model="haiku",
            status_icons=False,
            reference_dirs=["/ref1", "/ref2"],
            auto_resume_on_invalidate=True,
        )
        config.save(path=config_path)

        with open(config_path) as f:
            saved_data = json.load(f)

        assert saved_data["default_max_depth"] == 8
        assert saved_data["default_verbose"] is True
        assert saved_data["status_limit"] == 15
        assert saved_data["default_model"] == "haiku"
        assert saved_data["status_icons"] is False
        assert saved_data["reference_dirs"] == ["/ref1", "/ref2"]
        assert saved_data["auto_resume_on_invalidate"] is True

    def test_save_preserves_default_values(self, tmp_path: Path):
        """Saving a default config writes all default values."""
        config_path = tmp_path / "causalframe-config.json"

        config = CFConfig()
        config.save(path=config_path)

        with open(config_path) as f:
            saved_data = json.load(f)

        assert saved_data["default_max_depth"] == 3
        assert saved_data["default_verbose"] is False
        assert saved_data["status_limit"] == 5
        assert saved_data["default_model"] == "sonnet"
        assert saved_data["status_icons"] is True
        assert saved_data["reference_dirs"] == []
        assert saved_data["auto_resume_on_invalidate"] is False

    def test_save_overwrites_existing_file(self, tmp_path: Path):
        """Saving overwrites existing config file."""
        config_path = tmp_path / "causalframe-config.json"

        # Write initial config
        config1 = CFConfig(default_max_depth=5)
        config1.save(path=config_path)

        # Overwrite with different config
        config2 = CFConfig(default_max_depth=12, default_verbose=True)
        config2.save(path=config_path)

        with open(config_path) as f:
            saved_data = json.load(f)

        assert saved_data["default_max_depth"] == 12
        assert saved_data["default_verbose"] is True


class TestCFConfigRoundTrip:
    """Tests for save/load round-trip consistency."""

    def test_round_trip_preserves_values(self, tmp_path: Path):
        """Values are preserved through save/load cycle."""
        config_path = tmp_path / "causalframe-config.json"

        original = CFConfig(
            default_max_depth=6,
            default_verbose=True,
            status_limit=20,
            default_model="opus",
            status_icons=False,
            reference_dirs=["/a", "/b", "/c"],
            auto_resume_on_invalidate=True,
        )
        original.save(path=config_path)

        loaded = CFConfig.load(path=config_path)

        assert loaded.default_max_depth == original.default_max_depth
        assert loaded.default_verbose == original.default_verbose
        assert loaded.status_limit == original.status_limit
        assert loaded.default_model == original.default_model
        assert loaded.status_icons == original.status_icons
        assert loaded.reference_dirs == original.reference_dirs
        assert loaded.auto_resume_on_invalidate == original.auto_resume_on_invalidate

    def test_round_trip_with_defaults(self, tmp_path: Path):
        """Default values round-trip correctly."""
        config_path = tmp_path / "causalframe-config.json"

        original = CFConfig()
        original.save(path=config_path)

        loaded = CFConfig.load(path=config_path)

        assert loaded.default_max_depth == original.default_max_depth
        assert loaded.default_verbose == original.default_verbose
        assert loaded.status_limit == original.status_limit
        assert loaded.default_model == original.default_model
        assert loaded.status_icons == original.status_icons
        assert loaded.reference_dirs == original.reference_dirs
        assert loaded.auto_resume_on_invalidate == original.auto_resume_on_invalidate


class TestCFConfigFields:
    """Tests for individual field behavior."""

    def test_default_max_depth_accepts_various_ints(self):
        """default_max_depth accepts various integer values."""
        config1 = CFConfig(default_max_depth=1)
        config2 = CFConfig(default_max_depth=10)
        config3 = CFConfig(default_max_depth=100)

        assert config1.default_max_depth == 1
        assert config2.default_max_depth == 10
        assert config3.default_max_depth == 100

    def test_default_model_accepts_various_strings(self):
        """default_model accepts various model names."""
        config1 = CFConfig(default_model="sonnet")
        config2 = CFConfig(default_model="opus")
        config3 = CFConfig(default_model="haiku")

        assert config1.default_model == "sonnet"
        assert config2.default_model == "opus"
        assert config3.default_model == "haiku"

    def test_reference_dirs_accepts_various_lists(self):
        """reference_dirs accepts various list values."""
        config1 = CFConfig(reference_dirs=[])
        config2 = CFConfig(reference_dirs=["/single/path"])
        config3 = CFConfig(reference_dirs=["/path1", "/path2", "/path3"])

        assert config1.reference_dirs == []
        assert len(config2.reference_dirs) == 1
        assert len(config3.reference_dirs) == 3
