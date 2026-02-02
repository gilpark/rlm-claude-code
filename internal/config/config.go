package config

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// ConfigV1 is the original config format (no version field or version "1.0").
type ConfigV1 struct {
	Version    string `json:"version,omitempty"`
	Activation struct {
		Mode              string `json:"mode"`
		FallbackThreshold int    `json:"fallback_token_threshold"`
	} `json:"activation"`
	Models struct {
		Depth0 string `json:"depth_0"`
		Depth1 string `json:"depth_1"`
		Depth2 string `json:"depth_2"`
	} `json:"models"`
	Budget struct {
		MaxCostPerQuery float64 `json:"max_cost_per_query"`
		DailyLimit      float64 `json:"daily_limit"`
	} `json:"budget"`
}

// ConfigV2 adds new fields for cross-plugin coordination.
type ConfigV2 struct {
	Version    string `json:"version"`
	Activation struct {
		Mode              string `json:"mode"`
		FallbackThreshold int    `json:"fallback_token_threshold"`
		AutoCrossFile     bool   `json:"auto_cross_file"`
		DPPhaseAware      bool   `json:"dp_phase_aware"`
	} `json:"activation"`
	Models struct {
		Depth0 string `json:"depth_0"`
		Depth1 string `json:"depth_1"`
		Depth2 string `json:"depth_2"`
	} `json:"models"`
	Budget struct {
		MaxCostPerQuery float64 `json:"max_cost_per_query"`
		DailyLimit      float64 `json:"daily_limit"`
	} `json:"budget"`
	Events struct {
		EmitTrajectory bool `json:"emit_trajectory"`
		EmitModeChange bool `json:"emit_mode_change"`
	} `json:"events"`
}

// DefaultV2 returns a new V2 config with sensible defaults.
func DefaultV2() *ConfigV2 {
	c := &ConfigV2{Version: "2.0"}
	c.Activation.Mode = "auto"
	c.Activation.FallbackThreshold = 80000
	c.Activation.AutoCrossFile = true
	c.Activation.DPPhaseAware = true
	c.Models.Depth0 = "claude-opus-4-5"
	c.Models.Depth1 = "claude-sonnet-4"
	c.Models.Depth2 = "claude-haiku-4-5"
	c.Budget.MaxCostPerQuery = 1.00
	c.Budget.DailyLimit = 25.00
	c.Events.EmitTrajectory = true
	c.Events.EmitModeChange = true
	return c
}

// configPathOverride allows tests to redirect config I/O.
var configPathOverride string

// ConfigPath returns the path to ~/.claude/rlm-config.json.
func ConfigPath() string {
	if configPathOverride != "" {
		return configPathOverride
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".claude", "rlm-config.json")
}

// Load reads the config, migrating from V1 if needed.
// If no config exists, returns DefaultV2().
func Load() (*ConfigV2, error) {
	path := ConfigPath()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return DefaultV2(), nil
		}
		return nil, err
	}

	// Check version
	var base struct {
		Version string `json:"version"`
	}
	json.Unmarshal(data, &base)

	switch base.Version {
	case "2.0":
		var v2 ConfigV2
		if err := json.Unmarshal(data, &v2); err != nil {
			return nil, err
		}
		return &v2, nil
	default:
		// V1 or unversioned — migrate
		return migrateV1(data, path)
	}
}

func migrateV1(data []byte, configPath string) (*ConfigV2, error) {
	var v1 ConfigV1
	if err := json.Unmarshal(data, &v1); err != nil {
		return nil, err
	}

	// Backup old config
	backupPath := configPath + ".v1.bak"
	os.WriteFile(backupPath, data, 0644)

	// Migrate preserving user values
	v2 := DefaultV2()
	if v1.Activation.Mode != "" {
		v2.Activation.Mode = v1.Activation.Mode
	}
	if v1.Activation.FallbackThreshold > 0 {
		v2.Activation.FallbackThreshold = v1.Activation.FallbackThreshold
	}
	if v1.Models.Depth0 != "" {
		v2.Models.Depth0 = v1.Models.Depth0
	}
	if v1.Models.Depth1 != "" {
		v2.Models.Depth1 = v1.Models.Depth1
	}
	if v1.Models.Depth2 != "" {
		v2.Models.Depth2 = v1.Models.Depth2
	}
	if v1.Budget.MaxCostPerQuery > 0 {
		v2.Budget.MaxCostPerQuery = v1.Budget.MaxCostPerQuery
	}
	if v1.Budget.DailyLimit > 0 {
		v2.Budget.DailyLimit = v1.Budget.DailyLimit
	}

	// Save migrated config
	if err := Save(v2); err != nil {
		return nil, err
	}

	return v2, nil
}

// Save writes the config to disk.
func Save(c *ConfigV2) error {
	path := ConfigPath()
	os.MkdirAll(filepath.Dir(path), 0755)
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}
