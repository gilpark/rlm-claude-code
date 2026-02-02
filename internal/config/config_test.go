package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func setupTestDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "rlm-config.json")
	configPathOverride = path
	t.Cleanup(func() { configPathOverride = "" })
	return path
}

func TestDefaultV2(t *testing.T) {
	c := DefaultV2()
	if c.Version != "2.0" {
		t.Errorf("expected version 2.0, got %s", c.Version)
	}
	if c.Activation.Mode != "auto" {
		t.Errorf("expected mode auto, got %s", c.Activation.Mode)
	}
	if c.Activation.FallbackThreshold != 80000 {
		t.Errorf("expected threshold 80000, got %d", c.Activation.FallbackThreshold)
	}
	if !c.Activation.AutoCrossFile {
		t.Error("expected AutoCrossFile true")
	}
	if !c.Activation.DPPhaseAware {
		t.Error("expected DPPhaseAware true")
	}
	if c.Budget.MaxCostPerQuery != 1.00 {
		t.Errorf("expected max cost 1.00, got %f", c.Budget.MaxCostPerQuery)
	}
	if !c.Events.EmitTrajectory {
		t.Error("expected EmitTrajectory true")
	}
}

func TestLoadMissing(t *testing.T) {
	setupTestDir(t)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Version != "2.0" {
		t.Errorf("expected version 2.0, got %s", cfg.Version)
	}
	if cfg.Activation.Mode != "auto" {
		t.Errorf("expected default mode, got %s", cfg.Activation.Mode)
	}
}

func TestLoadV2(t *testing.T) {
	path := setupTestDir(t)

	v2 := DefaultV2()
	v2.Activation.Mode = "manual"
	v2.Budget.DailyLimit = 50.0
	data, _ := json.MarshalIndent(v2, "", "  ")
	os.WriteFile(path, data, 0644)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Version != "2.0" {
		t.Errorf("expected version 2.0, got %s", cfg.Version)
	}
	if cfg.Activation.Mode != "manual" {
		t.Errorf("expected mode manual, got %s", cfg.Activation.Mode)
	}
	if cfg.Budget.DailyLimit != 50.0 {
		t.Errorf("expected daily limit 50.0, got %f", cfg.Budget.DailyLimit)
	}
}

func TestMigrateV1(t *testing.T) {
	path := setupTestDir(t)

	v1 := ConfigV1{}
	v1.Version = "1.0"
	v1.Activation.Mode = "manual"
	v1.Activation.FallbackThreshold = 60000
	v1.Models.Depth0 = "claude-3-opus"
	v1.Models.Depth1 = "claude-3-sonnet"
	v1.Models.Depth2 = "claude-3-haiku"
	v1.Budget.MaxCostPerQuery = 0.50
	v1.Budget.DailyLimit = 10.0
	data, _ := json.MarshalIndent(v1, "", "  ")
	os.WriteFile(path, data, 0644)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Version upgraded
	if cfg.Version != "2.0" {
		t.Errorf("expected version 2.0, got %s", cfg.Version)
	}

	// User values preserved
	if cfg.Activation.Mode != "manual" {
		t.Errorf("expected mode manual, got %s", cfg.Activation.Mode)
	}
	if cfg.Activation.FallbackThreshold != 60000 {
		t.Errorf("expected threshold 60000, got %d", cfg.Activation.FallbackThreshold)
	}
	if cfg.Models.Depth0 != "claude-3-opus" {
		t.Errorf("expected depth0 claude-3-opus, got %s", cfg.Models.Depth0)
	}
	if cfg.Budget.MaxCostPerQuery != 0.50 {
		t.Errorf("expected max cost 0.50, got %f", cfg.Budget.MaxCostPerQuery)
	}

	// New V2 fields get defaults
	if !cfg.Activation.AutoCrossFile {
		t.Error("expected AutoCrossFile default true after migration")
	}
	if !cfg.Events.EmitTrajectory {
		t.Error("expected EmitTrajectory default true after migration")
	}

	// Backup created
	backupPath := path + ".v1.bak"
	if _, err := os.Stat(backupPath); os.IsNotExist(err) {
		t.Error("expected backup file to be created")
	}

	// Migrated config saved to disk
	diskData, _ := os.ReadFile(path)
	var diskCfg ConfigV2
	json.Unmarshal(diskData, &diskCfg)
	if diskCfg.Version != "2.0" {
		t.Errorf("expected saved config version 2.0, got %s", diskCfg.Version)
	}
}

func TestMigrateUnversioned(t *testing.T) {
	path := setupTestDir(t)

	// Config without version field
	raw := `{
  "activation": {
    "mode": "always",
    "fallback_token_threshold": 50000
  },
  "models": {
    "depth_0": "claude-3-opus",
    "depth_1": "claude-3-sonnet",
    "depth_2": "claude-3-haiku"
  },
  "budget": {
    "max_cost_per_query": 0.75,
    "daily_limit": 15.0
  }
}`
	os.WriteFile(path, []byte(raw), 0644)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.Version != "2.0" {
		t.Errorf("expected version 2.0, got %s", cfg.Version)
	}
	if cfg.Activation.Mode != "always" {
		t.Errorf("expected mode always, got %s", cfg.Activation.Mode)
	}
	if cfg.Activation.FallbackThreshold != 50000 {
		t.Errorf("expected threshold 50000, got %d", cfg.Activation.FallbackThreshold)
	}

	// Backup created
	backupPath := path + ".v1.bak"
	if _, err := os.Stat(backupPath); os.IsNotExist(err) {
		t.Error("expected backup file for unversioned config")
	}
}
