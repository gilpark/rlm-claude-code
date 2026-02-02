package events

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestEmitAndReadLatest(t *testing.T) {
	// Use temp dir for events
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	// Ensure events dir exists
	evDir := filepath.Join(tmpDir, ".claude", "events")
	os.MkdirAll(evDir, 0755)

	event := map[string]any{
		"type":   "test_event",
		"source": "test",
		"value":  42,
	}

	err := Emit(event, "test-source")
	if err != nil {
		t.Fatalf("Emit failed: %v", err)
	}

	// Verify latest file
	latest, err := ReadLatest("test-source")
	if err != nil {
		t.Fatalf("ReadLatest failed: %v", err)
	}

	if latest["type"] != "test_event" {
		t.Errorf("type = %v, want test_event", latest["type"])
	}

	// JSON numbers decode as float64
	if latest["value"] != float64(42) {
		t.Errorf("value = %v, want 42", latest["value"])
	}

	// Verify log file
	logFile := filepath.Join(evDir, "test-source-events.jsonl")
	data, err := os.ReadFile(logFile)
	if err != nil {
		t.Fatalf("Failed to read log: %v", err)
	}

	var logged map[string]any
	if err := json.Unmarshal(data, &logged); err != nil {
		t.Fatalf("Failed to parse log entry: %v", err)
	}
	if logged["type"] != "test_event" {
		t.Errorf("logged type = %v, want test_event", logged["type"])
	}
}

func TestReadLatestMissing(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)
	os.MkdirAll(filepath.Join(tmpDir, ".claude", "events"), 0755)

	_, err := ReadLatest("nonexistent")
	if err == nil {
		t.Error("Expected error for missing source")
	}
}

func TestGetDPPhaseUnknown(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)
	os.MkdirAll(filepath.Join(tmpDir, ".claude", "events"), 0755)

	phase := GetDPPhase()
	if phase != "unknown" {
		t.Errorf("GetDPPhase = %q, want %q", phase, "unknown")
	}
}

func TestGetDPPhaseFromEvent(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)
	evDir := filepath.Join(tmpDir, ".claude", "events")
	os.MkdirAll(evDir, 0755)

	event := map[string]any{
		"type":     "phase_transition",
		"to_phase": "spec",
	}
	Emit(event, "disciplined-process")

	phase := GetDPPhase()
	if phase != "spec" {
		t.Errorf("GetDPPhase = %q, want %q", phase, "spec")
	}
}

func TestGetRLMModeUnknown(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)
	os.MkdirAll(filepath.Join(tmpDir, ".claude", "events"), 0755)

	mode := GetRLMMode()
	if mode != "unknown" {
		t.Errorf("GetRLMMode = %q, want %q", mode, "unknown")
	}
}

func TestSuggestedRLMModeMapping(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)
	evDir := filepath.Join(tmpDir, ".claude", "events")
	os.MkdirAll(evDir, 0755)

	tests := []struct {
		phase    string
		wantMode string
	}{
		{"spec", "thorough"},
		{"review", "thorough"},
		{"test", "balanced"},
		{"implement", "balanced"},
		{"orient", "balanced"},
	}

	for _, tt := range tests {
		Emit(map[string]any{
			"type":     "phase_transition",
			"to_phase": tt.phase,
		}, "disciplined-process")

		got := SuggestedRLMMode()
		if got != tt.wantMode {
			t.Errorf("SuggestedRLMMode() for phase %q = %q, want %q", tt.phase, got, tt.wantMode)
		}
	}
}

func TestEmitMultipleAppends(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)
	evDir := filepath.Join(tmpDir, ".claude", "events")
	os.MkdirAll(evDir, 0755)

	Emit(map[string]any{"type": "first"}, "multi-test")
	Emit(map[string]any{"type": "second"}, "multi-test")

	// Latest should be the second event
	latest, _ := ReadLatest("multi-test")
	if latest["type"] != "second" {
		t.Errorf("latest type = %v, want second", latest["type"])
	}

	// Log should have both entries
	logFile := filepath.Join(evDir, "multi-test-events.jsonl")
	data, _ := os.ReadFile(logFile)
	lines := 0
	for _, b := range data {
		if b == '\n' {
			lines++
		}
	}
	if lines != 2 {
		t.Errorf("log has %d lines, want 2", lines)
	}
}
