package classify

import (
	"testing"
)

func TestIsFastPath(t *testing.T) {
	tests := []struct {
		prompt string
		want   bool
	}{
		{"git status", true},
		{"yes", true},
		{"ok", true},
		{"no", true},
		{"run pytest", true},
		{"show main.go", true},
		{"", true},
		{"Refactor the auth system across all handlers", false},
		{"Debug why the API returns 500 errors", false},
		{"find all places where we access the database", false},
	}

	for _, tt := range tests {
		got := IsFastPath(tt.prompt)
		if got != tt.want {
			t.Errorf("IsFastPath(%q) = %v, want %v", tt.prompt, got, tt.want)
		}
	}
}

func TestExtractSignals_MultipleFiles(t *testing.T) {
	s := ExtractSignals("Update main.go and utils.py to fix the bug")
	if !s.ReferencesMultipleFiles {
		t.Error("expected ReferencesMultipleFiles=true for prompt with two file extensions")
	}
}

func TestExtractSignals_ModulePair(t *testing.T) {
	s := ExtractSignals("coordinate auth and api modules")
	if !s.ReferencesMultipleFiles {
		t.Error("expected ReferencesMultipleFiles=true for module pair")
	}
	if !s.FilesSpanMultipleModules {
		t.Error("expected FilesSpanMultipleModules=true for module pair")
	}
}

func TestExtractSignals_CrossContext(t *testing.T) {
	s := ExtractSignals("why does this fail when we pass nil?")
	if !s.RequiresCrossContextReasoning {
		t.Error("expected RequiresCrossContextReasoning=true")
	}
}

func TestExtractSignals_Debugging(t *testing.T) {
	s := ExtractSignals("debug the error in the handler")
	if !s.DebuggingTask {
		t.Error("expected DebuggingTask=true")
	}
}

func TestExtractSignals_ExhaustiveSearch(t *testing.T) {
	s := ExtractSignals("find all instances of the deprecated function")
	if !s.RequiresExhaustiveSearch {
		t.Error("expected RequiresExhaustiveSearch=true")
	}
}

func TestExtractSignals_Security(t *testing.T) {
	s := ExtractSignals("check for SQL injection vulnerabilities")
	if !s.SecurityReviewTask {
		t.Error("expected SecurityReviewTask=true")
	}
}

func TestExtractSignals_Architecture(t *testing.T) {
	s := ExtractSignals("how does the authentication system work?")
	if !s.ArchitectureAnalysis {
		t.Error("expected ArchitectureAnalysis=true")
	}
}

func TestExtractSignals_Refactor(t *testing.T) {
	s := ExtractSignals("Refactor the auth system across all handlers to use JWT")
	if !s.ArchitectureAnalysis {
		t.Error("expected ArchitectureAnalysis=true for refactor")
	}
}

func TestExtractSignals_Thorough(t *testing.T) {
	s := ExtractSignals("make sure all tests pass before committing")
	if !s.UserWantsThorough {
		t.Error("expected UserWantsThorough=true")
	}
}

func TestExtractSignals_Fast(t *testing.T) {
	s := ExtractSignals("just show me the file")
	if !s.UserWantsFast {
		t.Error("expected UserWantsFast=true")
	}
}

func TestExtractSignals_Continuation(t *testing.T) {
	s := ExtractSignals("also fix the tests while you're at it")
	if !s.TaskIsContinuation {
		t.Error("expected TaskIsContinuation=true")
	}
}

func TestShouldActivate_Simple(t *testing.T) {
	activate, reason, mode := ShouldActivate("yes", "", "")
	if activate {
		t.Errorf("expected no activation for 'yes', got reason=%s mode=%s", reason, mode)
	}
}

func TestShouldActivate_FastIntent(t *testing.T) {
	activate, reason, _ := ShouldActivate("just show me the file contents", "", "")
	if activate {
		t.Errorf("expected no activation for fast intent, got reason=%s", reason)
	}
	if reason != "fast_intent" {
		t.Errorf("reason = %q, want fast_intent", reason)
	}
}

func TestShouldActivate_Debugging(t *testing.T) {
	activate, reason, _ := ShouldActivate("Debug why the API returns 500 errors on POST", "", "")
	if !activate {
		t.Error("expected activation for debugging prompt")
	}
	if reason != "debugging_task" {
		t.Errorf("reason = %q, want debugging_task", reason)
	}
}

func TestShouldActivate_Security(t *testing.T) {
	activate, reason, _ := ShouldActivate("Find all places where we access the database without auth", "", "")
	if !activate {
		t.Error("expected activation for security prompt")
	}
	// Could match exhaustive_search or security_review depending on order
	if reason != "exhaustive_search" && reason != "security_review" {
		t.Errorf("reason = %q, want exhaustive_search or security_review", reason)
	}
}

func TestShouldActivate_Architecture(t *testing.T) {
	activate, reason, _ := ShouldActivate("Refactor the auth system across all handlers to use JWT", "", "")
	if !activate {
		t.Error("expected activation for architecture prompt")
	}
	// refactor matches architecture_analysis
	if reason != "architecture_analysis" && reason != "exhaustive_search" && reason != "security_review" {
		t.Errorf("unexpected reason = %q", reason)
	}
}

func TestShouldActivate_MultiModule(t *testing.T) {
	activate, reason, _ := ShouldActivate("Update main.go and utils.py with the new config", "", "")
	if !activate {
		t.Error("expected activation for multi-module prompt")
	}
	if reason != "multi_module_task" {
		t.Errorf("reason = %q, want multi_module_task", reason)
	}
}

func TestSuggestMode_DPPhases(t *testing.T) {
	tests := []struct {
		phase string
		want  string
	}{
		{"spec", "thorough"},
		{"review", "thorough"},
		{"test", "balanced"},
		{"implement", "balanced"},
		{"orient", "balanced"},
		{"", "balanced"},
	}
	for _, tt := range tests {
		got := SuggestMode(true, tt.phase)
		if got != tt.want {
			t.Errorf("SuggestMode(true, %q) = %q, want %q", tt.phase, got, tt.want)
		}
	}
}

func TestSuggestMode_NotActivated(t *testing.T) {
	got := SuggestMode(false, "spec")
	if got != "micro" {
		t.Errorf("SuggestMode(false, spec) = %q, want micro", got)
	}
}

func TestScore_Accumulative(t *testing.T) {
	s := Signals{
		ReferencesMultipleFiles:   true,
		InvolvesTemporalReasoning: true,
	}
	score, reasons := Score(s)
	if score != 4 {
		t.Errorf("score = %d, want 4", score)
	}
	if len(reasons) != 2 {
		t.Errorf("reasons = %v, want 2 entries", reasons)
	}
}

func TestScore_BelowThreshold(t *testing.T) {
	s := Signals{
		TaskIsContinuation: true, // +1, below threshold of 2
	}
	score, _ := Score(s)
	if score >= 2 {
		t.Errorf("score = %d, want < 2 for single weak signal", score)
	}
}
