// Package classify provides prompt complexity classification for RLM activation.
// Ported from src/complexity_classifier.py to Go.
package classify

import (
	"regexp"
	"strings"
)

// Signals represents complexity signals extracted from a user prompt.
type Signals struct {
	ReferencesMultipleFiles       bool
	RequiresCrossContextReasoning bool
	InvolvesTemporalReasoning     bool
	AsksAboutPatterns             bool
	DebuggingTask                 bool
	RequiresExhaustiveSearch      bool
	SecurityReviewTask            bool
	ArchitectureAnalysis          bool
	UserWantsThorough             bool
	UserWantsFast                 bool
	PreviousTurnConfused          bool
	TaskIsContinuation            bool
	ContextHasMultipleDomains     bool
	FilesSpanMultipleModules      bool
}

// Precompiled pattern sets.
var (
	fileExtensionRe = regexp.MustCompile(`(?i)\b\w+\.(ts|js|py|go|rs|tsx|jsx|rb|java|cpp|c|h)\b`)
	modulePairRe    = regexp.MustCompile(`(?i)\b(auth|api|db|ui|test|config)\b.*\b(auth|api|db|ui|test|config)\b`)

	crossContextPatterns = compileAll(
		`(?i)\bwhy\b.*\b(when|if|given|since)\b`,
		`(?i)\bhow\b.*\b(relate|connect|affect|impact)\b`,
		`(?i)\bwhat\b.*\b(cause|led to|result)\b`,
		`(?i)\b(trace|follow|track)\b.*\b(through|across)\b`,
	)

	temporalPatterns = compileAll(
		`(?i)\b(before|after|since|when|changed|used to|previously)\b`,
		`(?i)\b(history|log|commit|version|diff)\b`,
		`(?i)\blast\s+(time|session|attempt)\b`,
	)

	patternSearchPatterns = compileAll(
		`(?i)\b(find|search|locate|grep|all|every|each)\b.*\b(where|that|which)\b`,
		`(?i)\bhow many\b`,
		`(?i)\blist\s+(all|every)\b`,
	)

	debugPatterns = compileAll(
		`(?i)\b(error|exception|fail\w*|crash\w*|bug|issue|broken)\b`,
		`(?i)\b(stack\s*trace|traceback|stderr)\b`,
		`(?i)\b(debug\w*|diagnos\w*|investigat\w*|troubleshoot\w*)\b`,
	)

	exhaustivePatterns = compileAll(
		`(?i)\b(find|list|show|get)\b.*\b(all|every)\b`,
		`(?i)\b(ensure|check|verify)\b.*\b(all|every|each)\b`,
		`(?i)\b(comprehensive|exhaustive|complete)\s+(list|search|scan|review)\b`,
		`(?i)\ball\s+(the\s+)?(places|instances|usages|occurrences|references)\b`,
		`(?i)\b(fix|update|change|remove)\b.*\b(all|every|each)\b`,
	)

	securityPatterns = compileAll(
		`(?i)\b(security|vulnerabilit|exploit|attack|injection|xss|csrf|auth)\w*\b`,
		`(?i)\b(review|audit)\b.*\b(code|pr|pull\s*request|changes?|implementation)\b`,
		`(?i)\b(code|pr|pull\s*request)\b.*\b(review|audit)\b`,
		`(?i)\b(check|scan|analyze)\b.*\b(security|vulnerabil|risk)\b`,
	)

	architecturePatterns = compileAll(
		`(?i)\b(architecture|architect)\b`,
		`(?i)\b(system|overall|high.?level)\s+(design|structure|overview)\b`,
		`(?i)\b(summarize|explain|understand)\b.*\b(codebase|project|system|architecture)\b`,
		`(?i)\b(how\s+does|how\s+do)\b.*\b(work|fit|connect|integrate|interact|communicate|call|use)\b`,
		`(?i)\b(design|structure)\s+(of|for)\s+(the|this)\b`,
		`(?i)\b(refactor|restructure|reorganize)\b`,
		`(?i)\b(explain|describe|trace)\b.*\b(data\s*flow|flow|path|pipeline)\b`,
	)

	thoroughPatterns = compileAll(
		`(?i)\bmake\s+sure\b`,
		`(?i)\bbe\s+careful\b`,
		`(?i)\b(thorough|thoroughly)\b`,
		`(?i)\bdon'?t\s+miss\b`,
		`(?i)\bcheck\s+everything\b`,
		`(?i)\b(verify|validate|confirm)\s+(all|every|each)\b`,
		`(?i)\b(important|critical|crucial)\b`,
	)

	fastPatterns = compileAll(
		`(?i)\b(quick|quickly)\b`,
		`(?i)\bjust\s+(show|tell|give)\b`,
		`(?i)\bbriefly\b`,
		`(?i)\bsimple\s+(answer|explanation)\b`,
	)

	fastPathPatterns = compileAll(
		`(?i)^git\s+(status|log|diff|branch)$`,
		`(?i)^(yes|no|ok|okay|sure|thanks|got it|understood)\.?$`,
		`(?i)^run\s+(pytest|npm|yarn|cargo|go|make)\b`,
		`(?i)^(show|cat|read|view|open)\s+[\w./]+$`,
	)
)

func compileAll(patterns ...string) []*regexp.Regexp {
	out := make([]*regexp.Regexp, len(patterns))
	for i, p := range patterns {
		out[i] = regexp.MustCompile(p)
	}
	return out
}

func matchAny(patterns []*regexp.Regexp, s string) bool {
	for _, p := range patterns {
		if p.MatchString(s) {
			return true
		}
	}
	return false
}

// IsFastPath returns true for trivial prompts that should bypass classification entirely.
func IsFastPath(prompt string) bool {
	trimmed := strings.TrimSpace(prompt)
	if len(trimmed) == 0 {
		return true
	}
	return matchAny(fastPathPatterns, trimmed)
}

// ExtractSignals extracts complexity signals from a user prompt using regex heuristics.
func ExtractSignals(prompt string) Signals {
	fileMatches := fileExtensionRe.FindAllString(prompt, -1)
	hasModulePair := modulePairRe.MatchString(prompt)

	// Deduplicate file extensions to count distinct ones
	extSet := make(map[string]struct{})
	for _, m := range fileMatches {
		parts := strings.Split(m, ".")
		if len(parts) > 1 {
			extSet[parts[len(parts)-1]] = struct{}{}
		}
	}

	refsMultiple := len(fileMatches) >= 2 || hasModulePair

	promptLower := strings.ToLower(prompt)

	return Signals{
		ReferencesMultipleFiles:       refsMultiple,
		RequiresCrossContextReasoning: matchAny(crossContextPatterns, prompt),
		InvolvesTemporalReasoning:     matchAny(temporalPatterns, prompt),
		AsksAboutPatterns:             matchAny(patternSearchPatterns, prompt),
		DebuggingTask:                 matchAny(debugPatterns, prompt),
		RequiresExhaustiveSearch:      matchAny(exhaustivePatterns, prompt),
		SecurityReviewTask:            matchAny(securityPatterns, prompt),
		ArchitectureAnalysis:          matchAny(architecturePatterns, prompt),
		UserWantsThorough:             matchAny(thoroughPatterns, prompt),
		UserWantsFast:                 matchAny(fastPatterns, prompt),
		FilesSpanMultipleModules:      len(extSet) >= 2 || hasModulePair,
		TaskIsContinuation: strings.Contains(promptLower, "continue") ||
			strings.Contains(promptLower, "same") ||
			(len(promptLower) > 4 && strings.Contains(promptLower[:min(50, len(promptLower))], "also")),
	}
}

// Score computes a cumulative complexity score from signals.
// Returns the score and list of contributing reasons.
func Score(s Signals) (int, []string) {
	score := 0
	var reasons []string

	if s.ReferencesMultipleFiles {
		score += 2
		reasons = append(reasons, "multi_file")
	}
	if s.InvolvesTemporalReasoning {
		score += 2
		reasons = append(reasons, "temporal")
	}
	if s.AsksAboutPatterns {
		score += 2
		reasons = append(reasons, "pattern_search")
	}
	if s.PreviousTurnConfused {
		score += 2
		reasons = append(reasons, "prior_confusion")
	}
	if s.UserWantsThorough {
		score += 2
		reasons = append(reasons, "user_thorough")
	}
	if s.ContextHasMultipleDomains {
		score += 1
		reasons = append(reasons, "multi_domain")
	}
	if s.TaskIsContinuation {
		score += 1
		reasons = append(reasons, "continuation")
	}

	return score, reasons
}

// SuggestMode maps activation state and DP phase to an RLM mode string.
func SuggestMode(activate bool, dpPhase string) string {
	if !activate {
		return "micro"
	}

	switch dpPhase {
	case "spec", "review":
		return "thorough"
	case "test", "implement", "orient", "decide":
		return "balanced"
	default:
		return "balanced"
	}
}

// ShouldActivate determines whether RLM should activate for the given prompt.
// Returns activation decision, reason string, and suggested mode.
func ShouldActivate(prompt, dpPhase, rigor string) (bool, string, string) {
	signals := ExtractSignals(prompt)

	// Fast intent suppresses activation
	if signals.UserWantsFast {
		return false, "fast_intent", "micro"
	}

	// High-signal indicators (each sufficient alone)
	if signals.RequiresCrossContextReasoning {
		return true, "cross_context_reasoning", SuggestMode(true, dpPhase)
	}
	if signals.DebuggingTask {
		return true, "debugging_task", SuggestMode(true, dpPhase)
	}
	if signals.RequiresExhaustiveSearch {
		return true, "exhaustive_search", SuggestMode(true, dpPhase)
	}
	if signals.SecurityReviewTask {
		return true, "security_review", SuggestMode(true, dpPhase)
	}
	if signals.ArchitectureAnalysis {
		return true, "architecture_analysis", SuggestMode(true, dpPhase)
	}
	if signals.ReferencesMultipleFiles && signals.FilesSpanMultipleModules {
		return true, "multi_module_task", SuggestMode(true, dpPhase)
	}

	// Accumulative scoring
	score, reasons := Score(signals)
	if score >= 2 {
		reason := "complexity_score:" + strings.Join(reasons, "+")
		return true, reason, SuggestMode(true, dpPhase)
	}

	return false, "simple_task", "micro"
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
