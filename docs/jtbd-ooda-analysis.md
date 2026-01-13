# RLM-Claude-Code: User JTBD and OODA Loop Analysis

A framework for understanding user jobs and the system's decision-making loops to guide empirical validation.

---

## Executive Summary

RLM-Claude-Code transforms Claude Code into a Recursive Language Model agent. The system's value proposition centers on handling **unbounded context** and **complex multi-step reasoning** that would otherwise overwhelm or confuse a direct LLM response.

The core insight: **RLM adds value when externalized reasoning leads to better answers than direct response**.

---

## User Jobs-to-be-Done (JTBD)

### JTBD-1: Debug Complex, Multi-Layer Issues

**Job Statement**: When I encounter an error that spans multiple files or system layers, I want to systematically trace the root cause so I can fix it efficiently.

**Trigger Conditions**:
- Stack traces spanning multiple modules
- "Used to work" scenarios
- Intermittent/flaky failures
- Error symptoms far from root cause

**Success Criteria**:
- Root cause identified correctly
- Fix targets the actual issue (not symptoms)
- Confidence in fix completeness

**OODA Loop**:
| Phase | System Behavior |
|-------|----------------|
| **Observe** | Extract complexity signals: `debugging_deep`, `multi_file_scope`, `temporal_reasoning` |
| **Orient** | Classify as high-value RLM scenario; select depth=3, model=opus, tools=read_only |
| **Decide** | Create plan: activate RLM, spawn sub-queries for each layer |
| **Act** | REPL: `search()` for error patterns, `llm()` sub-queries per module, `map_reduce()` for synthesis |

---

### JTBD-2: Understand Unfamiliar Codebase Architecture

**Job Statement**: When I'm working in an unfamiliar codebase, I want to understand how components interact so I can make changes safely.

**Trigger Conditions**:
- Questions like "How does X work in this codebase?"
- "Where is Y defined/implemented/used?"
- Cross-module interaction questions

**Success Criteria**:
- Accurate mental model of component relationships
- Identified key files and functions
- Understanding of data flow

**OODA Loop**:
| Phase | System Behavior |
|-------|----------------|
| **Observe** | Signals: `discovery_required`, `cross_module_reasoning` |
| **Orient** | Model tier=balanced/powerful, depth=2, tools=read_only |
| **Decide** | Strategy: peek structure, then deep-dive relevant files |
| **Act** | REPL: `peek(files)`, `find_relevant()`, `extract_functions()`, `llm()` for explanations |

---

### JTBD-3: Make Comprehensive Changes Across Codebase

**Job Statement**: When I need to update/refactor something used in multiple places, I want to find all usages and ensure consistent updates so nothing breaks.

**Trigger Conditions**:
- "Update all usages of X"
- Deprecated API migration
- Consistent error handling patterns

**Success Criteria**:
- All instances found (exhaustive)
- Changes consistent across all locations
- No breaking changes introduced

**OODA Loop**:
| Phase | System Behavior |
|-------|----------------|
| **Observe** | Signals: `synthesis_required`, `pattern_exhaustion`, `multi_file_scope` |
| **Orient** | Mode=thorough, depth=2, tools=full |
| **Decide** | Strategy: exhaustive search + parallel analysis per file |
| **Act** | REPL: `search()` with regex, `llm_batch()` for parallel context analysis, `memory_add_fact()` for tracking |

---

### JTBD-4: Make Informed Architectural Decisions

**Job Statement**: When facing a design decision with tradeoffs, I want to explore options systematically so I can make the best choice for my context.

**Trigger Conditions**:
- "Should I use X or Y?"
- "Best approach for adding Z?"
- Migration/refactoring planning

**Success Criteria**:
- Options clearly enumerated
- Tradeoffs explicit
- Recommendation justified for context

**OODA Loop**:
| Phase | System Behavior |
|-------|----------------|
| **Observe** | Signals: `architectural`, `uncertainty_high` |
| **Orient** | Model=powerful (opus), depth=3, tools=read_only |
| **Decide** | Strategy: enumerate options, analyze each, synthesize recommendation |
| **Act** | REPL: `llm()` for each option analysis, reasoning traces for decision tree |

---

### JTBD-5: Ensure Security/Quality Completeness

**Job Statement**: When reviewing code for security or quality issues, I want systematic coverage so nothing is missed.

**Trigger Conditions**:
- "Find all security vulnerabilities"
- "Ensure all edge cases handled"
- Large changeset review

**Success Criteria**:
- Systematic enumeration
- No false negatives
- Prioritized findings

**OODA Loop**:
| Phase | System Behavior |
|-------|----------------|
| **Observe** | Signals: `pattern_exhaustion`, `user_careful` |
| **Orient** | Mode=thorough, depth=3, model=opus |
| **Decide** | Strategy: partition codebase, systematic analysis per section |
| **Act** | REPL: `map_reduce()` with security-focused prompts, `memory_add_experience()` for findings |

---

### JTBD-6: Resume/Continue Previous Work Efficiently

**Job Statement**: When continuing work from a previous session, I want the system to remember context so I don't have to re-explain.

**Trigger Conditions**:
- "Continue from yesterday"
- "Same task as before"
- Large context built up over conversation

**Success Criteria**:
- Prior context recalled accurately
- No redundant analysis
- Smooth continuation

**OODA Loop**:
| Phase | System Behavior |
|-------|----------------|
| **Observe** | Signals: `continuation`, `task_is_continuation`, context_tokens > threshold |
| **Orient** | Check memory store for relevant facts/experiences |
| **Decide** | Use memory-augmented context, lower depth if facts already known |
| **Act** | REPL: `memory_query()`, `memory_get_context()` before new analysis |

---

### JTBD-7: Get Quick Answers Without Overhead

**Job Statement**: When I have a simple question or command, I want a fast response without unnecessary processing.

**Trigger Conditions**:
- Simple file reads
- Syntax questions
- Single-file changes
- Acknowledgments ("ok", "thanks")

**Success Criteria**:
- Fast response (<5s)
- No unnecessary token spend
- Direct, accurate answer

**OODA Loop**:
| Phase | System Behavior |
|-------|----------------|
| **Observe** | `is_definitely_simple()` returns True |
| **Orient** | Bypass RLM entirely |
| **Decide** | Return "simple_task" reason |
| **Act** | Direct Claude Code response (no REPL, no sub-queries) |

---

## System OODA Flow Diagram

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    OBSERVE                                   │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │ Complexity Signals  │    │ Context Analysis            │ │
│  │ • Multi-file refs   │    │ • Total tokens              │ │
│  │ • Cross-context     │    │ • Active modules            │ │
│  │ • Debugging keywords│    │ • Recent tool outputs       │ │
│  │ • Temporal patterns │    │ • Previous confusion        │ │
│  │ • Pattern search    │    │ • State changes             │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORIENT                                    │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │ Query Classifier    │    │ Intelligent Orchestrator    │ │
│  │ • Query type        │───►│ • Activate RLM? (bool)      │ │
│  │ • Complexity score  │    │ • Model tier selection      │ │
│  │ • Confidence        │    │ • Depth budget (0-3)        │ │
│  └─────────────────────┘    │ • Tool access level         │ │
│                             │ • Execution mode            │ │
│  ┌─────────────────────┐    └─────────────────────────────┘ │
│  │ User Preferences    │                                    │
│  │ • Forced mode       │                                    │
│  │ • Budget limits     │                                    │
│  │ • Auto-activate     │                                    │
│  └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    DECIDE                                    │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                OrchestrationPlan                        ││
│  │  • activate_rlm: true/false                             ││
│  │  • activation_reason: "discovery_required"              ││
│  │  • model_tier: FAST | BALANCED | POWERFUL               ││
│  │  • primary_model: "sonnet" | "opus" | "haiku"           ││
│  │  • depth_budget: 0-3                                    ││
│  │  • tool_access: none | repl_only | read_only | full     ││
│  │  • execution_mode: fast | balanced | thorough           ││
│  │  • signals: ["discovery_required", "multi_file_scope"]  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ACT                                       │
│                                                              │
│  IF activate_rlm == false:                                  │
│    └─► Direct Claude Code response                          │
│                                                              │
│  IF activate_rlm == true:                                   │
│    ┌────────────────────────────────────────────────────┐   │
│    │              RLM Execution Loop                     │   │
│    │                                                     │   │
│    │  1. Externalize context → Python variables         │   │
│    │  2. Claude generates REPL code                     │   │
│    │  3. Execute in sandbox:                            │   │
│    │     • peek(), search(), summarize()                │   │
│    │     • llm(), llm_batch() → sub-queries             │   │
│    │     • map_reduce() → parallel processing           │   │
│    │     • memory_*() → persistent storage              │   │
│    │  4. Feed results back to Claude                    │   │
│    │  5. Repeat until FINAL: <answer>                   │   │
│    │                                                     │   │
│    │  Recursion: Sub-queries can spawn their own REPL   │   │
│    │  Budget: Track tokens/cost, enforce limits         │   │
│    │  Trajectory: Stream events for visibility          │   │
│    └────────────────────────────────────────────────────┘   │
│                                                              │
│  FINALLY:                                                    │
│    • Emit cost report                                        │
│    • Store reasoning traces                                  │
│    • Update memory (promote facts if successful)            │
│    • Export trajectory if enabled                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Critical User Touchpoints

### 1. Slash Commands (Direct Control)

| Command | User Intent | System Response |
|---------|-------------|-----------------|
| `/rlm on` | Force RLM activation | Set `forced_rlm=true`, bypass classifier |
| `/rlm off` | Disable RLM | Set `auto_activate=false` |
| `/rlm mode thorough` | Need deep analysis | Set mode=THOROUGH, depth=3, model=opus |
| `/rlm mode fast` | Prioritize speed | Set mode=FAST, depth=1, model=haiku |
| `/simple` | Skip RLM this query | One-time bypass |
| `/rlm budget $5` | Cost control | Set max_cost_dollars limit |
| `/rlm verbosity debug` | See all decisions | Show activation reasoning |

### 2. Implicit Signals (Query Analysis)

| User Signal | Detected Pattern | System Interpretation |
|-------------|-----------------|----------------------|
| "why is..." | `discovery_required` | RLM activation likely |
| "find all..." | `pattern_exhaustion` | Mode=thorough, exhaustive search |
| "make sure" | `user_careful` | Mode=thorough |
| "quick" / "just" | `user_urgent` | Mode=fast |
| File mentions | `multi_file_scope` | Cross-file reasoning |
| Error keywords | `debugging_deep/surface` | Debug strategy |

### 3. Trajectory Stream (Feedback)

Users observe system state through trajectory events:

| Event Type | Information Provided |
|------------|---------------------|
| `RLM_START` | Activation reason, routing decision |
| `ANALYZE` | Context size, file count |
| `REPL_EXEC` | Code being executed |
| `REPL_RESULT` | Execution output |
| `RECURSE_START/END` | Sub-query spawning, depth |
| `COST_REPORT` | Token usage, cost breakdown |
| `BUDGET_ALERT` | Budget warnings, model downgrades |
| `FINAL` | Answer, total turns, cost |

### 4. Memory System (Persistence)

| Interaction | User Benefit |
|-------------|--------------|
| `memory_add_fact()` | "Remember this about my codebase" |
| `memory_query()` | "What do you already know?" |
| Session-to-longterm promotion | Learning across sessions |
| Reasoning traces | "Why did you decide that?" |

---

## Validation Hypotheses

Based on this analysis, key hypotheses to validate empirically:

### H1: Activation Accuracy
The complexity classifier correctly identifies when RLM adds value. Measure: false positive rate (RLM activated but wasn't needed) and false negative rate (RLM not activated but should have been).

### H2: JTBD Coverage
The seven identified JTBDs cover >80% of real user scenarios. Measure: user interview tagging of actual queries.

### H3: Trajectory Value
Users find trajectory stream helpful for understanding system behavior. Measure: survey on trajectory usefulness, correlation with confidence in results.

### H4: Memory Utility
Cross-session memory improves outcomes on recurring tasks. Measure: A/B test with/without memory on same-user return visits.

### H5: Mode Appropriateness
Fast/balanced/thorough modes correspond to user-perceived task complexity. Measure: user satisfaction correlation with mode selection.

### H6: Budget Satisfaction
Users are satisfied with cost/quality tradeoffs at different budget levels. Measure: survey on budget setting behavior and outcome satisfaction.

---

## Next Steps for Empirical Validation

1. **Instrument telemetry**: Log all activation decisions with outcomes
2. **User interviews**: Validate JTBD with actual usage patterns
3. **A/B testing**: Compare RLM vs direct for marginal complexity queries
4. **Trajectory analysis**: Correlate verbosity settings with user confidence
5. **Memory effectiveness**: Track fact recall accuracy across sessions

---

## References

- Source: `/Users/rand/src/rlm-claude-code/src/complexity_classifier.py`
- Source: `/Users/rand/src/rlm-claude-code/src/orchestrator/intelligent.py`
- Source: `/Users/rand/src/rlm-claude-code/src/auto_activation.py`
- Source: `/Users/rand/src/rlm-claude-code/src/trajectory.py`
- Spec: `/Users/rand/src/rlm-claude-code/docs/spec/00-overview.md`
