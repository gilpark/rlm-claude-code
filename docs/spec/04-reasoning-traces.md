# SPEC-04: Reasoning Traces

## Overview

Implement Deciduous-style reasoning traces with git integration, stored in the hypergraph memory.

## Dependencies

- Requires SPEC-02 (Memory Foundation)

## Requirements

### Decision Node Types

[SPEC-04.01] The system SHALL support decision subtypes: "goal", "decision", "option", "action", "outcome", "observation".

[SPEC-04.02] Decision nodes SHALL be stored as nodes with type="decision" and appropriate subtype.

[SPEC-04.03] Decision nodes SHALL have additional fields: prompt, files (JSON), branch, commit_hash, parent_id.

### Decision Graph Structure

[SPEC-04.04] Goals SHALL spawn decisions via "spawns" hyperedge.

[SPEC-04.05] Decisions SHALL consider options via "considers" hyperedge.

[SPEC-04.06] Decisions SHALL choose one option via "chooses" hyperedge.

[SPEC-04.07] Decisions SHALL reject other options via "rejects" hyperedge with reason in edge label.

[SPEC-04.08] Decisions SHALL implement actions via "implements" hyperedge.

[SPEC-04.09] Actions SHALL produce outcomes via "produces" hyperedge.

[SPEC-04.10] Observations SHALL inform decisions via "informs" hyperedge.

### Git Integration

[SPEC-04.11] The system SHALL provide `link_commit(decision_id, commit_hash)` to associate decisions with commits.

[SPEC-04.12] The system SHALL capture current branch when creating decision nodes.

[SPEC-04.13] The system SHALL store pre/post diffs as snippet nodes linked via "implements" edge.

[SPEC-04.14] The system SHALL provide `get_decisions_for_commit(commit_hash) -> list[DecisionNode]`.

[SPEC-04.15] Git integration SHALL be optional; decisions work without git context.

### Query Interface

[SPEC-04.16] The system SHALL provide `get_decision_tree(goal_id) -> DecisionTree`.

[SPEC-04.17] The system SHALL provide `get_rejected_options(decision_id) -> list[Option]`.

[SPEC-04.18] The system SHALL provide `get_outcome(goal_id) -> Outcome | None`.

[SPEC-04.19] The system SHALL provide `get_informing_observations(decision_id) -> list[Observation]`.

### Trajectory Integration

[SPEC-04.20] TrajectoryEvent SHALL map to decision nodes automatically.

[SPEC-04.21] RECURSE events SHALL create "goal" nodes.

[SPEC-04.22] ORCHESTRATE events SHALL create "decision" nodes with options.

[SPEC-04.23] FINAL events SHALL create "outcome" nodes.

[SPEC-04.24] Trajectory-to-decision mapping SHALL be configurable (on/off).

### Event Types

The following TrajectoryEventType values are supported:

| Type | Description |
|------|-------------|
| `RLM_START` | RLM orchestration started |
| `ANALYZE` | Context analysis phase |
| `REPL_EXEC` | REPL code execution started |
| `REPL_RESULT` | REPL execution result |
| `REASON` | LLM reasoning/response |
| `STREAM` | Real-time streaming token (for streaming mode) |
| `RECURSE_START` | Recursive sub-query started |
| `RECURSE_END` | Recursive sub-query completed |
| `FINAL` | Final answer produced |
| `ERROR` | Error occurred |
| `TOOL_USE` | Tool invocation |
| `COST_REPORT` | Cost tracking update |
| `BUDGET_ALERT` | Budget threshold warning |
| `VERIFICATION` | Epistemic verification checkpoint |

### Schema Extension

[SPEC-04.25] The decisions table SHALL extend the memory schema:
```sql
CREATE TABLE decisions (
    node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    decision_type TEXT NOT NULL CHECK(decision_type IN
        ('goal', 'decision', 'option', 'action', 'outcome', 'observation')),
    confidence REAL DEFAULT 0.5,
    prompt TEXT,
    files JSON,
    branch TEXT,
    commit_hash TEXT,
    parent_id TEXT REFERENCES decisions(node_id)
);

CREATE INDEX idx_decisions_type ON decisions(decision_type);
CREATE INDEX idx_decisions_parent ON decisions(parent_id);
CREATE INDEX idx_decisions_commit ON decisions(commit_hash);
```

## Interface

```python
@dataclass
class DecisionNode:
    id: str
    decision_type: str  # goal, decision, option, action, outcome, observation
    content: str
    confidence: float
    prompt: str | None
    files: list[str]
    branch: str | None
    commit_hash: str | None
    parent_id: str | None

class ReasoningTraces:
    def create_goal(self, content: str, prompt: str | None = None) -> str:
        """Create a goal node, return ID."""

    def create_decision(self, goal_id: str, content: str) -> str:
        """Create a decision node under a goal."""

    def add_option(self, decision_id: str, content: str) -> str:
        """Add an option to a decision."""

    def choose_option(self, decision_id: str, option_id: str) -> None:
        """Mark an option as chosen."""

    def reject_option(self, decision_id: str, option_id: str, reason: str) -> None:
        """Mark an option as rejected with reason."""

    def create_action(self, decision_id: str, content: str) -> str:
        """Create an action implementing a decision."""

    def create_outcome(self, action_id: str, content: str, success: bool) -> str:
        """Create an outcome from an action."""

    def link_commit(self, decision_id: str, commit_hash: str) -> None:
        """Link a decision to a git commit."""

    def get_decisions_for_commit(self, commit_hash: str) -> list[DecisionNode]:
        """Get all decisions linked to a commit."""
```

## Testing Requirements

[SPEC-04.26] Unit tests SHALL verify decision graph structure is maintained.

[SPEC-04.27] Unit tests SHALL verify git linking works with valid commits.

[SPEC-04.28] Unit tests SHALL verify trajectory-to-decision mapping.

[SPEC-04.29] Integration tests SHALL verify decisions persist across sessions.

[SPEC-04.30] Property tests SHALL verify decision trees are acyclic.
