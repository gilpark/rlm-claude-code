# RLM-Claude-Code API Cookbook

Copy-paste examples for common API operations. All examples assume you've installed the package and are working in Python.

## Table of Contents

- [Memory Store](#memory-store)
- [Budget Tracking](#budget-tracking)
- [REPL Environment](#repl-environment)
- [Trajectory Events](#trajectory-events)
- [Orchestration Plans](#orchestration-plans)
- [Complexity Classification](#complexity-classification)

---

## Memory Store

The `MemoryStore` provides persistent hypergraph memory with SQLite storage.

### Basic Setup

```python
from src.memory_store import MemoryStore

# In-memory database (for testing)
store = MemoryStore(":memory:")

# Persistent database (default location: ~/.claude/rlm-memory.db)
store = MemoryStore()

# Custom path
store = MemoryStore("/path/to/memory.db")
```

### Creating Nodes

```python
# Create a fact node
fact_id = store.create_node("fact", "Auth module uses JWT tokens")

# With explicit type= keyword (alternative syntax)
fact_id = store.create_node(type="fact", content="Auth uses JWT")

# With confidence and metadata
fact_id = store.create_node(
    "fact",
    "Rate limiting is set to 100 req/min",
    confidence=0.9,
    metadata={"source": "config.yaml", "line": 42}
)

# Valid node types: entity, fact, experience, decision, snippet
experience_id = store.create_node(
    "experience",
    "Refactoring auth module reduced bugs by 50%",
    tier="session"  # task, session, longterm, archive
)
```

### Convenience Methods (Recommended)

```python
# add_fact() - Create fact nodes easily
fact_id = store.add_fact("Auth uses JWT tokens")
fact_id = store.add_fact(
    "Rate limit is 100/min",
    confidence=0.9,
    tier="longterm",
    metadata={"source": "config.yaml"}
)

# add_experience() - Record learnings (outcome is required)
exp_id = store.add_experience(
    "Refactoring reduced bugs by 50%",
    outcome="success",         # Required: describes what happened
    success=True,              # Optional: boolean flag (default True)
    confidence=0.85,
    metadata={"project": "auth-module"}
)

# add_entity() - Track code entities
entity_id = store.add_entity(
    "AuthService",
    entity_type="class",       # class, function, module, variable
    metadata={"file": "src/auth.py", "line": 42}
)

# find() - Intuitive search with k parameter (common ML API style)
results = store.find("authentication", k=5)
results = store.find("JWT", k=10, node_type="fact", min_confidence=0.8)
```

### Querying Nodes

```python
# Get node by ID
node = store.get_node(fact_id)
print(f"Content: {node.content}")
print(f"Confidence: {node.confidence}")

# Query with filters
facts = store.query_nodes(node_type="fact", min_confidence=0.8, limit=10)
session_nodes = store.query_nodes(tier="session")
```

### Full-Text Search (BM25)

```python
# Basic search - returns SearchResult objects
results = store.search("authentication error")
for r in results:
    print(f"{r.content[:50]}... (score: {r.bm25_score:.2f})")

# With filters
results = store.search("JWT", node_type="fact", limit=5)

# Alternative syntax with k= (common in ML APIs)
results = store.search("token", k=10, type="fact")

# Prefix search
results = store.search_prefix("auth")  # Matches "auth", "authentication", etc.

# Exact phrase search
results = store.search_phrase("rate limiting")
```

### Linking Nodes (Relationships)

```python
# Simple two-node link (recommended for most cases)
edge_id = store.link(fact_id, decision_id, "supports")

# With edge type and weight
edge_id = store.link(
    cause_id,
    effect_id,
    "triggers",
    edge_type="causation",  # relation, composition, causation, context
    weight=0.8
)

# Get links for a node
outgoing = store.get_links(node_id, direction="outgoing")
incoming = store.get_links(node_id, direction="incoming")
all_links = store.get_links(node_id, direction="both")

# Filter by label
supports = store.get_links(node_id, label="supports")

# Remove links
store.unlink(source_id, target_id, "supports")  # Remove specific label
store.unlink(source_id, target_id)  # Remove all links between nodes
```

### Evidence Relationships

```python
# Pre-defined evidence labels: supports, contradicts, validates, invalidates
edge_id = store.create_evidence_edge("supports", fact_id, option_id, weight=0.9)

# Query evidence
supporting = store.get_supporting_facts(option_id)  # [(fact_id, weight), ...]
contradicting = store.get_contradicting_facts(option_id)
```

---

## Budget Tracking

The `EnhancedBudgetTracker` tracks token usage and costs across LLM calls.

### Basic Setup

```python
from src.enhanced_budget import EnhancedBudgetTracker, CostComponent

# Create tracker with limits
tracker = EnhancedBudgetTracker(
    max_tokens=100_000,
    max_cost=5.0  # dollars
)
```

### Recording Usage

```python
# Record an LLM call
tracker.record_llm_call(
    component=CostComponent.ORCHESTRATION,  # or REPL, RECURSIVE, SUMMARIZATION
    model="claude-sonnet-4-20250514",
    input_tokens=1500,
    output_tokens=500
)

# Check status
print(f"Total tokens: {tracker.total_tokens}")
print(f"Total cost: ${tracker.total_cost:.4f}")
print(f"Within budget: {tracker.within_budget}")
```

### Cost Components

```python
from src.enhanced_budget import CostComponent

# Available components for categorizing costs:
CostComponent.ORCHESTRATION  # Main orchestrator decisions
CostComponent.REPL           # REPL code generation/execution
CostComponent.RECURSIVE      # Recursive sub-calls
CostComponent.SUMMARIZATION  # Context summarization
```

### Budget Queries

```python
# Get remaining budget
remaining = tracker.remaining_tokens
remaining_cost = tracker.remaining_cost

# Get breakdown by component
breakdown = tracker.get_cost_breakdown()
# {'orchestration': 0.05, 'repl': 0.02, 'recursive': 0.10, ...}

# Check if specific operation fits in budget
can_afford = tracker.can_afford(input_tokens=5000, output_tokens=2000)
```

---

## REPL Environment

The `ReplEnvironment` provides a sandboxed Python execution environment.

### Basic Setup

```python
from src.repl_environment import ReplEnvironment
from src.types import SessionContext

# Create context with files
context = SessionContext(
    files={
        "src/auth.py": "def login(): ...",
        "src/api.py": "def fetch(): ...",
    }
)

# Create REPL environment
env = ReplEnvironment(context)
```

### Executing Code

```python
# Execute code - expressions return their value
result = env.execute("1 + 1")
print(result.output)  # 2

# Access context files
result = env.execute("list(files.keys())")
print(result.output)  # ['src/auth.py', 'src/api.py']

# Multi-line code
result = env.execute("""
def count_functions(content):
    return content.count('def ')

total = sum(count_functions(c) for c in files.values())
total
""")
print(result.output)  # Number of functions

# Check for errors
if not result.success:
    print(f"Error: {result.error}")
```

### ExecutionResult Properties

```python
# ExecutionResult fields
result = env.execute("print('hello'); 42")

result.success           # True if no error
result.output            # Return value (42)
result.stdout            # Captured print output ("hello\n")
result.error             # Error message (None if success)
result.execution_time_ms # Execution time

# Truncation handling for large outputs
result.output_truncated  # True if output was truncated
result.original_length   # Original length before truncation

# Pretty-print with format_output()
result = env.execute("{'a': 1, 'b': [1, 2, 3]}")
print(result.format_output())
# {
#   "a": 1,
#   "b": [1, 2, 3]
# }

# Custom max length
print(result.format_output(max_length=100))
```

### Enhanced Error Messages

The REPL provides helpful suggestions for common errors:

```python
# NameError - suggests similar names
result = env.execute("prnt('hello')")
# Error: name 'prnt' is not defined
# Did you mean: print?

# KeyError - shows available keys
result = env.execute("d = {'name': 'test'}; d['nmae']")
# Error: 'nmae'
# Available keys: ['name']

# AttributeError - suggests similar attributes
result = env.execute("'hello'.uper()")
# Error: 'str' object has no attribute 'uper'
# Did you mean: upper?
```

### Built-in Helper Functions

```python
# peek() - View slices of content
result = env.execute("peek(files['src/auth.py'], 0, 100)")

# search() - Find patterns
result = env.execute("search(files['src/auth.py'], 'def ')")

# Available in REPL: peek, search, grep, summarize, llm, llm_batch,
#                    map_reduce, find_relevant, extract_functions,
#                    memory_query, memory_add_fact, memory_add_experience
```

---

## Trajectory Events

Track execution events for debugging and analysis.

### Creating Events

```python
from src.trajectory import TrajectoryEvent, TrajectoryEventType
import time

# Create an event
event = TrajectoryEvent(
    type=TrajectoryEventType.REPL_EXEC,  # Not event_type!
    content="result = search(files, 'auth')",
    depth=1,
    timestamp=time.time()
)

# Event types
TrajectoryEventType.PROMPT      # Initial prompt received
TrajectoryEventType.REPL_EXEC   # REPL code executed
TrajectoryEventType.REPL_RESULT # REPL execution result
TrajectoryEventType.RECURSE_START  # Recursive call started
TrajectoryEventType.RECURSE_END    # Recursive call completed
TrajectoryEventType.FINAL       # Final answer produced
TrajectoryEventType.ERROR       # Error occurred
```

### Analyzing Trajectories

```python
from src.trajectory_analysis import TrajectoryAnalyzer, analyze_trajectory

# Analyze a list of events
analyzer = TrajectoryAnalyzer()
analysis = analyzer.analyze(events)

# Or use convenience function
analysis = analyze_trajectory(events)

# Access results
print(f"Primary strategy: {analysis.primary_strategy.value}")
print(f"Confidence: {analysis.strategy_confidence:.0%}")
print(f"Success: {analysis.success}")
print(f"Effectiveness: {analysis.effectiveness_score:.0%}")

# Strategy types: PEEKING, GREPPING, PARTITION_MAP, PROGRAMMATIC,
#                 RECURSIVE, ITERATIVE, DIRECT, UNKNOWN
```

---

## Orchestration Plans

Configure how RLM processes queries.

### Creating Plans

```python
from src.orchestration_schema import (
    OrchestrationPlan,
    ExecutionMode,
    ToolAccessLevel,
)
from src.smart_router import ModelTier, QueryType

# Create from execution mode (recommended)
plan = OrchestrationPlan.from_mode(
    ExecutionMode.BALANCED,  # FAST, BALANCED, or THOROUGH
    query_type=QueryType.DEBUG,
    activation_reason="Complex debugging task"
)

# Create bypass plan (skip RLM)
plan = OrchestrationPlan.bypass(reason="simple_task")

# Full manual configuration
plan = OrchestrationPlan(
    activate_rlm=True,
    activation_reason="Multi-file refactoring",
    model_tier=ModelTier.POWERFUL,
    primary_model="claude-opus-4-5-20251101",
    depth_budget=3,
    tokens_per_depth=50_000,
    execution_mode=ExecutionMode.THOROUGH,
    tool_access=ToolAccessLevel.FULL,
    max_cost_dollars=10.0
)
```

### Execution Modes

| Mode | Depth | Model | Tools | Budget |
|------|-------|-------|-------|--------|
| FAST | 1 | Haiku | REPL only | $0.50 |
| BALANCED | 2 | Sonnet | Read-only | $2.00 |
| THOROUGH | 3 | Opus | Full | $10.00 |

### Tool Access Levels

```python
from src.orchestration_schema import ToolAccessLevel

ToolAccessLevel.NONE       # Pure reasoning, no tools
ToolAccessLevel.REPL_ONLY  # Only Python REPL
ToolAccessLevel.READ_ONLY  # REPL + file reading
ToolAccessLevel.FULL       # All Claude Code tools
```

### Query Plan Properties

```python
# Check plan properties
plan.allows_recursion      # True if depth_budget > 0
plan.allows_tools          # True if tool_access != NONE
plan.allows_file_read      # True if READ_ONLY or FULL
plan.allows_file_write     # True if FULL
plan.total_token_budget    # depth_budget * tokens_per_depth
```

---

## Complexity Classification

Determine when to activate RLM based on query complexity.

### Check Activation

```python
from src.complexity_classifier import (
    should_activate_rlm,
    extract_complexity_signals,
    is_definitely_simple,
)
from src.types import SessionContext

context = SessionContext()  # Your session context

# Check if RLM should activate
should_activate, reason = should_activate_rlm(
    prompt="Find all places where auth is used",
    context=context
)
print(f"Activate: {should_activate}, Reason: {reason}")

# Force activation
should_activate, reason = should_activate_rlm(
    prompt="Simple question",
    context=context,
    rlm_mode_forced=True
)

# Force bypass
should_activate, reason = should_activate_rlm(
    prompt="Complex task",
    context=context,
    simple_mode_forced=True
)
```

### Extract Signals

```python
# Get detailed complexity signals
signals = extract_complexity_signals(prompt, context)

print(f"Multiple files: {signals.references_multiple_files}")
print(f"Cross-context: {signals.requires_cross_context_reasoning}")
print(f"Debugging: {signals.debugging_task}")
print(f"Exhaustive search: {signals.requires_exhaustive_search}")
print(f"User wants thorough: {signals.user_wants_thorough}")
```

### Quick Simple Check

```python
# Fast check for definitely simple queries
if is_definitely_simple(prompt, context):
    # Skip RLM entirely
    pass
```

---

## Complete Example

```python
from src.memory_store import MemoryStore
from src.enhanced_budget import EnhancedBudgetTracker, CostComponent
from src.repl_environment import ReplEnvironment
from src.complexity_classifier import should_activate_rlm
from src.orchestration_schema import OrchestrationPlan, ExecutionMode
from src.types import SessionContext

# Setup
store = MemoryStore(":memory:")
tracker = EnhancedBudgetTracker(max_tokens=50_000, max_cost=2.0)
context = SessionContext(files={"main.py": "def main(): pass"})

# Store facts using convenience methods
fact_id = store.add_fact("Main entry point is main.py", confidence=0.95)
store.add_experience(
    "Async I/O improved response times",
    outcome="3x faster response times",  # Required: describes what happened
    success=True,
    metadata={"improvement": "3x faster"}
)
store.add_entity("main", entity_type="function", metadata={"file": "main.py"})

# Search memory
results = store.find("async", k=5)
for r in results:
    print(f"Found: {r.content[:50]}...")

# Check if query needs RLM
prompt = "Find all async functions and ensure they handle errors"
should_activate, reason = should_activate_rlm(prompt, context)

if should_activate:
    # Create execution plan
    plan = OrchestrationPlan.from_mode(
        ExecutionMode.BALANCED,
        activation_reason=reason
    )
    print(f"RLM activated: {reason}")
    print(f"Using model: {plan.primary_model}")
    print(f"Depth budget: {plan.depth_budget}")

    # Use REPL for analysis
    env = ReplEnvironment(context)
    result = env.execute("list(files.keys())")
    if result.success:
        print(f"Files: {result.format_output()}")
    else:
        print(f"Error: {result.error}")
else:
    print("Simple query - direct response")

# Record usage
tracker.record_llm_call(
    component=CostComponent.ORCHESTRATION,
    model="claude-sonnet-4-20250514",
    input_tokens=1000,
    output_tokens=500
)
print(f"Cost so far: ${tracker.total_cost:.4f}")
```
