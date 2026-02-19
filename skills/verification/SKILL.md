# Causeway Constraint Verification Skill

## When to Activate

This skill activates when:
- Verifying proposed code changes won't break invariants
- Analyzing dependencies between components
- Checking resource constraints or scheduling feasibility
- Generating test inputs satisfying path constraints

## CPMpy Quick Reference

```python
import cpmpy as cp

# Integer variables
x = cp.intvar(lb=0, ub=100, name="x")
xs = cp.intvar(0, 100, shape=10, name="xs")  # Array

# Boolean variables
b = cp.boolvar(name="b")
bs = cp.boolvar(shape=5, name="bs")

# Model and constraints
model = cp.Model()
model += x + y == 10          # Arithmetic
model += x < y                 # Comparison
model += b.implies(x > 5)      # Implication
model += cp.AllDifferent(xs)   # Global constraint

# Solve
if model.solve():
    print(x.value(), y.value())
```

## Common Patterns

### Dependency Graph Verification
```python
# Verify no circular dependencies
n_modules = len(modules)
order = cp.intvar(0, n_modules-1, shape=n_modules, name="order")
model = cp.Model()
model += cp.AllDifferent(order)

for (a, b) in dependencies:  # a depends on b
    model += order[a] > order[b]  # a must come after b

if model.solve():
    print("Valid ordering exists")
else:
    print("Circular dependency detected")
```

### Resource Bound Checking
```python
# Verify tasks fit within resource capacity
starts = cp.intvar(0, horizon, shape=n_tasks, name="start")
model = cp.Model()
model += cp.Cumulative(starts, durations, ends, demands, capacity=max_capacity)

if model.solve():
    print("Schedule feasible")
```

## Integration with Causeway

At depth=2, use constraint verification to check proposed changes:

```python
def verify_with_constraints(change, context):
    """
    Verify a proposed change using constraint modeling.
    Called at depth=2 for safety verification.
    """
    import cpmpy as cp

    # Extract invariants from docstrings/comments
    invariants = extract_invariants(context)

    # Build constraint model
    model = cp.Model()
    for inv in invariants:
        model += encode_invariant(inv)

    # Add change effects
    model += encode_change(change)

    # Check satisfiability
    if model.solve():
        return {"safe": True, "witness": get_solution(model)}
    else:
        return {"safe": False, "conflicts": analyze_unsat(model)}
```

## References

- CPMpy Docs: https://cpmpy.readthedocs.io/
- Design: docs/plans/2026-02-19-design.md
- Whitepaper: docs/plans/2026-02-19-whitepaper.md
