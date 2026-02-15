# RLM-Claude-Code

Context for developers working on the RLM-Claude-Code project.

## Quick Start

```bash
# Setup (smart: downloads or builds based on release availability)
npm install

# Verify installation
npm run verify

# Type check
uv run ty check src/

# Test (3000+ tests)
npm run test:full

# Install as plugin
claude plugins install . --scope user
```

## Project Structure

```
rlm-claude-code/
├── CLAUDE.md                       # You are here (developer context)
├── README.md                       # User-facing overview
├── docs/
│   ├── getting-started.md          # Installation guide
│   ├── user-guide.md               # Complete usage docs
│   ├── spec/                       # Capability specifications
│   └── process/                    # Architecture, ADRs, testing
├── src/                            # Python source code
│   ├── orchestrator.py             # Main RLM loop
│   ├── intelligent_orchestrator.py # Claude-powered decisions
│   ├── context_manager.py          # Context externalization
│   ├── repl_environment.py         # Sandboxed Python REPL
│   ├── memory_store.py             # SQLite memory (SPEC-02)
│   ├── memory_evolution.py         # Memory tiers (SPEC-03)
│   ├── reasoning_traces.py         # Decision trees (SPEC-04)
│   ├── enhanced_budget.py          # Cost tracking (SPEC-05)
│   └── ...
├── tests/                          # Test suite
├── scripts/npm/                    # TypeScript npm scripts
│   ├── ensure-setup.ts             # Self-healing setup
│   ├── hook-dispatch.ts            # Cross-platform hook dispatcher
│   ├── download-binaries.ts        # Download Go binaries
│   └── download-wheel.ts           # Download Python wheel
├── hooks/                          # hooks.json (Claude Code hooks)
└── commands/                       # Slash commands
```

## Essential Context

**Read before making changes:**

1. `README.md` — Architecture overview
2. `docs/spec/00-overview.md` — Capability specifications
3. `docs/process/architecture.md` — Design decisions (ADRs)

## Development Commands

```bash
# Setup
npm install             # Smart setup (download or build)
npm run ensure-setup    # Check and auto-fix dependencies
npm run ensure-setup -- --check  # Check only, don't fix
npm run verify          # Verify installation

# Testing
npm run test            # Run smoke tests
npm run test:full       # Run full test suite (3000+ tests)
npm run test:npm        # Run TypeScript tests for npm scripts

# Building
npm run build           # Build from source (needs Rust + Go)
npm run rebuild         # Clean + build from source

# Python tools
uv run ty check src/    # Type check (must pass)
uv run ruff check src/ --fix  # Lint (must pass)
uv run ruff format src/       # Format

# Direct pytest
uv run pytest tests/ -v       # All tests
uv run pytest tests/unit/ -v  # Unit tests only
```

## Key Technologies

| Tool | Purpose |
|------|---------|
| uv | Package management |
| ty | Type checking |
| ruff | Linting/formatting |
| pydantic | Data validation |
| hypothesis | Property testing |
| RestrictedPython | REPL sandbox |
| SQLite | Memory persistence |

## Code Style

- Type annotations on all public functions
- Google-style docstrings with spec references
- No functions >50 lines
- Pydantic models at API boundaries

## Before Committing

1. `uv run ty check src/` — Must pass
2. `uv run ruff check src/` — Must pass
3. `uv run pytest tests/ -v` — Must pass
4. `npm run test:npm` — Must pass (npm scripts)

## Contributing

### Commit Message Format

Use conventional commits:

```
<type>(<scope>): <description>

[optional body]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### PR Checklist

- [ ] Type check passes: `uv run ty check src/`
- [ ] Lint passes: `uv run ruff check src/`
- [ ] Tests pass: `uv run pytest tests/ -v`
- [ ] NPM tests pass: `npm run test:npm`
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional format

## Self-Healing Setup System

The plugin includes a single smart entry point for dependency management:

### ensure-setup.ts

**What it does:**
1. Reads version from `marketplace.json`
2. Checks GitHub for matching release
3. If release exists: downloads binaries + wheel
4. If no release: builds from source
5. Sets up Python venv and dependencies

**Usage:**
```bash
npm run ensure-setup            # Auto-fix (download or build)
npm run ensure-setup -- --check # Check only, don't fix
npm run ensure-setup -- --json  # JSON output (for hooks)
```

### Components

| File | Purpose |
|------|---------|
| `scripts/npm/ensure-setup.ts` | Smart setup (check, download, build) |
| `scripts/npm/hook-dispatch.ts` | Cross-platform hook dispatcher |
| `scripts/npm/build.ts` | Build from source |
| `scripts/npm/verify.ts` | Verify installation |
| `hooks/hooks.json` | Hook configuration |

### Installation Modes

| Mode | Detection | Behavior |
|------|-----------|----------|
| marketplace | No `.git`, no `/dev/` in path | Download from GitHub releases |
| dev | `.git` exists or `/dev/` in path | Build from source |

## Implementation Status

| Spec | Component | Status |
|------|-----------|--------|
| SPEC-01 | Advanced REPL Functions | Complete |
| SPEC-02 | Memory Foundation | Complete |
| SPEC-03 | Memory Evolution | Complete |
| SPEC-04 | Reasoning Traces | Complete |
| SPEC-05 | Enhanced Budget Tracking | Complete |

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601v1)
- [README](./README.md) — User overview
- [Getting Started](./docs/getting-started.md) — Installation
- [User Guide](./docs/user-guide.md) — Usage details
- [SPEC Overview](./docs/spec/00-overview.md) — Specifications
