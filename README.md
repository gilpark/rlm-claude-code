# CausalFrame

> Causal awareness for Claude Code — externalize reasoning, evolve with your environment.

## The Problem

Every Claude session starts cold. Prior conclusions cannot be trusted because there is no way to know if the premises that produced them still hold. The AI cannot say "I analyzed this before, but auth.py changed since then, so my prior analysis may be invalid."

This is reasoning amnesia. And it is an architecture problem, not a hardware problem.

## The Solution

CausalFrame implements two mechanisms that work together:

### 1. REPL — Spatial Externalization
The model doesn't receive context passively — it navigates it actively:
```python
# Model peeks, searches, loads only what this sub-task needs
result = llm("analyze auth flow", context={"files": ["auth.py"]})
```

### 2. CausalFrame — Temporal Persistence
Store why conclusions were reached, not just what was concluded:
```python
CausalFrame(
    query="auth 모듈 분석",
    conclusion="JWT expiry는 24시간...",
    evidence=["auth.py line 47"],
    invalidation_condition="auth.py modified",
    status=FrameStatus.PROMOTED
)
```

When `auth.py` changes, CausalFrame:
1. Finds all frames where `auth.py` was in the context
2. Marks those frames invalidated
3. Cascades to dependent frames
4. Surfaces: "these conclusions may no longer hold"

## Installation

```bash
# Clone the plugin
git clone https://github.com/rand/causalframe.git ~/.claude/plugins/marketplaces/causalframe

# Install dependencies
cd ~/.claude/plugins/marketplaces/causalframe
uv sync
```

## Usage

### As a Claude Code Plugin

Add to your `~/.claude/settings.json`:
```json
{
  "plugins": ["~/.claude/plugins/marketplaces/causalframe"]
}
```

### Skills

- `/causalframe` - Activate causal mode for complex reasoning
- `/verification` - Constraint verification for proposed changes

### Hooks

CausalFrame hooks into Claude Code's lifecycle:

| Hook | Purpose |
|------|---------|
| `SessionStart` | Compare with prior session, surface invalidated frames |
| `PostToolUse` | Capture tool outputs into active CausalFrame |
| `Stop` | Extract frame tree, save to FrameStore |

## Architecture

```
src/
├── rlaph_loop.py          # Immediate execution loop
├── repl_environment.py    # peek, search, llm functions
├── llm_client.py          # Provider-agnostic LLM calls
├── causal_frame.py        # CausalFrame, FrameStatus
├── frame_store.py         # JSONL persistence
├── frame_invalidation.py  # Cascade invalidation
└── session_comparison.py  # Cross-session diff
```

**18 src files. Zero SQL dependencies. Human-readable JSONL.**

## The Vision

A developer says: "auth 쪽 어떻게 됐어?"

The AI responds with a current picture:
- Prior analysis: JWT expiry issue, confidence 0.9, auth.py line 47
- auth.py was modified yesterday — that analysis is invalidated
- DB hypothesis was explored and suspended — still resumable
- Related documentation needs updating

Not because the context window is large enough to hold everything — but because the reasoning structure makes the right connections automatically.

## References

- [Whitepaper](docs/WHITEPAPER.md) - "Externalizing Causation: Toward AI That Evolves With Its Environment"
- [Design Doc](docs/DESIGN.md) - Architecture and implementation details
- [Zhang et al., "Recursive Language Models" (2025)](https://arxiv.org/abs/2512.24601v1) - The foundation

## License

MIT
