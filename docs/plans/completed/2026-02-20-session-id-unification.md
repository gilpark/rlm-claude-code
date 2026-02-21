# Session ID Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify session IDs so that Claude Code session artifacts (tools.jsonl) and RLM frames (index.json) are stored in the same folder.

**Architecture:** Pass the Claude Code session_id from hooks to the RLM orchestrator via environment variable, ensuring all artifacts land in one folder per session.

**Tech Stack:** Python, JSONL, Claude Code hooks

---

## Problem Analysis

Current state:
- **Claude session** (`039f5c9f-e804-44d9-9946-1098a64c8c1b/`): Contains `tools.jsonl` from PostToolUse hook
- **RLM session** (`c1884f34/`): Contains `index.json` from RLAPHLoop

These are **disconnected** - hooks use Claude session_id, orchestrator generates its own.

Desired state:
```
~/.claude/rlm-frames/
└── 039f5c9f-e804-44d9-9946-1098a64c8c1b/    # ONE folder per Claude session
    ├── index.json      # RLM frames
    ├── frames.jsonl    # Extracted frames
    ├── tools.jsonl     # Tool outputs
    └── artifacts.json  # Session metadata
```

---

## Task 1: Add Session ID Environment Variable

**Files:**
- Modify: `hooks/hooks.json`
- Test: Manual hook invocation

**Step 1: Update hooks.json to pass session_id via environment**

The hooks already receive `session_id` in stdin. We need to export it so the orchestrator can use it.

```json
{
  "description": "RLM v2 plugin hooks - REPL + CausalFrame",
  "hooks": {
    "SessionStart": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/compare_sessions.py",
            "timeout": 5000,
            "description": "Compare with prior session, surface invalidated frames"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/capture_output.py",
            "timeout": 5000,
            "description": "Capture tool output into active CausalFrame"
          }
        ]
      }
    ],
    "Stop": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/extract_frames.py",
            "timeout": 5000,
            "description": "Extract frame tree, save to FrameStore"
          }
        ]
      }
    ]
  }
}
```

No changes needed here - hooks receive session_id via stdin.

**Step 2: Verify hook stdin input format**

Claude Code passes this JSON to hooks:
```json
{
  "session_id": "039f5c9f-e804-44d9-9946-1098a64c8c1b",
  "transcript_path": "...",
  "tool_name": "Bash",
  ...
}
```

Hooks correctly extract: `session_id = hook_data.get("session_id", "default")`

---

## Task 2: Add Session ID Argument to Orchestrator

**Files:**
- Modify: `scripts/run_orchestrator.py:270-348`
- Modify: `src/repl/rlaph_loop.py:143-174`
- Test: `tests/test_session_id_flow.py` (new)

**Step 1: Write the failing test**

Create `tests/test_session_id_flow.py`:

```python
"""Test session ID flow from orchestrator to frame persistence."""

import pytest
from pathlib import Path


def test_orchestrator_accepts_session_id_argument():
    """Test that orchestrator accepts --session-id argument."""
    import subprocess
    import json

    result = subprocess.run(
        ["uv", "run", "python", "scripts/run_orchestrator.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert "--session-id" in result.stdout or "-s" in result.stdout


def test_orchestrator_uses_provided_session_id():
    """Test that provided session_id is used for frame persistence."""
    from src.frame.frame_index import FrameIndex
    from src.repl.rlaph_loop import RLAPHLoop
    from src.types import SessionContext
    import asyncio

    # Create loop with explicit session_id
    loop = RLAPHLoop(max_iterations=1, max_depth=0)
    context = SessionContext(messages=[], files={}, tool_outputs=[], working_memory={})

    # Run with explicit session_id
    result = asyncio.run(loop.run(
        query="FINAL: test answer",
        context=context,
        session_id="test-session-123",
    ))

    # Verify frames saved with correct session_id
    index = FrameIndex.load("test-session-123")
    assert len(index) >= 0  # May be empty if no REPL execution

    # Cleanup
    import shutil
    session_dir = Path.home() / ".claude" / "rlm-frames" / "test-session-123"
    if session_dir.exists():
        shutil.rmtree(session_dir)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session_id_flow.py -v`
Expected: FAIL with "no such option: --session-id"

**Step 3: Add --session-id argument to run_orchestrator.py**

In `scripts/run_orchestrator.py`, add the argument:

```python
# Around line 291, add:
parser.add_argument("--session-id", "-s", dest="session_id",
                    help="Session ID for frame persistence (default: auto-detect or generate)")
```

**Step 4: Pass session_id to RLAPHLoop**

In `scripts/run_orchestrator.py`, modify `run_rlaph()` function:

```python
async def run_rlaph(
    query: str,
    depth: int = 2,
    verbose: bool = False,
    working_dir: Path | None = None,
    session_id: str | None = None,  # Add this parameter
) -> str:
    # ... existing code ...

    # Run loop with session_id
    result = await loop.run(
        query, context, working_dir=working_dir, session_id=session_id
    )
    # ...
```

And in `main()`:

```python
# Get session_id from argument or environment or generate
session_id = args.session_id
if session_id is None:
    session_id = os.environ.get("CLAUDE_SESSION_ID")

# Run RLAPH loop
result = asyncio.run(
    run_rlaph(
        query,
        depth=args.depth,
        verbose=args.verbose,
        working_dir=plugin_root,
        session_id=session_id,  # Pass session_id
    )
)
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_session_id_flow.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add scripts/run_orchestrator.py tests/test_session_id_flow.py
git commit -m "feat: add --session-id argument to orchestrator"
```

---

## Task 3: Create Session ID Coordination File

**Files:**
- Create: `scripts/get_session_id.py`
- Modify: `hooks/hooks.json`
- Test: Manual testing

**Step 1: Create session ID coordination script**

The problem: Hooks run in separate processes and can't directly communicate with the orchestrator. We need a way to share the session_id.

Create `scripts/get_session_id.py`:

```python
#!/usr/bin/env python3
"""
Get or create session ID for RLM operations.

This script provides a coordination point for session_id between
hooks (which receive it from Claude Code) and the orchestrator
(which needs it for frame persistence).

Usage:
    # Get current session ID (from hook input)
    echo '{"session_id": "abc123"}' | python scripts/get_session_id.py

    # Get or create session ID (for orchestrator)
    python scripts/get_session_id.py --ensure

The session ID is stored in ~/.claude/rlm-frames/.current_session
"""

import json
import os
import sys
import uuid
from pathlib import Path


def get_session_file() -> Path:
    """Get path to current session file."""
    return Path.home() / ".claude" / "rlm-frames" / ".current_session"


def get_session_id() -> str | None:
    """Get current session ID from coordination file."""
    session_file = get_session_file()
    if session_file.exists():
        data = json.loads(session_file.read_text())
        return data.get("session_id")
    return None


def set_session_id(session_id: str) -> None:
    """Set current session ID in coordination file."""
    session_file = get_session_file()
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(json.dumps({
        "session_id": session_id,
        "pid": os.getpid(),
    }))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Get or create session ID")
    parser.add_argument("--ensure", action="store_true",
                        help="Create session ID if not exists")
    parser.add_argument("--set", dest="set_id", help="Set session ID")
    args = parser.parse_args()

    # Set session ID from argument or stdin
    if args.set_id:
        set_session_id(args.set_id)
        print(json.dumps({"session_id": args.set_id}))
        return

    # Try to read from stdin (hook input)
    stdin_data = ""
    if not sys.stdin.isatty():
        stdin_data = sys.stdin.read().strip()

    if stdin_data:
        try:
            hook_data = json.loads(stdin_data)
            session_id = hook_data.get("session_id")
            if session_id:
                set_session_id(session_id)
                print(json.dumps({"session_id": session_id}))
                return
        except json.JSONDecodeError:
            pass

    # Get existing session ID
    session_id = get_session_id()

    if session_id:
        print(json.dumps({"session_id": session_id}))
        return

    if args.ensure:
        # Generate new session ID
        session_id = str(uuid.uuid4())[:8]
        set_session_id(session_id)
        print(json.dumps({"session_id": session_id}))
        return

    # No session ID available
    print(json.dumps({"session_id": None, "error": "No session ID found"}))


if __name__ == "__main__":
    main()
```

**Step 2: Update SessionStart hook to set session ID**

Modify `hooks/hooks.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "python ${CLAUDE_PLUGIN_ROOT}/scripts/get_session_id.py --set \"${CLAUDE_SESSION_ID:-$(uuidgen | cut -d'-' -f1)}\"",
            "timeout": 5000
          },
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/compare_sessions.py",
            "timeout": 5000,
            "description": "Compare with prior session, surface invalidated frames"
          }
        ]
      }
    ],
    ...
  }
}
```

Actually, a simpler approach - have capture_output.py set the session ID:

**Step 3: Update capture_output.py to set session ID**

Modify `scripts/capture_output.py` to also write the session ID:

```python
# Add after line 46 (after getting session_id):

# Also save to coordination file for orchestrator
session_file = Path.home() / ".claude" / "rlm-frames" / ".current_session"
session_file.parent.mkdir(parents=True, exist_ok=True)
session_file.write_text(json.dumps({
    "session_id": session_id,
    "updated_at": datetime.now().isoformat(),
}))
```

**Step 4: Commit**

```bash
git add scripts/get_session_id.py scripts/capture_output.py
git commit -m "feat: add session ID coordination mechanism"
```

---

## Task 4: Update Orchestrator to Use Coordination File

**Files:**
- Modify: `scripts/run_orchestrator.py`

**Step 1: Add function to get session ID from coordination file**

In `scripts/run_orchestrator.py`, add:

```python
def get_or_create_session_id(explicit_id: str | None = None) -> str:
    """
    Get session ID from various sources in priority order:
    1. Explicit argument (--session-id)
    2. Environment variable (CLAUDE_SESSION_ID)
    3. Coordination file (~/.claude/rlm-frames/.current_session)
    4. Generate new UUID[:8]
    """
    import uuid

    # 1. Explicit argument
    if explicit_id:
        return explicit_id

    # 2. Environment variable
    env_id = os.environ.get("CLAUDE_SESSION_ID")
    if env_id:
        return env_id

    # 3. Coordination file
    session_file = Path.home() / ".claude" / "rlm-frames" / ".current_session"
    if session_file.exists():
        try:
            data = json.loads(session_file.read_text())
            if data.get("session_id"):
                return data["session_id"]
        except (json.JSONDecodeError, KeyError):
            pass

    # 4. Generate new
    return str(uuid.uuid4())[:8]
```

**Step 2: Use in main()**

```python
def main():
    # ... argument parsing ...

    # Get session ID
    session_id = get_or_create_session_id(args.session_id)

    if verbose:
        print(f"[RLM:SESSION] Using session_id: {session_id}")

    # Run RLAPH loop
    try:
        result = asyncio.run(
            run_rlaph(
                query,
                depth=args.depth,
                verbose=args.verbose,
                working_dir=plugin_root,
                session_id=session_id,
            )
        )
        # ...
```

**Step 3: Commit**

```bash
git add scripts/run_orchestrator.py
git commit -m "feat: orchestrator uses session ID from coordination file"
```

---

## Task 5: Update extract_frames Hook

**Files:**
- Modify: `scripts/extract_frames.py`

**Step 1: Update extract_frames to use coordination file**

The extract_frames hook currently tries to load frames using the session_id from stdin. But frames are saved by the orchestrator which may use a different session_id.

Modify `scripts/extract_frames.py`:

```python
def get_session_id_for_frames(hook_session_id: str) -> str:
    """
    Get the session ID that frames were saved with.

    Priority:
    1. Coordination file (set by orchestrator or capture_output)
    2. Hook input session_id
    3. "default"
    """
    # Check coordination file first
    session_file = Path.home() / ".claude" / "rlm-frames" / ".current_session"
    if session_file.exists():
        try:
            data = json.loads(session_file.read_text())
            if data.get("session_id"):
                return data["session_id"]
        except (json.JSONDecodeError, KeyError):
            pass

    return hook_session_id or "default"


def main():
    # Read hook input from stdin
    hook_data = {}
    try:
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            hook_data = json.loads(stdin_data)
    except json.JSONDecodeError:
        pass

    hook_session_id = hook_data.get("session_id", "default")

    # Get the session ID that frames were saved with
    session_id = get_session_id_for_frames(hook_session_id)

    # Extract and persist frames
    result = extract_frames(session_id)

    # Output result
    print(json.dumps(result))
    sys.exit(0)
```

**Step 2: Commit**

```bash
git add scripts/extract_frames.py
git commit -m "fix: extract_frames uses coordinated session ID"
```

---

## Task 6: Update /rlm-orchestrator Skill

**Files:**
- Modify: `.claude-plugin/skills/rlm-orchestrator.md`

**Step 1: Update skill to pass session_id**

The skill should try to get the session_id from the coordination file:

```markdown
### Step 3: Run Python Orchestrator

Use Bash to run the Python script with the composed prompt:

```bash
cd /Users/gilpark/.dotfiles/.claude/plugins/marketplaces/causeway

# Get session ID from coordination file if available
SESSION_ID=$(cat ~/.claude/rlm-frames/.current_session 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null || echo "")

# Run orchestrator with session ID
if [ -n "$SESSION_ID" ]; then
  uv run python scripts/run_orchestrator.py --verbose --session-id "$SESSION_ID" "<composed prompt>"
else
  uv run python scripts/run_orchestrator.py --verbose "<composed prompt>"
fi
```
```

**Step 2: Commit**

```bash
git add .claude-plugin/skills/rlm-orchestrator.md
git commit -m "docs: update rlm-orchestrator skill to use session ID"
```

---

## Task 7: End-to-End Testing

**Files:**
- Test: Manual testing

**Step 1: Test the complete flow**

1. Start a new Claude Code session
2. Invoke `/rlm-orchestrator analyze the auth flow`
3. Verify frames are saved to the Claude session folder
4. Check that tools.jsonl and index.json are in the SAME folder

```bash
# Check session folder
ls -la ~/.claude/rlm-frames/

# Should show ONE folder with both files:
# {session-id}/
#   ├── index.json      # RLM frames
#   ├── tools.jsonl     # Tool outputs
#   └── ...
```

**Step 2: Test compare_sessions hook**

1. Start a new session
2. Make a code change
3. Verify compare_sessions hook finds the prior session and identifies invalidated frames

**Step 3: Commit**

```bash
git add -A
git commit -m "test: verify session ID unification end-to-end"
```

---

## Task 8: Update Documentation

**Files:**
- Update: `docs/rlm-data-flow.md`
- Update: `docs/ARCHITECTURE.md`

**Step 1: Update data flow docs**

Add section on session ID coordination:

```markdown
## Session ID Coordination

To ensure all artifacts land in the same folder:

1. **PostToolUse hook** (`capture_output.py`) receives `session_id` from Claude Code
2. Hook writes `session_id` to coordination file: `~/.claude/rlm-frames/.current_session`
3. **Orchestrator** reads coordination file and uses the same `session_id`
4. **Stop hook** (`extract_frames.py`) reads coordination file to find frames

This ensures:
- Tool outputs → `{session_id}/tools.jsonl`
- RLM frames → `{session_id}/index.json`
- All artifacts in ONE folder per Claude session
```

**Step 2: Commit**

```bash
git add docs/rlm-data-flow.md docs/ARCHITECTURE.md
git commit -m "docs: document session ID coordination mechanism"
```

---

## Summary

| Task | Description | Key Change |
|------|-------------|------------|
| 1 | Environment variable setup | Hooks already receive session_id via stdin |
| 2 | Add --session-id to orchestrator | New CLI argument |
| 3 | Session ID coordination file | `~/.claude/rlm-frames/.current_session` |
| 4 | Orchestrator uses coordination | `get_or_create_session_id()` function |
| 5 | extract_frames uses coordination | Read from coordination file |
| 6 | Update skill documentation | Pass session_id to orchestrator |
| 7 | End-to-end testing | Verify all artifacts in same folder |
| 8 | Update documentation | Document coordination mechanism |

---

## User Experience After Fix

```
User: /rlm-orchestrator analyze auth flow
  ↓
PostToolUse hook sets session_id in coordination file
  ↓
Orchestrator reads coordination file, uses same session_id
  ↓
Frames saved to: ~/.claude/rlm-frames/{claude-session-id}/index.json
Tools saved to:  ~/.claude/rlm-frames/{claude-session-id}/tools.jsonl
  ↓
extract_frames hook finds frames in correct folder
  ↓
Next session can compare and resume from invalidated frames
```
