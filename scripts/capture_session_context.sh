#!/usr/bin/env bash
# Capture session context from Claude Code hooks
# Called by: hooks/hooks.json SessionStart
#
# Hook input (JSON via stdin):
#   - session_id: Unique session identifier
#   - transcript_path: Path to conversation transcript (JSONL)
#   - cwd: Current working directory
#
# Outputs:
#   - ~/.claude/rlm-state/session-metadata.json
#   - ~/.claude/rlm-state/current-transcript.jsonl (symlink)

set -euo pipefail

STATE_DIR="${HOME}/.claude/rlm-state"
mkdir -p "${STATE_DIR}"

# Read hook input from stdin
HOOK_INPUT=$(cat)
if [[ -z "${HOOK_INPUT}" ]]; then
    echo '{"status": "error", "reason": "No stdin input"}' >&2
    exit 1
fi

# Extract fields using jq (or fallback to grep)
if command -v jq &>/dev/null; then
    SESSION_ID=$(echo "${HOOK_INPUT}" | jq -r '.session_id // "default"')
    TRANSCRIPT_PATH=$(echo "${HOOK_INPUT}" | jq -r '.transcript_path // ""')
    CWD=$(echo "${HOOK_INPUT}" | jq -r '.cwd // ""')
else
    # Fallback without jq - basic extraction
    SESSION_ID="default"
    TRANSCRIPT_PATH=""
    CWD="${PWD}"
fi

# Save session metadata
METADATA_FILE="${STATE_DIR}/session-metadata.json"
cat > "${METADATA_FILE}" <<EOF
{
  "session_id": "${SESSION_ID}",
  "transcript_path": "${TRANSCRIPT_PATH}",
  "cwd": "${CWD}",
  "started_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

# Export to CLAUDE_ENV_FILE for subsequent Bash commands
if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
    echo "export CLAUDE_SESSION_ID='${SESSION_ID}'" >> "${CLAUDE_ENV_FILE}"
    echo "export CLAUDE_TRANSCRIPT_PATH='${TRANSCRIPT_PATH}'" >> "${CLAUDE_ENV_FILE}"
fi

# Create symlink to transcript for easy access
if [[ -n "${TRANSCRIPT_PATH}" && -f "${TRANSCRIPT_PATH}" ]]; then
    SYMLINK="${STATE_DIR}/current-transcript.jsonl"
    rm -f "${SYMLINK}" 2>/dev/null || true
    ln -s "${TRANSCRIPT_PATH}" "${SYMLINK}"
fi

# Output result for hook status
echo "{\"status\": \"captured\", \"session_id\": \"${SESSION_ID}\", \"transcript_path\": \"${TRANSCRIPT_PATH}\"}"
