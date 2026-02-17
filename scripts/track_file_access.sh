#!/usr/bin/env bash
# Track file access for RLM context
# Called by: hooks/hooks.json PostToolUse
#
# Hook input (JSON via stdin):
#   - tool_name: Name of tool (Read, Edit, Write, Glob, Grep)
#   - tool_input: Tool parameters
#   - tool_response: Tool output
#
# Outputs:
#   - ~/.claude/rlm-state/file-access-${SESSION_ID}.jsonl

set -euo pipefail

STATE_DIR="${HOME}/.claude/rlm-state"
mkdir -p "${STATE_DIR}"

# Read hook input from stdin
HOOK_INPUT=$(cat)
if [[ -z "${HOOK_INPUT}" ]]; then
    echo '{"status": "skipped", "reason": "No stdin input"}'
    exit 0
fi

# Check for jq availability
if ! command -v jq &>/dev/null; then
    echo '{"status": "skipped", "reason": "jq not available"}'
    exit 0
fi

# Extract tool info
TOOL_NAME=$(echo "${HOOK_INPUT}" | jq -r '.tool_name // ""')

# Load session metadata for session_id
METADATA_FILE="${STATE_DIR}/session-metadata.json"
SESSION_ID="default"
if [[ -f "${METADATA_FILE}" ]]; then
    SESSION_ID=$(jq -r '.session_id // "default"' "${METADATA_FILE}")
fi

# Log file access
ACCESS_LOG="${STATE_DIR}/file-access-${SESSION_ID}.jsonl"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

case "${TOOL_NAME}" in
    Read)
        FILE_PATH=$(echo "${HOOK_INPUT}" | jq -r '.tool_input.file_path // ""')
        if [[ -n "${FILE_PATH}" && "${FILE_PATH}" != "null" ]]; then
            echo "{\"timestamp\": \"${TIMESTAMP}\", \"tool\": \"Read\", \"file\": \"${FILE_PATH}\"}" >> "${ACCESS_LOG}"
        fi
        ;;
    Edit|Write)
        FILE_PATH=$(echo "${HOOK_INPUT}" | jq -r '.tool_input.file_path // ""')
        if [[ -n "${FILE_PATH}" && "${FILE_PATH}" != "null" ]]; then
            echo "{\"timestamp\": \"${TIMESTAMP}\", \"tool\": \"${TOOL_NAME}\", \"file\": \"${FILE_PATH}\"}" >> "${ACCESS_LOG}"
        fi
        ;;
    Glob)
        PATTERN=$(echo "${HOOK_INPUT}" | jq -r '.tool_input.pattern // ""')
        if [[ -n "${PATTERN}" && "${PATTERN}" != "null" ]]; then
            echo "{\"timestamp\": \"${TIMESTAMP}\", \"tool\": \"Glob\", \"pattern\": \"${PATTERN}\"}" >> "${ACCESS_LOG}"
        fi
        ;;
    Grep)
        PATTERN=$(echo "${HOOK_INPUT}" | jq -r '.tool_input.pattern // ""')
        if [[ -n "${PATTERN}" && "${PATTERN}" != "null" ]]; then
            echo "{\"timestamp\": \"${TIMESTAMP}\", \"tool\": \"Grep\", \"pattern\": \"${PATTERN}\"}" >> "${ACCESS_LOG}"
        fi
        ;;
esac

echo "{\"status\": \"tracked\", \"tool\": \"${TOOL_NAME}\"}"
