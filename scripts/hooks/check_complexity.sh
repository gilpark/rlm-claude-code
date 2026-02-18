#!/bin/bash
# Check if RLM should activate based on task complexity
# Used by: hooks.json (UserPromptSubmit)
# Reads PROMPT from environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Checking complexity for prompt..."

if [[ -z "${PROMPT:-}" ]]; then
    log_debug "No PROMPT provided, skipping complexity check"
    exit 0
fi

run_python scripts/check_complexity.py "$PROMPT"
exit $?
