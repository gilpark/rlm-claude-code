#!/bin/bash
# Sync tool context to RLM state
# Used by: hooks.json (PreToolUse)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Syncing context..."

run_python scripts/sync_context.py
exit $?
