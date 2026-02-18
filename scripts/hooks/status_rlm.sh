#!/bin/bash
# Load current RLM status before displaying configuration
# Used by: commands/rlm.md (PreToolUse)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Loading RLM status..."

run_python scripts/run_orchestrator.py --status
exit $?
