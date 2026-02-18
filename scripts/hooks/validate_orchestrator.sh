#!/bin/bash
# Validate orchestrator dependencies before running
# Used by: commands/rlm-orchestrator.md (PreToolUse)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Validating orchestrator dependencies..."

if ! run_python scripts/run_orchestrator.py --validate; then
    log_error "Orchestrator validation failed"
    exit 2  # Block execution
fi

log_debug "Orchestrator validation passed"
exit 0
