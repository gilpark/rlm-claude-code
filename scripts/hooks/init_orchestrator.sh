#!/bin/bash
# Initialize RLM context when orchestrator agent starts
# Used by: agents/rlm-orchestrator.md (SubagentStart)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Initializing RLM context from session state..."

if ! run_python scripts/run_orchestrator.py --init; then
    log_error "Failed to initialize RLM context"
    exit 2  # Block execution
fi

log_success "RLM context initialized"
exit 0
