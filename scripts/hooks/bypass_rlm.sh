#!/bin/bash
# Set RLM bypass flag for simple operations
# Used by: commands/simple.md (PreToolUse)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Setting RLM bypass flag..."

run_python scripts/run_orchestrator.py --bypass
exit $?
