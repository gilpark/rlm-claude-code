#!/bin/bash
# Capture tool output for RLM context
# Used by: hooks.json (PostToolUse)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Capturing tool output..."

run_python scripts/capture_output.py
exit $?
