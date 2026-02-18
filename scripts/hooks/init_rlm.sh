#!/bin/bash
# Initialize RLM environment and load config
# Used by: hooks.json (SessionStart)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Initializing RLM environment..."

run_python scripts/init_rlm.py
exit $?
