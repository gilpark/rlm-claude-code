#!/bin/bash
# Save RLM trajectory on session end
# Used by: hooks.json (Stop)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Saving trajectory..."

run_python scripts/save_trajectory.py
exit $?
