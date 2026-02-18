#!/bin/bash
# Initialize RLM environment and load config
# Used by: hooks.json (SessionStart)
#
# This hook receives JSON input via stdin with session_id and transcript_path
# We pass stdin through to the Python script for parsing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Initializing RLM environment..."

# Pass stdin through to Python script
run_python scripts/init_rlm.py
exit $?
