#!/bin/bash
# Externalize context before compaction
# Used by: hooks.json (PreCompact)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_plugin_root

log_debug "Externalizing context..."

run_python scripts/externalize_context.py
exit $?
