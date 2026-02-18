#!/bin/bash
# Common functions for RLM hook scripts
# Source this file: source "${SCRIPT_DIR}/common.sh"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo "[RLM] $1" >&2
}

log_debug() {
    if [[ "${RLM_DEBUG:-}" == "1" ]]; then
        echo -e "${BLUE}[RLM DEBUG]${NC} $1" >&2
    fi
}

log_success() {
    echo -e "${GREEN}[RLM]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[RLM WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[RLM ERROR]${NC} $1" >&2
}

# Ensure we have the plugin root
ensure_plugin_root() {
    if [[ -z "${CLAUDE_PLUGIN_ROOT:-}" ]]; then
        # Try to detect from script location
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        CLAUDE_PLUGIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
        export CLAUDE_PLUGIN_ROOT
        log_debug "CLAUDE_PLUGIN_ROOT not set, detected: ${CLAUDE_PLUGIN_ROOT}"
    fi
}

# Check if venv exists
check_venv() {
    if [[ ! -d "${CLAUDE_PLUGIN_ROOT}/.venv" ]]; then
        log_error "Virtual environment not found at ${CLAUDE_PLUGIN_ROOT}/.venv"
        log_error "Run: cd ${CLAUDE_PLUGIN_ROOT} && uv sync"
        return 1
    fi
}

# Check if Python script exists
check_script() {
    local script="$1"
    if [[ ! -f "${CLAUDE_PLUGIN_ROOT}/${script}" ]]; then
        log_error "Script not found: ${CLAUDE_PLUGIN_ROOT}/${script}"
        return 1
    fi
}

# Run a Python script with the venv Python
run_python() {
    local script="$1"
    shift
    check_venv
    check_script "$script"
    log_debug "Running: ${CLAUDE_PLUGIN_ROOT}/.venv/bin/python ${CLAUDE_PLUGIN_ROOT}/${script} $*"
    cd "${CLAUDE_PLUGIN_ROOT}"
    "${CLAUDE_PLUGIN_ROOT}/.venv/bin/python" "${CLAUDE_PLUGIN_ROOT}/${script}" "$@"
}
