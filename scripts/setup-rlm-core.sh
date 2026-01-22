#!/bin/bash
# Setup rlm_core for the currently installed rlm-claude-code plugin
# Run this after installing/updating the plugin if you want rlm_core performance

set -e

# Find the latest installed plugin version
PLUGIN_BASE=~/.claude/plugins/cache/rlm-claude-code/rlm-claude-code
if [ ! -d "$PLUGIN_BASE" ]; then
    echo "Error: rlm-claude-code plugin not installed"
    exit 1
fi

LATEST_VERSION=$(ls "$PLUGIN_BASE" | sort -V | tail -1)
PLUGIN_DIR="$PLUGIN_BASE/$LATEST_VERSION"

echo "Setting up rlm_core for rlm-claude-code v$LATEST_VERSION"

# Check if rlm-core exists
RLM_CORE_PATH="$HOME/src/loop/rlm-core/python"
if [ ! -d "$RLM_CORE_PATH" ]; then
    echo "Warning: rlm-core not found at $RLM_CORE_PATH"
    echo "Clone it with: git clone https://github.com/rand/loop.git ~/src/loop"
    echo "Then build: cd ~/src/loop/rlm-core && maturin develop --release"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$PLUGIN_DIR/.venv" ]; then
    echo "Creating venv..."
    cd "$PLUGIN_DIR" && uv sync
fi

# Link rlm_core
SITE_PACKAGES="$PLUGIN_DIR/.venv/lib/python3.12/site-packages"
if [ ! -d "$SITE_PACKAGES" ]; then
    # Try python3.13 or find the right version
    SITE_PACKAGES=$(find "$PLUGIN_DIR/.venv/lib" -name "site-packages" -type d | head -1)
fi

if [ -z "$SITE_PACKAGES" ]; then
    echo "Error: Could not find site-packages directory"
    exit 1
fi

echo "$RLM_CORE_PATH" > "$SITE_PACKAGES/rlm_core.pth"
echo "Created: $SITE_PACKAGES/rlm_core.pth"

# Verify it works
echo "Verifying rlm_core import..."
"$PLUGIN_DIR/.venv/bin/python" -c "import rlm_core; print('✓ rlm_core loaded:', rlm_core.__file__)"

# Remind about config
echo ""
echo "Done! Make sure use_rlm_core is enabled in ~/.claude/rlm-config.json:"
echo '  "use_rlm_core": true'
