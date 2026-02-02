# Changelog

## [0.6.0] - 2026-02-02

### Added
- Go hook binaries replacing Python scripts (~5ms vs ~500ms startup)
- Cross-plugin event system (`~/.claude/events/`) for DP↔RLM coordination
- JSON Schema definitions for all event types
- Python event emission/consumption helpers (`src/events/`)
- Version-aware config migration (V1→V2) preserving user customizations
- Platform-aware hook dispatcher with fallback chain
- GitHub Actions CI for cross-compilation (5 platforms)
- Unit tests for hookio, events, and config packages

### Changed
- hooks.json now uses Go binaries + prompt-based hooks
- Complexity check responds to all DP phases (not just spec/review)
- RLM orchestrator agent informed about parallel tool call behavior

### Deprecated
- Python hook scripts moved to `scripts/legacy/`
- Set `RLM_USE_LEGACY_HOOKS=1` to use legacy Python hooks
