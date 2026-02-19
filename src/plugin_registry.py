"""Plugin registry for loading and managing plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .plugin_interface import RLMPlugin


class PluginRegistry:
    """
    Load and manage plugins.

    Plugins are registered by name and retrieved by name.
    The "default" plugin is used when no specific plugin is requested.
    """

    def __init__(self):
        self._plugins: dict[str, "RLMPlugin"] = {}

    def register(self, name: str, plugin: "RLMPlugin") -> None:
        """Register a plugin by name."""
        self._plugins[name] = plugin

    def get(self, name: str) -> "RLMPlugin | None":
        """Get a plugin by name, or None if not found."""
        return self._plugins.get(name)

    def get_default(self) -> "RLMPlugin | None":
        """Get the default plugin, or None if not registered."""
        return self._plugins.get("default")

    def list_plugins(self) -> list[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._plugins
