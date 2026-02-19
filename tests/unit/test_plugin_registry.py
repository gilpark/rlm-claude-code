"""Tests for plugin registry."""

from src.plugin_interface import CoreContext, RLMPlugin
from src.plugin_registry import PluginRegistry


class MockPlugin:
    """Mock plugin for testing."""

    def transform_input(self, raw_input, ctx: CoreContext):
        return {"transformed": True}

    def parse_output(self, raw_output, frame):
        return {"parsed": True}

    def store(self, parsed, frame):
        pass


def test_plugin_registry_register_and_get():
    """Registry should store and retrieve plugins."""
    registry = PluginRegistry()
    plugin = MockPlugin()

    registry.register("mock", plugin)

    assert registry.get("mock") == plugin
    assert registry.get("nonexistent") is None


def test_plugin_registry_get_default():
    """Registry should return the default plugin."""
    registry = PluginRegistry()
    plugin = MockPlugin()
    registry.register("default", plugin)

    assert registry.get_default() == plugin


def test_plugin_registry_list_plugins():
    """Registry should list all registered plugins."""
    registry = PluginRegistry()
    registry.register("default", MockPlugin())
    registry.register("custom", MockPlugin())

    names = registry.list_plugins()
    assert "default" in names
    assert "custom" in names
    assert len(names) == 2


def test_plugin_registry_contains():
    """Registry should support 'in' operator."""
    registry = PluginRegistry()
    registry.register("mock", MockPlugin())

    assert "mock" in registry
    assert "nonexistent" not in registry
