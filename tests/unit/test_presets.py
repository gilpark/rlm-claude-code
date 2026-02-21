"""Tests for pre-configured agent presets."""

import pytest

from src.agents import (
    analyzer_agent,
    debugger_agent,
    security_agent,
    summarizer_agent,
)


class TestAnalyzerAgent:
    """Test suite for analyzer_agent preset."""

    def test_analyzer_agent_exists(self):
        """analyzer_agent should be an RLMSubAgent instance."""
        assert analyzer_agent is not None
        from src.agents.sub_agent import RLMSubAgent
        assert isinstance(analyzer_agent, RLMSubAgent)

    def test_analyzer_agent_config_name(self):
        """analyzer_agent should have correct name."""
        assert analyzer_agent.config.name == "analyzer"

    def test_analyzer_agent_config_system_prompt_override(self):
        """analyzer_agent should have detailed analysis prompt."""
        expected_prompt = (
            "You are a detailed code analyst. "
            "Focus on architecture, structure, and correctness."
        )
        assert analyzer_agent.config.system_prompt_override == expected_prompt

    def test_analyzer_agent_config_max_depth(self):
        """analyzer_agent should have max_depth=4 for detailed analysis."""
        assert analyzer_agent.config.default_max_depth == 4

    def test_analyzer_agent_config_scope(self):
        """analyzer_agent should have overview scope."""
        assert analyzer_agent.config.default_scope == "overview"

    def test_analyzer_agent_config_verbose(self):
        """analyzer_agent should have verbose=False by default."""
        assert analyzer_agent.config.verbose is False


class TestSummarizerAgent:
    """Test suite for summarizer_agent preset."""

    def test_summarizer_agent_exists(self):
        """summarizer_agent should be an RLMSubAgent instance."""
        assert summarizer_agent is not None
        from src.agents.sub_agent import RLMSubAgent
        assert isinstance(summarizer_agent, RLMSubAgent)

    def test_summarizer_agent_config_name(self):
        """summarizer_agent should have correct name."""
        assert summarizer_agent.config.name == "summarizer"

    def test_summarizer_agent_config_system_prompt_override(self):
        """summarizer_agent should have concise summarization prompt."""
        expected_prompt = (
            "You are a concise summarizer. "
            "Keep answers short, structured, and to the point."
        )
        assert summarizer_agent.config.system_prompt_override == expected_prompt

    def test_summarizer_agent_config_max_depth(self):
        """summarizer_agent should have max_depth=2 for concise output."""
        assert summarizer_agent.config.default_max_depth == 2

    def test_summarizer_agent_config_scope(self):
        """summarizer_agent should have overview scope."""
        assert summarizer_agent.config.default_scope == "overview"

    def test_summarizer_agent_config_verbose(self):
        """summarizer_agent should have verbose=False by default."""
        assert summarizer_agent.config.verbose is False


class TestDebuggerAgent:
    """Test suite for debugger_agent preset."""

    def test_debugger_agent_exists(self):
        """debugger_agent should be an RLMSubAgent instance."""
        assert debugger_agent is not None
        from src.agents.sub_agent import RLMSubAgent
        assert isinstance(debugger_agent, RLMSubAgent)

    def test_debugger_agent_config_name(self):
        """debugger_agent should have correct name."""
        assert debugger_agent.config.name == "debugger"

    def test_debugger_agent_config_system_prompt_override(self):
        """debugger_agent should have bug hunting prompt."""
        expected_prompt = (
            "You are a bug hunter. "
            "Focus on reproduction steps, root causes, and fixes."
        )
        assert debugger_agent.config.system_prompt_override == expected_prompt

    def test_debugger_agent_config_max_depth(self):
        """debugger_agent should have max_depth=5 for deep investigation."""
        assert debugger_agent.config.default_max_depth == 5

    def test_debugger_agent_config_scope(self):
        """debugger_agent should have correctness scope."""
        assert debugger_agent.config.default_scope == "correctness"

    def test_debugger_agent_config_verbose(self):
        """debugger_agent should have verbose=False by default."""
        assert debugger_agent.config.verbose is False


class TestSecurityAgent:
    """Test suite for security_agent preset."""

    def test_security_agent_exists(self):
        """security_agent should be an RLMSubAgent instance."""
        assert security_agent is not None
        from src.agents.sub_agent import RLMSubAgent
        assert isinstance(security_agent, RLMSubAgent)

    def test_security_agent_config_name(self):
        """security_agent should have correct name."""
        assert security_agent.config.name == "security"

    def test_security_agent_config_system_prompt_override(self):
        """security_agent should have security auditing prompt."""
        expected_prompt = (
            "You are a security auditor. "
            "Focus on vulnerabilities, attack vectors, and mitigations."
        )
        assert security_agent.config.system_prompt_override == expected_prompt

    def test_security_agent_config_max_depth(self):
        """security_agent should have max_depth=4 for thorough analysis."""
        assert security_agent.config.default_max_depth == 4

    def test_security_agent_config_scope(self):
        """security_agent should have security scope."""
        assert security_agent.config.default_scope == "security"

    def test_security_agent_config_verbose(self):
        """security_agent should have verbose=False by default."""
        assert security_agent.config.verbose is False


class TestPresetImports:
    """Test suite for preset imports and exports."""

    def test_analyzer_agent_importable_from_agents(self):
        """analyzer_agent should be importable from src.agents."""
        from src.agents import analyzer_agent as imported
        assert imported is analyzer_agent

    def test_summarizer_agent_importable_from_agents(self):
        """summarizer_agent should be importable from src.agents."""
        from src.agents import summarizer_agent as imported
        assert imported is summarizer_agent

    def test_debugger_agent_importable_from_agents(self):
        """debugger_agent should be importable from src.agents."""
        from src.agents import debugger_agent as imported
        assert imported is debugger_agent

    def test_security_agent_importable_from_agents(self):
        """security_agent should be importable from src.agents."""
        from src.agents import security_agent as imported
        assert imported is security_agent

    def test_all_presets_in_package_all(self):
        """All preset agents should be in src.agents.__all__."""
        from src.agents import __all__ as agents_all
        assert "analyzer_agent" in agents_all
        assert "summarizer_agent" in agents_all
        assert "debugger_agent" in agents_all
        assert "security_agent" in agents_all

    def test_all_presets_in_module_all(self):
        """All preset agents should be in presets module __all__."""
        from src.agents.presets import __all__ as presets_all
        assert "analyzer_agent" in presets_all
        assert "summarizer_agent" in presets_all
        assert "debugger_agent" in presets_all
        assert "security_agent" in presets_all


class TestPresetConfigurations:
    """Test suite for preset configuration differentiation."""

    def test_analyzer_has_highest_depth_after_debugger(self):
        """analyzer depth (4) should be higher than summarizer but lower than debugger."""
        assert analyzer_agent.config.default_max_depth > summarizer_agent.config.default_max_depth
        assert analyzer_agent.config.default_max_depth < debugger_agent.config.default_max_depth

    def test_debugger_has_highest_depth(self):
        """debugger should have the highest max_depth (5) for deep investigation."""
        depths = [
            analyzer_agent.config.default_max_depth,
            summarizer_agent.config.default_max_depth,
            debugger_agent.config.default_max_depth,
            security_agent.config.default_max_depth,
        ]
        assert debugger_agent.config.default_max_depth == max(depths)
        assert debugger_agent.config.default_max_depth == 5

    def test_summarizer_has_lowest_depth(self):
        """summarizer should have the lowest max_depth (2) for concise output."""
        depths = [
            analyzer_agent.config.default_max_depth,
            summarizer_agent.config.default_max_depth,
            debugger_agent.config.default_max_depth,
            security_agent.config.default_max_depth,
        ]
        assert summarizer_agent.config.default_max_depth == min(depths)
        assert summarizer_agent.config.default_max_depth == 2

    def test_security_agent_has_security_scope(self):
        """security_agent should have unique security scope."""
        scopes = [
            analyzer_agent.config.default_scope,
            summarizer_agent.config.default_scope,
            debugger_agent.config.default_scope,
            security_agent.config.default_scope,
        ]
        assert scopes.count("security") == 1
        assert security_agent.config.default_scope == "security"

    def test_debugger_agent_has_correctness_scope(self):
        """debugger_agent should have correctness scope for bug hunting."""
        assert debugger_agent.config.default_scope == "correctness"

    def test_all_agents_have_unique_names(self):
        """Each agent should have a unique name."""
        names = [
            analyzer_agent.config.name,
            summarizer_agent.config.name,
            debugger_agent.config.name,
            security_agent.config.name,
        ]
        assert len(names) == len(set(names))
        assert names == ["analyzer", "summarizer", "debugger", "security"]

    def test_all_agents_have_non_empty_prompts(self):
        """Each agent should have a non-empty system prompt override."""
        assert len(analyzer_agent.config.system_prompt_override) > 0
        assert len(summarizer_agent.config.system_prompt_override) > 0
        assert len(debugger_agent.config.system_prompt_override) > 0
        assert len(security_agent.config.system_prompt_override) > 0
