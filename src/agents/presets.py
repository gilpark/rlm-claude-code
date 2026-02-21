"""Pre-configured sub-agents for common tasks."""
from src.agents.sub_agent import RLMSubAgent, RLMSubAgentConfig

analyzer_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="analyzer",
        system_prompt_override="You are a detailed code analyst. Focus on architecture, structure, and correctness.",
        default_max_depth=4,
        default_scope="overview",
    )
)

summarizer_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="summarizer",
        system_prompt_override="You are a concise summarizer. Keep answers short, structured, and to the point.",
        default_max_depth=2,
        default_scope="overview",
    )
)

debugger_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="debugger",
        system_prompt_override="You are a bug hunter. Focus on reproduction steps, root causes, and fixes.",
        default_max_depth=5,
        default_scope="correctness",
    )
)

security_agent = RLMSubAgent(
    RLMSubAgentConfig(
        name="security",
        system_prompt_override="You are a security auditor. Focus on vulnerabilities, attack vectors, and mitigations.",
        default_max_depth=4,
        default_scope="security",
    )
)


__all__ = [
    "analyzer_agent",
    "summarizer_agent",
    "debugger_agent",
    "security_agent",
]
