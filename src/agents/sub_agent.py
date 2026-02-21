"""RLMSubAgent - Reusable sub-agent wrapper around RLAPHLoop.

This module provides specialized sub-agents with persona/prompt overrides
for top-level RLM workflows. Each sub-agent wraps RLAPHLoop with its own
configuration and can be invoked independently or as part of a larger system.

Example:
    analyzer = RLMSubAgent(RLMSubAgentConfig(
        name="analyzer",
        system_prompt_override="You are a code analysis specialist.",
        default_max_depth=2,
    ))
    result = await analyzer.run("Analyze the auth module")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..repl.rlaph_loop import RLAPHLoop, RLPALoopResult
from ..frame.canonical_task import CanonicalTask
from ..types import SessionContext

if TYPE_CHECKING:
    pass


@dataclass
class RLMSubAgentConfig:
    """Configuration for a specialized sub-agent.

    Attributes:
        name: Agent name/identifier (e.g., "analyzer", "debugger").
            Used to prefix answers for clarity.
        system_prompt_override: Optional specialized prompt that gets
            prepended to each query. Use this to give the agent a persona
            or specific domain expertise.
        default_max_depth: Default maximum recursion depth for llm() calls.
            Can be overridden per-run.
        default_scope: Default analysis scope for CanonicalTask when none
            is provided (e.g., "overview", "correctness", "security").
        verbose: Enable verbose logging for recursion decisions.
    """

    name: str
    system_prompt_override: str = ""
    default_max_depth: int = 3
    default_scope: str = "overview"
    verbose: bool = False


class RLMSubAgent:
    """Reusable sub-agent wrapper around RLAPHLoop.

    RLMSubAgent provides a convenient way to create specialized agents with
    specific personas, prompts, and configurations. Each agent maintains its
    own RLAPHLoop instance and can be invoked with optional overrides.

    Key design points:
    - Wraps RLAPHLoop for top-level RLM workflows (NOT for internal llm() calls)
    - System prompt override is prepended to queries when provided
    - Configuration can be overridden per-run
    - Result answers are prefixed with agent name for clarity

    Example:
        # Create a specialized code analyzer
        analyzer = RLMSubAgent(RLMSubAgentConfig(
            name="analyzer",
            system_prompt_override="You are a senior code reviewer. Focus on "
                                  "security, performance, and maintainability.",
            default_max_depth=4,
            verbose=True,
        ))

        # Use the analyzer
        result = await analyzer.run("Review the authentication module")

        # With per-run overrides
        result = await analyzer.run(
            "Check for SQL injection vulnerabilities",
            max_depth=5,
            working_dir=Path("/path/to/project"),
        )
    """

    def __init__(self, config: RLMSubAgentConfig):
        """Initialize a new RLMSubAgent.

        Args:
            config: Configuration for this agent including name, prompt override,
                max depth, scope, and verbosity settings.
        """
        self.config = config
        self.loop = RLAPHLoop(
            max_depth=config.default_max_depth,
            verbose=config.verbose,
        )

    async def run(
        self,
        query: str,
        working_dir: Path | None = None,
        session_id: str | None = None,
        canonical_task: CanonicalTask | None = None,
        max_depth: Optional[int] = None,
        context: SessionContext | None = None,
    ) -> RLPALoopResult:
        """Run this sub-agent with optional overrides.

        Executes the agent with the provided query, applying any overrides
        to the default configuration. The result's answer is prefixed with
        the agent name for clarity.

        Args:
            query: The query/prompt to execute.
            working_dir: Working directory for file operations (default: cwd).
            session_id: Optional session identifier for frame persistence
                (default: auto-generated UUID[:8]).
            canonical_task: Optional CanonicalTask for frame deduplication.
                If not provided, a default one is created using the config's
                default_scope.
            max_depth: Override the default max_depth for this run (default:
                use config.default_max_depth).
            context: Optional session context (default: empty SessionContext).

        Returns:
            RLPALoopResult with answer prefixed by agent name and execution
            metadata.
        """
        effective_depth = max_depth or self.config.default_max_depth
        effective_task = canonical_task or CanonicalTask(
            task_type="analyze",
            target="**/*",
            analysis_scope=self.config.default_scope,
            params={}
        )

        # Inject persona into query if override exists
        full_query = query
        if self.config.system_prompt_override:
            full_query = f"{self.config.system_prompt_override}\n\n{query}"

        # Create new loop with effective depth
        loop = RLAPHLoop(max_depth=effective_depth, verbose=self.config.verbose)
        ctx = context or SessionContext()
        wd = Path(working_dir) if working_dir else Path.cwd()

        result = await loop.run(full_query, ctx, wd, session_id)

        # Prefix answer with agent name for clarity
        result.answer = f"[{self.config.name.upper()}] {result.answer}"

        return result


__all__ = [
    "RLMSubAgent",
    "RLMSubAgentConfig",
]
