"""
RLAPH Loop - Clean Recursive Language Agent with Python Handler.

Implements the RLAPH pattern from plan.md:
- Single clean loop (not deferred operations)
- llm() returns actual result synchronously
- Depth management built-in
- History tracking for debugging

Reference: plan.md "After (RLAPH Loop)" architecture
"""

from __future__ import annotations

import asyncio
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Suppress third-party warnings at import time
warnings.filterwarnings("ignore", category=UserWarning, module="cpmpy")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="RestrictedPython")

from .config import RLMConfig, default_config
from .repl_environment import RLMEnvironment
from .response_parser import ResponseAction, ResponseParser
from .router_integration import ModelRouter
from .trajectory import StreamingTrajectory, TrajectoryEvent, TrajectoryEventType
from .types import RecursionDepthError, SessionContext

if TYPE_CHECKING:
    from .recursive_handler import RecursiveREPL
    from .trajectory import TrajectoryRenderer


@dataclass
class RLPALoopState:
    """State for a single RLAPH loop execution."""

    turn: int = 0
    max_turns: int = 20
    depth: int = 0
    final_answer: str | None = None
    messages: list[dict] = field(default_factory=list)
    consecutive_thinking: int = 0
    last_thinking_content: str = ""


@dataclass
class RLPALoopResult:
    """Result from RLAPH loop execution."""

    answer: str
    iterations: int
    depth_used: int
    tokens_used: int
    execution_time_ms: float
    history: list[dict]


class RLAPHLoop:
    """
    Clean RLM agent loop with synchronous LLM calls.

    RLAPH = Recursive Language Agent with Python Handler

    Key difference from legacy RLMOrchestrator:
    - llm() returns actual result immediately (not DeferredOperation)
    - Single predictable loop
    - Clear iteration history for debugging
    - No deferred operation complexity

    Usage:
        loop = RLAPHLoop(max_depth=2)
        result = await loop.run("Analyze the auth flow", context)
        print(result.answer)
    """

    def __init__(
        self,
        max_iterations: int = 20,
        max_depth: int = 3,
        config: RLMConfig | None = None,
        router: ModelRouter | None = None,
        renderer: TrajectoryRenderer | None = None,
    ):
        """
        Initialize RLAPH loop.

        Args:
            max_iterations: Maximum loop iterations
            max_depth: Maximum recursion depth for llm() calls
            config: RLM configuration
            router: Model router for API calls
            renderer: Trajectory renderer for output
        """
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.config = config or default_config
        self.router = router or ModelRouter(self.config)

        # State
        self.repl: RLMEnvironment | None = None
        self.recursive_handler: RecursiveREPL | None = None
        self.history: list[dict] = []
        self._depth = 0
        self._tokens_used = 0

        # Trajectory
        if renderer:
            self.trajectory = StreamingTrajectory(renderer)
        else:
            self.trajectory = None

        # Parser
        self.parser = ResponseParser()

    @property
    def depth(self) -> int:
        """Current recursion depth."""
        return self._depth

    @property
    def total_tokens_used(self) -> int:
        """Total tokens used including recursive calls."""
        total = self._tokens_used
        if self.recursive_handler:
            total += self.recursive_handler.total_tokens_used
        return total

    async def run(
        self,
        query: str,
        context: SessionContext,
        working_dir: Path | str | None = None,
    ) -> RLPALoopResult:
        """
        Main loop - clean, predictable, debuggable.

        This is the core RLAPH loop:
        1. Build system prompt with REPL instructions
        2. For each iteration:
           a. Call LLM
           b. Parse response
           c. Execute REPL code (llm() is synchronous now)
           d. Check for FINAL answer
        3. Return result

        Args:
            query: User query
            context: Session context (files, conversation, etc.)
            working_dir: Working directory for file operations (default: cwd)

        Returns:
            RLPALoopResult with answer and metadata
        """
        start_time = time.time()
        state = RLPALoopState(
            max_turns=self.max_iterations,
            depth=self._depth,
        )

        # Initialize REPL with recursive handler
        from .recursive_handler import RecursiveREPL

        self.recursive_handler = RecursiveREPL(
            context=context,
            depth=self._depth,
            max_depth=self.max_depth,
            config=self.config,
            router=self.router,
            trajectory=self.trajectory,
        )
        self.repl = RLMEnvironment(context, recursive_handler=self.recursive_handler)

        # Enable file access for context externalization
        # This is critical for RLM to handle large contexts:
        # Files are NOT passed in the prompt, REPL reads them on-demand
        work_dir = Path(working_dir) if working_dir else Path.cwd()
        self.repl.enable_file_access(working_dir=work_dir)

        # Build initial messages
        system_prompt = self._build_system_prompt()
        state.messages = [
            {"role": "user", "content": query},
        ]

        # Emit start event
        if self.trajectory:
            await self.trajectory.emit(
                TrajectoryEvent(
                    type=TrajectoryEventType.RLM_START,
                    depth=self._depth,
                    content=f"RLAPH loop starting: {query[:100]}",
                    metadata={"max_iterations": self.max_iterations, "max_depth": self.max_depth},
                )
            )

        # Main loop
        while state.turn < state.max_turns and state.final_answer is None:
            state.turn += 1

            # Get LLM response - convert messages to prompt
            model = self._get_model_for_depth()
            prompt = self._messages_to_prompt(state.messages)
            response = await self.router.complete(
                model=model,
                prompt=prompt,
                system=system_prompt,
                max_tokens=16384,
            )

            # Track tokens
            self._tokens_used += len(response.content) // 4  # Rough estimate

            # Emit reasoning event
            if self.trajectory:
                await self.trajectory.emit(
                    TrajectoryEvent(
                        type=TrajectoryEventType.REASON,
                        depth=self._depth,
                        content=response.content[:500],
                        metadata={"turn": state.turn, "model": model},
                    )
                )

            # Parse response
            parsed_items = self.parser.parse(response.content)

            # Process each parsed item
            for item in parsed_items:
                if item.action == ResponseAction.FINAL_ANSWER:
                    state.final_answer = item.content
                    break

                elif item.action == ResponseAction.FINAL_VAR:
                    var_name = item.content
                    try:
                        var_value = self.repl.get_variable(var_name)
                        state.final_answer = str(var_value)
                    except KeyError:
                        state.final_answer = f"Variable '{var_name}' not found"
                    break

                elif item.action == ResponseAction.REPL_EXECUTE:
                    # Execute code - llm() now returns actual result!
                    state.consecutive_thinking = 0
                    code = item.content

                    # Emit REPL exec event
                    if self.trajectory:
                        await self.trajectory.emit(
                            TrajectoryEvent(
                                type=TrajectoryEventType.REPL_EXEC,
                                depth=self._depth,
                                content=code[:100],
                                metadata={"turn": state.turn},
                            )
                        )

                    # Execute code synchronously
                    exec_result = self.repl.execute(code)
                    repl_result = (
                        exec_result.output if exec_result.success else f"Error: {exec_result.error}"
                    )

                    # Emit REPL result event
                    if self.trajectory:
                        await self.trajectory.emit(
                            TrajectoryEvent(
                                type=TrajectoryEventType.REPL_RESULT,
                                depth=self._depth,
                                content=str(repl_result)[:500],
                            )
                        )

                    # Truncate REPL output
                    MAX_REPL_OUTPUT = 1500
                    repl_result_str = str(repl_result) if repl_result else ""
                    truncated_result = repl_result_str[:MAX_REPL_OUTPUT]
                    if len(repl_result_str) > MAX_REPL_OUTPUT:
                        truncated_result += f"\n... [truncated, {len(repl_result_str)} chars total]"

                    # Add to conversation
                    state.messages.append({"role": "assistant", "content": response.content})
                    state.messages.append(
                        {
                            "role": "user",
                            "content": f"[SYSTEM - Code execution result]:\n```\n{truncated_result}\n```\n\nNow provide your FINAL: <answer> based on these results.",
                        }
                    )

                elif item.action == ResponseAction.THINKING:
                    # Track consecutive thinking
                    state.consecutive_thinking += 1
                    state.last_thinking_content = item.content

                    # Fallback after 2+ thinking turns with substantial content
                    if state.consecutive_thinking >= 2 and len(item.content) >= 100:
                        content_stripped = item.content.strip()
                        if not content_stripped.endswith(("...", ":", "-", "•")):
                            state.final_answer = item.content
                            break

                    state.messages.append({"role": "assistant", "content": response.content})
                    state.messages.append(
                        {
                            "role": "user",
                            "content": "Please continue with REPL actions or provide your final answer.",
                        }
                    )

            # Record history
            self.history.append(
                {
                    "turn": state.turn,
                    "action": str([item.action for item in parsed_items]),
                    "has_answer": state.final_answer is not None,
                }
            )

        # Fallback: Use last thinking content if no answer produced
        if state.final_answer is None and state.last_thinking_content:
            state.final_answer = state.last_thinking_content

        # Build result
        execution_time = (time.time() - start_time) * 1000

        # Emit final event
        if self.trajectory:
            await self.trajectory.emit(
                TrajectoryEvent(
                    type=TrajectoryEventType.FINAL,
                    depth=self._depth,
                    content=state.final_answer or "No answer produced",
                    metadata={"iterations": state.turn, "tokens_used": self.total_tokens_used},
                )
            )

        return RLPALoopResult(
            answer=state.final_answer or "No answer produced",
            iterations=state.turn,
            depth_used=self._depth,
            tokens_used=self.total_tokens_used,
            execution_time_ms=execution_time,
            history=self.history.copy(),
        )

    def llm_sync(self, query: str, context: str = "") -> str:
        """
        Synchronous LLM call - returns actual result immediately.

        This is the key method that makes RLAPH work:
        - Called from REPL's llm() function
        - Returns actual string result
        - Handles depth management

        Args:
            query: Query string
            context: Optional context string

        Returns:
            LLM response as string

        Raises:
            RecursionDepthError: If max depth exceeded
        """
        if not self.recursive_handler:
            return "Error: No recursive handler available"

        # Check depth
        if self._depth >= self.max_depth:
            raise RecursionDepthError(self._depth + 1, self.max_depth)

        # Use recursive handler's sync method
        return self.recursive_handler.llm_sync(query, context)

    def _build_system_prompt(self) -> str:
        """Build system prompt with REPL instructions."""
        return """You are an RLM (Recursive Language Model) agent with access to a REAL Python REPL.

CRITICAL RULES:
1. When you write code in ```python blocks, the system EXECUTES it and returns REAL output
2. DO NOT generate fake "REPL output" or "Human:" messages yourself
3. DO NOT pretend to see execution results - wait for the actual system response
4. After writing code, STOP and wait for the [SYSTEM - Code execution result]
5. When you have the final answer, write: FINAL: <answer>

Your workflow:
1. Write Python code in ```python blocks
2. STOP - the system will execute and return [SYSTEM - Code execution result]
3. Read the REAL output from the system
4. Write more code OR provide FINAL: <answer>

File Access Functions:
- `read_file(path, offset=0, limit=2000)`: Read file content from disk
- `glob_files(pattern)`: Find files matching pattern
- `grep_files(pattern, path)`: Search for pattern in files
- `list_dir(path)`: List directory contents

Example:
User: How many Python files in src/?

Your response:
```python
files = glob_files("src/**/*.py")
print(len(files))
```
[STOP HERE - wait for system to execute]

System returns: [SYSTEM - Code execution result]:
```
81
```

Now you can answer:
FINAL: There are 81 Python files in src/

Other functions: peek(), search(), summarize(), llm(), llm_batch(), map_reduce()
Working memory: working_memory dict for storing results across code blocks"""

    def _get_model_for_depth(self) -> str:
        """Get appropriate model for current depth."""
        import os

        # Check for custom model env vars
        opus_model = os.environ.get("ANTHROPIC_DEFAULT_OPUS_MODEL", "glm-5")
        sonnet_model = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "glm-4.7")
        haiku_model = os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", "glm-4.7")

        # Route to cheaper models at deeper depths
        depth_model_map = {
            0: sonnet_model,
            1: sonnet_model,
            2: haiku_model,
            3: haiku_model,
        }
        return depth_model_map.get(self._depth, haiku_model)

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert messages list to a single prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Use clear markers that won't confuse LLM into generating fake turns
            if role == "user":
                # Check if this is a system message (code execution result)
                if content.startswith("[SYSTEM"):
                    parts.append(content)  # Don't add prefix for system messages
                else:
                    parts.append(f"User query:\n{content}")
            elif role == "assistant":
                parts.append(f"Your response:\n{content}")
            else:
                parts.append(f"{role}: {content}")
        return "\n\n---\n\n".join(parts)


__all__ = [
    "RLAPHLoop",
    "RLPALoopResult",
    "RLPALoopState",
]
