"""
Main RLM orchestration loop.

Implements: Spec §2 Architecture Overview
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .api_client import ClaudeClient, init_client
from .complexity_classifier import should_activate_rlm
from .config import RLMConfig, default_config
from .context_manager import externalize_context
from .cost_tracker import CostComponent, get_cost_tracker
from .prompts import build_rlm_system_prompt
from .repl_environment import RLMEnvironment
from .response_parser import ResponseAction, ResponseParser
from .trajectory import (
    StreamingTrajectory,
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryRenderer,
)
from .types import SessionContext


@dataclass
class OrchestrationState:
    """State for the orchestration loop."""

    depth: int = 0
    turn: int = 0
    max_turns: int = 20
    messages: list[dict[str, str]] = field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None


class RLMOrchestrator:
    """
    Main RLM orchestration loop.

    Implements: Spec §2.1 High-Level Design
    """

    def __init__(
        self,
        config: RLMConfig | None = None,
        client: ClaudeClient | None = None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: RLM configuration (uses default if None)
            client: Claude API client (creates one if None)
        """
        self.config = config or default_config
        self.client = client
        self.activation_reason: str = ""
        self.parser = ResponseParser()

    def _ensure_client(self) -> ClaudeClient:
        """Ensure we have an API client."""
        if self.client is None:
            self.client = init_client(default_model=self.config.model.root_model)
        return self.client

    async def run(
        self, query: str, context: SessionContext
    ) -> AsyncIterator[TrajectoryEvent | str]:
        """
        Run RLM loop on a query.

        Implements: Spec §2 Architecture Overview

        Args:
            query: User query
            context: Session context

        Yields:
            TrajectoryEvents and final response
        """
        # Check if RLM should activate
        should_activate, self.activation_reason = should_activate_rlm(query, context)

        if not should_activate:
            # Bypass RLM, return direct
            yield TrajectoryEvent(
                type=TrajectoryEventType.FINAL,
                depth=0,
                content=f"[Direct mode: {self.activation_reason}]",
            )
            return

        # Initialize components
        client = self._ensure_client()
        renderer = TrajectoryRenderer(
            verbosity=self.config.trajectory.verbosity,
            colors=self.config.trajectory.colors,
        )
        trajectory = StreamingTrajectory(renderer)

        # Initialize state
        state = OrchestrationState(
            max_turns=self.config.depth.max * 10,  # Allow multiple turns per depth
        )

        # Start event
        start_event = TrajectoryEvent(
            type=TrajectoryEventType.RLM_START,
            depth=0,
            content=f"depth=0/{self.config.depth.max} • task: {self.activation_reason}",
            metadata={"query": query, "context_tokens": context.total_tokens},
        )
        await trajectory.emit(start_event)
        yield start_event

        # Initialize REPL environment
        repl = RLMEnvironment(context)

        # Set up recursive query function in REPL
        async def recursive_query_wrapper(q: str, ctx: str) -> str:
            return await client.recursive_query(q, ctx)

        # Note: We can't directly use async in REPL, so recursive_query
        # will need special handling in the loop

        # Analyze context
        externalized = externalize_context(context)
        analyze_event = TrajectoryEvent(
            type=TrajectoryEventType.ANALYZE,
            depth=0,
            content=f"Context: {context.total_tokens} tokens, {len(context.files)} files",
            metadata=externalized.get("context_stats"),
        )
        await trajectory.emit(analyze_event)
        yield analyze_event

        # Build system prompt
        system_prompt = build_rlm_system_prompt(context, query)

        # Initial user message
        state.messages = [{"role": "user", "content": query}]

        # Main orchestration loop
        while state.turn < state.max_turns and state.final_answer is None:
            state.turn += 1

            # Get response from Claude
            try:
                response = await client.complete(
                    messages=state.messages,
                    system=system_prompt,
                    max_tokens=4096,
                    component=CostComponent.ROOT_PROMPT,
                )
            except Exception as e:
                error_event = TrajectoryEvent(
                    type=TrajectoryEventType.ERROR,
                    depth=state.depth,
                    content=f"API error: {e}",
                )
                await trajectory.emit(error_event)
                yield error_event
                state.error = str(e)
                break

            # Emit reasoning event
            reason_event = TrajectoryEvent(
                type=TrajectoryEventType.REASON,
                depth=state.depth,
                content=response.content[:500] + ("..." if len(response.content) > 500 else ""),
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
            await trajectory.emit(reason_event)
            yield reason_event

            # Parse response
            parsed_items = self.parser.parse(response.content)

            if not parsed_items:
                # No actionable content, might be stuck
                state.messages.append({"role": "assistant", "content": response.content})
                state.messages.append({
                    "role": "user",
                    "content": "Please continue. Use ```python blocks for REPL code, "
                    "or output FINAL: <answer> when you have the answer.",
                })
                continue

            # Process each parsed item
            for item in parsed_items:
                if item.action == ResponseAction.FINAL_ANSWER:
                    state.final_answer = item.content
                    break

                elif item.action == ResponseAction.FINAL_VAR:
                    # Get answer from REPL variable
                    var_name = item.content
                    try:
                        var_value = repl.get_variable(var_name)
                        state.final_answer = str(var_value)
                    except KeyError:
                        state.messages.append({"role": "assistant", "content": response.content})
                        state.messages.append({
                            "role": "user",
                            "content": f"Variable '{var_name}' not found. Available variables: "
                            f"{list(repl.globals.get('working_memory', {}).keys())}",
                        })
                    break

                elif item.action == ResponseAction.REPL_EXECUTE:
                    # Execute code in REPL
                    code = item.content

                    # Emit REPL exec event
                    exec_event = TrajectoryEvent(
                        type=TrajectoryEventType.REPL_EXEC,
                        depth=state.depth,
                        content=code[:200] + ("..." if len(code) > 200 else ""),
                    )
                    await trajectory.emit(exec_event)
                    yield exec_event

                    # Check for recursive_query calls and handle them
                    if "recursive_query(" in code or "recursive_llm(" in code:
                        # Extract and execute recursive queries
                        result = await self._handle_recursive_code(
                            code, repl, client, state.depth, trajectory
                        )
                        for event in result.get("events", []):
                            yield event
                        repl_result = result.get("output", "")
                    else:
                        # Regular REPL execution
                        exec_result = repl.execute(code)
                        repl_result = exec_result.output if exec_result.success else f"Error: {exec_result.error}"

                    # Emit REPL result event
                    result_event = TrajectoryEvent(
                        type=TrajectoryEventType.REPL_RESULT,
                        depth=state.depth,
                        content=str(repl_result)[:500],
                    )
                    await trajectory.emit(result_event)
                    yield result_event

                    # Add to conversation
                    state.messages.append({"role": "assistant", "content": response.content})
                    state.messages.append({
                        "role": "user",
                        "content": f"REPL output:\n```\n{repl_result}\n```\n\nContinue your analysis or provide FINAL: <answer>",
                    })

                elif item.action == ResponseAction.THINKING:
                    # Just thinking, add to messages and continue
                    state.messages.append({"role": "assistant", "content": response.content})
                    state.messages.append({
                        "role": "user",
                        "content": "Please continue with REPL actions or provide your final answer.",
                    })

            # Break if we have answer
            if state.final_answer:
                break

        # Final event
        if state.final_answer:
            final_content = state.final_answer
        elif state.error:
            final_content = f"[Error: {state.error}]"
        else:
            final_content = "[Max turns reached without answer]"

        final_event = TrajectoryEvent(
            type=TrajectoryEventType.FINAL,
            depth=0,
            content=final_content,
            metadata={
                "turns": state.turn,
                "cost": get_cost_tracker().get_summary(),
            },
        )
        await trajectory.emit(final_event)
        yield final_event

        # Export trajectory if enabled
        if self.config.trajectory.export_enabled:
            import os
            import time
            from pathlib import Path

            export_dir = Path(os.path.expanduser(self.config.trajectory.export_path))
            export_dir.mkdir(parents=True, exist_ok=True)
            filename = f"trajectory_{int(time.time())}.json"
            trajectory.export_json(str(export_dir / filename))

    async def _handle_recursive_code(
        self,
        code: str,
        repl: RLMEnvironment,
        client: ClaudeClient,
        depth: int,
        trajectory: StreamingTrajectory,
    ) -> dict:
        """
        Handle code containing recursive_query calls.

        This extracts recursive_query calls, executes them via API,
        and substitutes results back into the code.
        """
        import re

        events = []

        # Pattern to match recursive_query calls
        pattern = r'recursive_query\s*\(\s*["\'](.+?)["\']\s*,\s*(.+?)\s*\)'

        matches = list(re.finditer(pattern, code))

        if not matches:
            # No recursive calls, execute normally
            result = repl.execute(code)
            return {"output": result.output if result.success else f"Error: {result.error}"}

        # Process recursive calls
        modified_code = code
        for match in reversed(matches):  # Reverse to preserve positions
            query = match.group(1)
            context_expr = match.group(2)

            # Emit recurse start event
            start_event = TrajectoryEvent(
                type=TrajectoryEventType.RECURSE_START,
                depth=depth + 1,
                content=f"Query: {query[:100]}",
            )
            await trajectory.emit(start_event)
            events.append(start_event)

            # Evaluate context expression in REPL to get actual context
            try:
                ctx_result = repl.execute(f"__ctx__ = {context_expr}")
                if ctx_result.success:
                    context_value = repl.get_variable("__ctx__")
                    context_str = str(context_value)[:8000]  # Limit context size
                else:
                    context_str = context_expr
            except Exception:
                context_str = context_expr

            # Make recursive API call
            try:
                answer = await client.recursive_query(query, context_str)
            except Exception as e:
                answer = f"[Recursive query error: {e}]"

            # Emit recurse end event
            end_event = TrajectoryEvent(
                type=TrajectoryEventType.RECURSE_END,
                depth=depth + 1,
                content=answer[:200],
            )
            await trajectory.emit(end_event)
            events.append(end_event)

            # Store result in working memory
            var_name = f"_rq_result_{len(events)}"
            repl.execute(f"working_memory['{var_name}'] = '''{answer}'''")

            # Replace call with result reference
            modified_code = (
                modified_code[: match.start()]
                + f"working_memory['{var_name}']"
                + modified_code[match.end():]
            )

        # Execute modified code
        result = repl.execute(modified_code)

        return {
            "events": events,
            "output": result.output if result.success else f"Error: {result.error}",
        }


async def main():
    """CLI entry point for testing."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="RLM Orchestrator")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument(
        "--verbosity", default="normal", choices=["minimal", "normal", "verbose", "debug"]
    )
    parser.add_argument("--export-trajectory", help="Path to export trajectory JSON")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: export ANTHROPIC_API_KEY=your-key-here")
        print("Or pass --api-key argument")
        return

    # Create mock context for testing with actual project files
    files = {}
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(src_dir):
        if filename.endswith(".py"):
            filepath = os.path.join(src_dir, filename)
            try:
                with open(filepath) as f:
                    files[filename] = f.read()
            except Exception:
                pass

    context = SessionContext(files=files)

    config = RLMConfig()
    config.trajectory.verbosity = args.verbosity

    if args.export_trajectory:
        config.trajectory.export_enabled = True
        config.trajectory.export_path = args.export_trajectory

    # Initialize client
    client = init_client(api_key=api_key)

    orchestrator = RLMOrchestrator(config, client)

    async for event in orchestrator.run(args.query, context):
        if isinstance(event, TrajectoryEvent):
            renderer = TrajectoryRenderer(verbosity=args.verbosity)
            print(renderer.render_event(event))


if __name__ == "__main__":
    asyncio.run(main())
