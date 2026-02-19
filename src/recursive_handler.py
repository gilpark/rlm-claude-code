"""
Recursive call handling for RLM-Claude-Code.

Implements: Spec §4.2 Recursive Call Implementation, §6.4 Depth=2
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from .config import RLMConfig, default_config
from .repl_environment import RLMEnvironment
from .router_integration import ModelRouter
from .types import (
    CostLimitError,
    RecursionDepthError,
    RecursiveCallResult,
    SessionContext,
)

if TYPE_CHECKING:
    from .trajectory import TrajectoryStream


class RecursiveREPL:
    """
    REPL environment that can spawn child REPLs for depth=2.

    Implements: Spec §6.4 Recursive Depth = 2

    Provides:
    - Depth management with configurable max depth
    - Model routing based on depth (Opus → Sonnet → Haiku)
    - Cost tracking and enforcement
    - Result aggregation from child REPLs
    - Trajectory event emission for observability
    - Adaptive depth budgeting (Feature 3e0.4)
    """

    def __init__(
        self,
        context: Any,
        depth: int = 0,
        max_depth: int | None = None,
        config: RLMConfig | None = None,
        router: ModelRouter | None = None,
        trajectory: TrajectoryStream | None = None,
        parent: RecursiveREPL | None = None,
        budget_tracker: Any | None = None,
        parent_frame_id: str | None = None,
    ):
        """
        Initialize recursive REPL.

        Args:
            context: Context for this REPL (SessionContext or string)
            depth: Current recursion depth (0 = root)
            max_depth: Maximum allowed depth (defaults to config)
            config: RLM configuration
            router: Model router for API calls
            trajectory: Trajectory stream for event logging
            parent: Parent REPL (for cost aggregation)
            budget_tracker: Optional EnhancedBudgetTracker for adaptive depth
            parent_frame_id: Parent frame ID for CausalFrame tree (SPEC-17)
        """
        self.context = context
        self.depth = depth
        self.config = config or default_config
        self.max_depth = max_depth if max_depth is not None else self.config.depth.max
        self.router = router or ModelRouter(self.config)
        self.trajectory = trajectory
        self.parent = parent
        self.child_repls: list[RecursiveREPL] = []
        self.budget_tracker = budget_tracker

        # CausalFrame tracking (SPEC-17)
        self.parent_frame_id = parent_frame_id
        self._current_frame_id: str | None = parent_frame_id

        # Cost tracking
        self._tokens_used = 0
        self._recursive_calls = 0
        self._results: list[RecursiveCallResult] = []

        # Adaptive depth state (Feature 3e0.4)
        self._model_override: str | None = None  # Override from adaptive depth
        self._depth_override: int | None = None  # Override from adaptive depth

        # Create REPL environment if context is SessionContext
        if isinstance(context, SessionContext):
            self.repl = RLMEnvironment(context, recursive_handler=self)
        else:
            self.repl = None

    @property
    def total_tokens_used(self) -> int:
        """Total tokens used including all children."""
        total = self._tokens_used
        for child in self.child_repls:
            total += child.total_tokens_used
        return total

    @property
    def total_recursive_calls(self) -> int:
        """Total recursive calls including all children."""
        total = self._recursive_calls
        for child in self.child_repls:
            total += child.total_recursive_calls
        return total

    def _check_cost_limits(self) -> None:
        """
        Check if cost limits are exceeded.

        Raises:
            CostLimitError: If token or call limits exceeded
        """
        # Check token limit
        abort_threshold = self.config.cost_controls.abort_on_cost_threshold
        if self.total_tokens_used >= abort_threshold:
            raise CostLimitError(self.total_tokens_used, abort_threshold)

        # Check call limit (only at root)
        if self.parent is None:
            max_calls = self.config.cost_controls.max_recursive_calls_per_turn
            if self.total_recursive_calls >= max_calls:
                raise CostLimitError(
                    self.total_recursive_calls,
                    max_calls,
                )

    async def recursive_query(
        self,
        query: str,
        context: Any,
        spawn_repl: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """
        Spawn a recursive call.

        Implements: Spec §4.2 Recursive Call Implementation

        Args:
            query: Query for the sub-call
            context: Context to pass to sub-call
            spawn_repl: If True and depth < max, child gets own REPL
            max_tokens: Max tokens for response (defaults to config)

        Returns:
            Sub-model's response

        Raises:
            RecursionDepthError: If max depth exceeded
            CostLimitError: If cost limits exceeded
        """
        # Apply adaptive depth if budget tracker available (Feature 3e0.4)
        effective_max_depth = self.max_depth
        if self.budget_tracker is not None and self._model_override is None:
            try:
                recommendation = self.budget_tracker.compute_adaptive_depth(
                    planned_depth=self.max_depth,
                    planned_model=self.get_model_for_depth(),
                )
                if recommendation.was_downgraded:
                    self._model_override = recommendation.recommended_model
                    effective_max_depth = min(self.max_depth, recommendation.recommended_depth)
                    self._depth_override = effective_max_depth

                    # Emit trajectory event for model downgrade
                    if self.trajectory:
                        self.trajectory.emit_model_downgrade(
                            original_model=recommendation.original_model,
                            new_model=recommendation.recommended_model,
                            reason=recommendation.downgrade_reason or "budget_constraint",
                            budget_utilization=recommendation.budget_utilization,
                        )
            except Exception:
                # Don't fail on adaptive depth errors, just continue with defaults
                pass

        # Check depth limit
        if self.depth >= effective_max_depth:
            raise RecursionDepthError(self.depth + 1, effective_max_depth)

        # Check cost limits before making call
        self._check_cost_limits()

        # Increment call counter
        self._recursive_calls += 1

        max_tokens = max_tokens or self.config.cost_controls.max_tokens_per_recursive_call

        # Emit trajectory event if available
        if self.trajectory:
            self.trajectory.emit_recursive_start(
                depth=self.depth + 1,
                query=query[:100],
                spawn_repl=spawn_repl,
            )

        try:
            if spawn_repl and self.depth + 1 < self.max_depth:
                # Create child REPL with isolated state
                result = await self._spawn_child_repl(query, context, max_tokens)
            else:
                # Simple sub-call without REPL
                result = await self._simple_completion(query, context, max_tokens)

            # Track result
            self._results.append(result)

            # Emit completion event
            if self.trajectory:
                self.trajectory.emit_recursive_complete(
                    depth=result.depth,
                    tokens_used=result.tokens_used,
                    execution_time_ms=result.execution_time_ms,
                )

            return result.content

        except Exception as e:
            # Emit error event
            if self.trajectory:
                self.trajectory.emit_recursive_error(
                    depth=self.depth + 1,
                    error=str(e),
                )
            raise

    def llm_sync(self, query: str, context: str = "") -> str:
        """
        Synchronous LLM call - returns actual result immediately.

        This is the key method for RLAPH-style loop: llm() returns the
        actual string result instead of a DeferredOperation.

        Uses a separate thread with its own event loop to avoid blocking
        the parent async context.

        Args:
            query: Query string for sub-call
            context: Optional context string

        Returns:
            LLM response as string

        Raises:
            RecursionDepthError: If max depth exceeded
            CostLimitError: If cost limits exceeded
        """
        import asyncio
        import concurrent.futures
        import threading

        def run_in_thread():
            """Run the async query in a new thread with its own event loop."""
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.recursive_query(query, context))
            finally:
                new_loop.close()

        try:
            # Check if we're in an async context
            asyncio.get_running_loop()

            # We're in async context - run in a separate thread
            # to avoid blocking/nesting issues
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=120.0)  # 2 minute timeout

        except RuntimeError:
            # No running loop, can run directly
            return asyncio.run(self.recursive_query(query, context))

    async def _spawn_child_repl(
        self,
        query: str,
        context: Any,
        max_tokens: int,
    ) -> RecursiveCallResult:
        """
        Create child REPL with isolated state.

        Implements: Spec §6.4 Recursive Depth = 2
        """
        start_time = time.time()

        # Use depth override if set by adaptive budgeting
        effective_max_depth = self._depth_override or self.max_depth

        child = RecursiveREPL(
            context=context,
            depth=self.depth + 1,
            max_depth=effective_max_depth,
            config=self.config,
            router=self.router,
            trajectory=self.trajectory,
            parent=self,
            budget_tracker=self.budget_tracker,  # Pass down for adaptive depth
            parent_frame_id=self._current_frame_id,  # Pass current frame as parent (SPEC-17)
        )
        # Inherit model override if set
        if self._model_override:
            child._model_override = self._model_override
        self.child_repls.append(child)

        # Run RLM loop in child
        content = await child.run_rlm_loop(query, max_tokens)

        execution_time = (time.time() - start_time) * 1000

        return RecursiveCallResult(
            content=content,
            depth=self.depth + 1,
            model_used=child.get_model_for_depth(),
            tokens_used=child.total_tokens_used,
            execution_time_ms=execution_time,
            had_repl=True,
            child_results=child._results.copy(),
        )

    async def _simple_completion(
        self,
        query: str,
        context: Any,
        max_tokens: int,
    ) -> RecursiveCallResult:
        """
        Simple completion without REPL (for max depth or non-REPL calls).

        Implements: Spec §6.4 Recursive Depth = 2
        """
        start_time = time.time()

        # Serialize context for prompt
        context_str = self._serialize_context(context)

        # Build sub-call prompt per spec §4.2
        sub_prompt = self._build_sub_prompt(query, context_str)

        # Get model for this depth
        model = self.get_model_for_depth()

        # Make API call
        completion = await self.router.complete(
            model=model,
            prompt=sub_prompt,
            max_tokens=max_tokens,
            depth=self.depth + 1,
        )

        # Track tokens
        self._tokens_used += completion.tokens_used

        execution_time = (time.time() - start_time) * 1000

        return RecursiveCallResult(
            content=completion.content,
            depth=self.depth + 1,
            model_used=model,
            tokens_used=completion.tokens_used,
            execution_time_ms=execution_time,
            had_repl=False,
        )

    def _serialize_context(self, context: Any) -> str:
        """Serialize context for sub-call prompt."""
        if isinstance(context, str):
            return context
        elif isinstance(context, SessionContext):
            # Summarize session context
            parts = []
            if context.messages:
                parts.append(f"Conversation: {len(context.messages)} messages")
            if context.files:
                parts.append(f"Files: {', '.join(context.files.keys())}")
            if context.tool_outputs:
                parts.append(f"Tool outputs: {len(context.tool_outputs)} results")
            return "\n".join(parts)
        elif isinstance(context, dict):
            import json

            return json.dumps(context, indent=2, default=str)
        elif isinstance(context, list):
            return "\n".join(str(item) for item in context)
        else:
            return str(context)

    def _build_sub_prompt(self, query: str, context_str: str) -> str:
        """
        Build sub-call prompt per Spec §4.2.

        The sub-call receives:
        - The query
        - The context (as a string variable)
        - NO access to parent REPL state (isolated)
        """
        # Truncate context if too large
        max_context_chars = 8000
        if len(context_str) > max_context_chars:
            context_preview = context_str[:max_context_chars]
            truncate_note = f"... [truncated, {len(context_str)} chars total]"
        else:
            context_preview = context_str
            truncate_note = ""

        return f"""You are analyzing a portion of context for a larger task.

## Context
The following is stored in variable `context`:
{context_preview}{truncate_note}

## Query
{query}

Provide a focused, factual response. If you need to examine
the context programmatically, you have access to a Python REPL
with `context` as a variable."""

    async def run_rlm_loop(self, query: str, max_tokens: int = 4000) -> str:
        """
        Run RLM loop at this depth.

        Implements: Spec §4.2 Recursive Call Implementation

        This is the main execution loop that:
        1. Receives the query
        2. May execute REPL code to analyze context
        3. May spawn further recursive calls
        4. Returns final response
        """
        # Get model for this depth
        model = self.get_model_for_depth()

        # Build initial prompt with REPL instructions
        prompt = self._build_rlm_prompt(query)

        # Emit loop start event
        if self.trajectory:
            self.trajectory.emit_rlm_loop_start(
                depth=self.depth,
                model=model,
            )

        # Make API call
        completion = await self.router.complete(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            depth=self.depth,
        )

        self._tokens_used += completion.tokens_used

        # Emit loop complete event
        if self.trajectory:
            self.trajectory.emit_rlm_loop_complete(
                depth=self.depth,
                tokens_used=completion.tokens_used,
            )

        return completion.content

    def _build_rlm_prompt(self, query: str) -> str:
        """Build RLM prompt with context and REPL instructions."""
        context_str = self._serialize_context(self.context)

        return f"""You are operating in RLM (Recursive Language Model) mode at depth={self.depth}.

## Available Context
{context_str}

## Query
{query}

## Instructions
You have access to a Python REPL for context manipulation. Available:
- `context` variable containing the full context
- `peek(var, start, end)` to view portions
- `search(var, pattern, regex=False)` to search
- `recursive_query(query, context, spawn_repl=False)` for sub-calls

Signal completion with:
- FINAL(your answer) for direct answers
- FINAL_VAR(variable_name) if answer is in a variable

Respond with focused analysis. Spawn recursive calls for complex sub-tasks."""

    def get_model_for_depth(self) -> str:
        """
        Get appropriate model for current depth.

        Implements: Spec §5.3 Model Selection by Depth

        Depth 0 (root): Opus (most capable)
        Depth 1: Sonnet (balanced)
        Depth 2+: Haiku (fast, cost-effective)

        May be overridden by adaptive depth budgeting (Feature 3e0.4).
        """
        # Use override if set by adaptive depth
        if self._model_override is not None:
            return self._model_override
        return self.router.get_model(self.depth)

    def inject_tool_result(self, result: Any) -> None:
        """
        Inject tool execution result back into REPL state.

        Args:
            result: Tool execution result
        """
        if self.repl:
            self.repl.inject_result("_tool_result", result)
            self.repl.globals["tool_outputs"].append(
                {
                    "tool": "injected",
                    "content": str(result),
                }
            )

    def get_aggregated_results(self) -> list[RecursiveCallResult]:
        """
        Get all results from this REPL and its children.

        Returns:
            Flattened list of all recursive call results
        """
        results = self._results.copy()
        for child in self.child_repls:
            results.extend(child.get_aggregated_results())
        return results

    def get_cost_summary(self) -> dict[str, Any]:
        """
        Get cost summary for this REPL tree.

        Returns:
            Dict with token counts, call counts, and model usage
        """
        model_usage: dict[str, int] = {}

        def collect_model_usage(repl: RecursiveREPL) -> None:
            model = repl.get_model_for_depth()
            model_usage[model] = model_usage.get(model, 0) + repl._tokens_used
            for child in repl.child_repls:
                collect_model_usage(child)

        collect_model_usage(self)

        return {
            "total_tokens": self.total_tokens_used,
            "total_calls": self.total_recursive_calls,
            "max_depth_reached": self._get_max_depth_reached(),
            "model_usage": model_usage,
        }

    def _get_max_depth_reached(self) -> int:
        """Get the maximum depth reached in this REPL tree."""
        max_depth = self.depth
        for child in self.child_repls:
            child_max = child._get_max_depth_reached()
            if child_max > max_depth:
                max_depth = child_max
        return max_depth


__all__ = ["RecursiveREPL"]
