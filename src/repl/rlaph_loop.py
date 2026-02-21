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
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterable

# Suppress third-party warnings at import time
warnings.filterwarnings("ignore", category=UserWarning, module="cpmpy")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="RestrictedPython")

from ..config import RLMConfig, default_config
from ..frame.causal_frame import CausalFrame, FrameStatus, compute_frame_id, generate_invalidation_condition
from ..frame.intent_extractor import extract_canonical_task
from ..frame.context_map import ContextMap
from ..frame.context_slice import ContextSlice
from ..frame.frame_index import FrameIndex
from ..types import RecursionDepthError, SessionContext
from .llm_client import LLMClient
from .repl_environment import RLMEnvironment
from .response_parser import ResponseAction, ResponseParser
from .json_parser import parse_llm_response, StructuredResponse

if TYPE_CHECKING:
    pass  # No external type imports needed for v2


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
        llm_client: LLMClient | None = None,
        context_map: ContextMap | None = None,
        verbose: bool = False,
    ):
        """
        Initialize RLAPH loop.

        Args:
            max_iterations: Maximum loop iterations
            max_depth: Maximum recursion depth for llm() calls
            config: RLM configuration
            llm_client: LLM client for API calls
            context_map: Optional ContextMap for externalized file access (SPEC-17)
            verbose: Enable verbose logging for recursion decisions
        """
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.config = config or default_config
        self.llm_client = llm_client or LLMClient()
        self._context_map = context_map
        self._verbose = verbose

        # State
        self.repl: RLMEnvironment | None = None
        self.history: list[dict] = []
        self._depth = 0
        self._tokens_used = 0

        # Parser
        self.parser = ResponseParser()

        # CausalFrame tracking (SPEC-17)
        self.frame_index: FrameIndex = FrameIndex()
        self._current_frame_id: str | None = None

        # Verification tracking - files accessed in last execution
        self._last_repl_files_accessed: list[str] = []

    def _collect_evidence(self) -> list[str]:
        """
        Collect frame IDs that current execution depends on.

        NOTE: This is a stub implementation. Full implementation requires
        AST analysis or variable tracking to detect which frames' results
        were used in current code.
        """
        # TODO: Implement proper evidence collection
        # For now, return empty list - evidence must be added manually
        return []

    @property
    def depth(self) -> int:
        """Current recursion depth."""
        return self._depth

    @property
    def total_tokens_used(self) -> int:
        """Total tokens used."""
        return self._tokens_used

    async def run(
        self,
        query: str,
        context: SessionContext,
        working_dir: Path | str | None = None,
        session_id: str | None = None,
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
            session_id: Optional session identifier for persistence (default: auto-generated UUID[:8])

        Returns:
            RLPALoopResult with answer and metadata
        """
        # Generate session_id if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        start_time = time.time()
        state = RLPALoopState(
            max_turns=self.max_iterations,
            depth=self._depth,
        )

        # Initialize REPL with LLM client, ContextMap, and loop reference for v2
        self.repl = RLMEnvironment(
            context,
            llm_client=self.llm_client,
            context_map=self._context_map,
            loop=self,  # Phase 18 A1: Pass loop for llm_sync integration
        )

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

        # Main loop
        while state.turn < state.max_turns and state.final_answer is None:
            state.turn += 1

            # Get LLM response - convert messages to prompt
            model = self._get_model_for_depth()
            prompt = self._messages_to_prompt(state.messages)
            response_content = self.llm_client.call(
                query=prompt,
                model=model,
                system=system_prompt,
                max_tokens=16384,
                depth=self._depth,
            )

            # Track tokens
            self._tokens_used += len(response_content) // 4  # Rough estimate

            # Parse response
            parsed_items = self.parser.parse(response_content)

            # Process each parsed item
            for item in parsed_items:
                if item.action == ResponseAction.FINAL_ANSWER:
                    state.final_answer = item.content
                    # Verify the answer for hallucinations before committing
                    is_valid, reason = self._verify_result(state.final_answer)
                    if not is_valid:
                        # Verification failed - clear answer and continue loop
                        print(f"[RLAPH] Verification failed: {reason}")
                        state.messages.append({
                            "role": "user",
                            "content": f"[VERIFICATION FAILED] {reason}\n\nYour previous answer contained errors. Please try again with accurate information based on the actual code execution results."
                        })
                        state.final_answer = None  # Clear to continue loop
                        break  # Exit this iteration to continue loop
                    # Answer verified - proceed to return
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

                    # Execute code synchronously
                    exec_result = self.repl.execute(code)
                    repl_result = (
                        exec_result.output if exec_result.success else f"Error: {exec_result.error}"
                    )

                    # === POST-HOC FRAME CREATION (SPEC-17) ===
                    context_slice = ContextSlice(
                        files=dict(self.repl.files_read),
                        memory_refs=list(self.repl.memory_refs),
                        tool_outputs=dict(self.repl.tool_outputs_tracked),
                        token_budget=self.config.cost_controls.max_tokens_per_recursive_call,
                    )

                    # Parse LLM response for structured conclusion and confidence
                    parsed = parse_llm_response(response_content)

                    frame = CausalFrame(
                        frame_id=compute_frame_id(
                            self._current_frame_id,
                            code[:100],  # Use code snippet as query
                            context_slice,
                        ),
                        depth=self._depth,
                        parent_id=self._current_frame_id,
                        children=[],
                        query=code[:200],
                        context_slice=context_slice,
                        evidence=self._collect_evidence(),
                        conclusion=parsed.conclusion[:500] if parsed.conclusion else str(exec_result.output)[:500],
                        confidence=parsed.confidence,
                        invalidation_condition=generate_invalidation_condition(context_slice),
                        status=FrameStatus.COMPLETED if exec_result.success else FrameStatus.INVALIDATED,
                        canonical_task=extract_canonical_task(code[:200], self.llm_client, use_llm_fallback=False),
                        branched_from=None,
                        escalation_reason=exec_result.error if not exec_result.success else None,
                        created_at=datetime.now(),
                        completed_at=datetime.now(),
                    )

                    self.frame_index.add(frame)
                    self._current_frame_id = frame.frame_id

                    # Track files accessed for verification
                    self._last_repl_files_accessed = list(self.repl.files_read.keys())

                    # Clear tracking for next frame
                    self.repl.files_read.clear()
                    self.repl.tool_outputs_tracked.clear()
                    self.repl.memory_refs.clear()
                    # === END FRAME CREATION ===

                    # === HANDLE SUB_TASKS FOR EXPLICIT RECURSION (Phase 18.5 Task 5) ===
                    if parsed.should_continue and self._depth < self.max_depth:
                        # Execute sub-tasks in order of priority
                        sorted_tasks = sorted(parsed.sub_tasks, key=lambda t: t.get("priority", 999))

                        for task in sorted_tasks:
                            sub_query = task.get("query")
                            if sub_query:
                                if self._verbose:
                                    print(f"[RLM] Auto-recursing to sub-task: {sub_query[:60]}...")
                                sub_result = self.llm_sync(sub_query)
                                # Sub-result is tracked as child frame automatically

                    elif parsed.should_ask_user:
                        # Agent needs clarification - log it
                        if self._verbose:
                            print(f"[RLM] Agent requesting user input: {parsed.reasoning[:100]}...")
                    # === END SUB_TASKS HANDLING ===

                    # Truncate REPL output
                    MAX_REPL_OUTPUT = 1500
                    repl_result_str = str(repl_result) if repl_result else ""
                    truncated_result = repl_result_str[:MAX_REPL_OUTPUT]
                    if len(repl_result_str) > MAX_REPL_OUTPUT:
                        truncated_result += f"\n... [truncated, {len(repl_result_str)} chars total]"

                    # Add to conversation
                    state.messages.append({"role": "assistant", "content": response_content})
                    state.messages.append(
                        {
                            "role": "user",
                            "content": f"[SYSTEM - Code execution result]:\n```\n{truncated_result}\n```\n\nYou MUST now provide your JSON response with \"next_action\" based on these results. Use \"finalize\" if you have the complete answer, or \"continue\" if more work is needed.",
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

                    state.messages.append({"role": "assistant", "content": response_content})
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

        # Save frame index before returning (Phase 13: Persistence)
        if len(self.frame_index) > 0:
            try:
                self.frame_index.save(session_id)
            except Exception:
                # Don't fail the run if save fails
                pass

        return RLPALoopResult(
            answer=state.final_answer or "No answer produced",
            iterations=state.turn,
            depth_used=self._depth,
            tokens_used=self.total_tokens_used,
            execution_time_ms=execution_time,
            history=self.history.copy(),
        )

    def llm_sync(self, query: str, context: str = "", depth: int | None = None) -> str:
        """
        Synchronous LLM call - returns actual result immediately.

        This is the key method that makes RLAPH work:
        - Called from REPL's llm() function
        - Returns actual string result
        - Handles depth management
        - Creates child frames with explicit depth parameter

        Args:
            query: Query string
            context: Optional context string
            depth: Explicit depth for this call (None = parent_depth + 1)

        Returns:
            LLM response as string

        Raises:
            RecursionDepthError: If max depth exceeded
        """
        # Calculate depth from parent frame (not self._depth)
        if depth is not None:
            current_depth = depth
        elif self._current_frame_id:
            parent_frame = self.frame_index.get(self._current_frame_id)
            current_depth = (parent_frame.depth + 1) if parent_frame else 1
        else:
            current_depth = 1  # First child of root

        # Check depth limit
        if current_depth > self.max_depth:
            raise RecursionDepthError(current_depth, self.max_depth)

        # Verbose logging for recursion decisions
        if self._verbose:
            print(f"[RLM] Recursion at depth {current_depth}: {query[:80]}...")

        # Use LLMClient directly for v2
        result = self.llm_client.call(
            query=query,
            context={"prior": context} if context else None,
            depth=current_depth,
        )

        # Parse the response to extract structured data
        parsed = parse_llm_response(result)

        # Create child frame for this llm call
        context_slice = ContextSlice(
            files={},  # No files read at llm call level
            memory_refs=[],
            tool_outputs={},
            token_budget=self.config.cost_controls.max_tokens_per_recursive_call,
        )

        child_frame = CausalFrame(
            frame_id=compute_frame_id(
                self._current_frame_id,
                query[:100],  # Use query snippet as identifier
                context_slice,
            ),
            depth=current_depth,
            parent_id=self._current_frame_id,
            children=[],
            query=query[:200],
            context_slice=context_slice,
            evidence=self._collect_evidence(),
            conclusion=parsed.conclusion[:500] if parsed.conclusion else (result[:500] if result else None),
            confidence=parsed.confidence,
            invalidation_condition=generate_invalidation_condition(context_slice),
            status=FrameStatus.COMPLETED,
            canonical_task=extract_canonical_task(query, self.llm_client, use_llm_fallback=False),
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

        self.frame_index.add(child_frame)

        return result

    def _build_system_prompt(self) -> str:
        """Build system prompt with REPL instructions and recursion guidance."""
        return f"""You are an RLM (Recursive Language Model) agent with access to a REAL Python REPL.

CRITICAL OUTPUT FORMAT:
You MUST respond in valid JSON only. No other text before or after.
The response must be raw JSON only, starting with {{ and ending with }}.
Do NOT wrap in ```json code blocks.

Use this exact schema:

{{
  "reasoning": "Your step-by-step thinking here",
  "conclusion": "Final answer or summary",
  "confidence": 0.0 to 1.0,
  "files": ["relevant/file.py", ...] or [],
  "sub_tasks": [{{"query": "sub query", "priority": 1}}] or [],
  "needs_more_info": true or false,
  "next_action": "continue" or "finalize" or "ask_user"
}}

"next_action": Must be one of:
- "continue" (more sub-tasks needed)
- "finalize" (ready for final answer)
- "ask_user" (need clarification)

If you cannot follow this format, output: {{"error": "invalid format", "conclusion": "your answer"}}

CRITICAL RULES:
1. ALWAYS respond in valid JSON format with the required schema - no other text before or after
2. Do NOT wrap JSON in ```json code blocks - output raw JSON only
3. When you write code in ```python blocks, the system EXECUTES it and returns REAL output
4. DO NOT generate fake "REPL output" or "Human:" messages yourself
5. DO NOT pretend to see execution results - wait for the actual system response
6. After writing code, STOP and wait for the [SYSTEM - Code execution result]
7. When you have the final answer, output JSON with "next_action": "finalize"
8. NEVER use import statements - they are blocked. Pre-loaded: hashlib, json, re, os, sys

RECURSION - DECOMPOSE COMPLEX TASKS:
You can call llm(sub_query) to delegate sub-tasks. This creates a CHILD FRAME.
- Max recursion depth: {self.max_depth}
- Use llm(sub_query) for parallel/branching analysis
- Each llm() call is tracked as a child frame
- Example: For codebase summary, first glob files, then llm("summarize auth/*.py")

IMPORTANT RECURSION RULES:
- Only call llm(sub_query) when the sub-task is meaningfully independent or parallelizable
- Do NOT call llm() for tiny steps — that wastes depth budget
- Always prefer small, focused sub-queries (1–3 sentences)
- If you're unsure, try simple code first, then recurse if needed

Your workflow:
1. Write Python code in ```python blocks
2. STOP - the system will execute and return [SYSTEM - Code execution result]
3. Read the REAL output from the system
4. For complex tasks, use llm(sub_query) to decompose
5. ALWAYS end with JSON output including "next_action": "finalize" when done

MANDATORY: After all sub-tasks complete, you MUST synthesize and output JSON:
{{
  "reasoning": "Your synthesis reasoning",
  "conclusion": "<complete answer combining all results>",
  "confidence": 0.9,
  "next_action": "finalize"
}}

Do NOT leave the answer incomplete. Do NOT end without proper JSON output.

Pre-loaded Libraries (NO import needed):
- hashlib: Use directly as `hashlib.sha256(data.encode()).hexdigest()`
- json: Use directly as `json.loads()`, `json.dumps()`
- re: Use directly for regex operations

File Access Functions:
- `read_file(path, offset=0, limit=2000)`: Read file content from disk
- `glob_files(pattern)`: Find files matching pattern
- `grep_files(pattern, path)`: Search for pattern in files
- `list_dir(path)`: List directory contents

Recursion Function:
- `llm(query)`: Call LLM with sub-query, returns result string
  - Creates child frame at depth+1
  - Use for task decomposition
  - Keep sub-queries focused (1-3 sentences)

Example with Recursion:
User: Analyze the auth module architecture

Your response:
```python
auth_files = glob_files("src/auth/**/*.py")
print(f"Found {{len(auth_files)}} auth files")
```
[STOP - wait for system]

System returns: Found 5 auth files

Your response:
```python
# Decompose into parallel sub-analyses (each is independent)
auth_summary = llm("Summarize the main authentication flow in src/auth/login.py")
oauth_summary = llm("Summarize the OAuth implementation in src/auth/oauth.py")
print(f"Auth: {{auth_summary[:100]}}...")
print(f"OAuth: {{oauth_summary[:100]}}...")
```
[STOP - wait for system]

System returns results from child frames

Your response:
{{
  "reasoning": "Combined analysis of auth files shows login.py handles main auth, oauth.py implements OAuth 2.0",
  "conclusion": "The auth module consists of login.py (main flow) and oauth.py (OAuth 2.0)...",
  "confidence": 0.9,
  "next_action": "finalize"
}}

Other functions: peek(), search(), summarize(), llm_batch(), map_reduce()
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

    def _verify_result(self, answer: str) -> tuple[bool, str]:
        """
        Verify the LLM's final answer for obvious hallucinations.

        Detects:
        - Non-existent file paths
        - Obviously fake hash patterns
        - Claims about files never accessed

        Returns:
            (is_valid, reason) tuple
        """
        import re
        from pathlib import Path

        # Pattern 1: Detect file paths in answer and verify they exist
        file_pattern = r'(?:src|tests|scripts)[/\w\-\.]+\.py(?::\d+)?'
        file_matches = re.findall(file_pattern, answer)

        for file_match in file_matches:
            # Extract just the path (remove line number)
            path = file_match.split(':')[0]
            # Check relative to working dir if available
            if hasattr(self.repl, '_working_dir') and self.repl._working_dir:
                full_path = Path(self.repl._working_dir) / path
            else:
                full_path = Path(path)

            if not full_path.exists():
                return (False, f"Hallucinated file path: {path} does not exist")

        # Pattern 2: Detect obviously fake hashes (sequential hex)
        fake_hash_pattern = r'[a-f0-9]{16}'
        hash_matches = re.findall(fake_hash_pattern, answer.lower())

        for hash_match in hash_matches:
            if self._is_sequential_pattern(hash_match):
                return (False, f"Obviously fake hash pattern: {hash_match}")

        # Pattern 3: If answer mentions a file, check if we actually accessed it
        if hasattr(self, '_last_repl_files_accessed') and file_matches:
            for file_match in file_matches:
                path = file_match.split(':')[0]
                # Only validate if we tracked file access
                if self._last_repl_files_accessed and path not in self._last_repl_files_accessed:
                    # Only flag if it's a specific file claim, not general mention
                    if f"{path}:" in answer or f'"{path}"' in answer or f"'{path}'" in answer:
                        return (False, f"Answer mentions {path} but file was never accessed")

        return (True, "Answer verified")

    def _is_sequential_pattern(self, s: str) -> bool:
        """Check if string is an obviously fake sequential pattern."""
        sequential = ['a1b2c3d4e5f6a7b8', '12345678', 'abcdefgh']
        s_lower = s.lower()
        if s_lower in sequential:
            return True

        # Check for simple alternation patterns (aaaa, 1212)
        if len(s) >= 4:
            if all(s_lower[i] == s_lower[i % 2] for i in range(len(s_lower))):
                return True

        return False

    async def _read_files_parallel(self, paths: list[str]) -> dict[str, str]:
        """
        Read multiple files in parallel using async I/O.

        This is used when the REPL needs to read multiple files
        for context externalization.

        Args:
            paths: List of file paths to read

        Returns:
            Dict mapping path -> content
        """
        if not self.repl:
            return {}

        return await self.repl.read_files_async(paths)


async def run_rlaph(
    query: str,
    working_dir: Path | str | None = None,
    session_id: str | None = None,
    max_depth: int = 3,
    verbose: bool = False,
    context: SessionContext | None = None,
    stream: bool = False,
) -> RLPALoopResult:
    """
    Library interface for running RLAPH loop.

    Provides a simple, convenient way to run RLAPH without manually
    creating an RLAPHLoop instance. This is the recommended way to use
    RLAPH as a library.

    Args:
        query: The query/prompt string
        working_dir: Working directory for the loop (default: cwd)
        session_id: Optional session ID for frame persistence (default: auto-generated UUID[:8])
        max_depth: Maximum recursion depth (default: 3)
        verbose: Enable verbose logging (default: False)
        context: Optional session context (default: empty SessionContext)
        stream: Reserved for future streaming support (currently ignored)

    Returns:
        RLPALoopResult with the answer and execution metadata

    Examples:
        >>> # Simple usage
        >>> result = await run_rlaph("Analyze the auth flow")
        >>> print(result.answer)

        >>> # With custom depth and verbosity
        >>> result = await run_rlaph("Summarize the codebase", max_depth=5, verbose=True)

        >>> # With custom context
        >>> ctx = SessionContext(files={"auth.py": "...", "login.py": "..."})
        >>> result = await run_rlaph("Review auth security", context=ctx)

    Note:
        The `stream` parameter is reserved for future implementation.
        When True, streaming will yield chunks in real-time for better UX.
    """
    loop = RLAPHLoop(max_depth=max_depth, verbose=verbose)
    ctx = context or SessionContext()
    wd = Path(working_dir) if working_dir else Path.cwd()

    # Run the loop and return the result
    # Note: Future tasks will add proper streaming support when stream=True
    return await loop.run(query, ctx, wd, session_id)


async def run_rlaph_stream(
    query: str,
    working_dir: Path | str | None = None,
    session_id: str | None = None,
    max_depth: int = 3,
    verbose: bool = False,
    context: SessionContext | None = None,
) -> AsyncIterable[str]:
    """
    Streaming variant of run_rlaph that yields chunks in real-time.

    This is a convenience wrapper that runs the RLAPH loop and streams
    the final answer in chunks. For true streaming during execution,
    future tasks will add support for yielding LLM responses as they arrive.

    Args:
        query: The query/prompt string
        working_dir: Working directory for the loop (default: cwd)
        session_id: Optional session ID for frame persistence (default: auto-generated UUID[:8])
        max_depth: Maximum recursion depth (default: 3)
        verbose: Enable verbose logging (default: False)
        context: Optional session context (default: empty SessionContext)

    Yields:
        Chunks of the final answer as they become available

    Examples:
        >>> async for chunk in run_rlaph_stream("Explain this code"):
        ...     print(chunk, end="")
    """
    loop = RLAPHLoop(max_depth=max_depth, verbose=verbose)
    ctx = context or SessionContext()
    wd = Path(working_dir) if working_dir else Path.cwd()

    # Run the loop
    result = await loop.run(query, ctx, wd, session_id)

    # Yield the answer in chunks for real-time display
    chunk_size = 100  # Chunk size for streaming output
    for i in range(0, len(result.answer), chunk_size):
        yield result.answer[i:i + chunk_size]


__all__ = [
    "RLAPHLoop",
    "RLPALoopResult",
    "RLPALoopState",
    "run_rlaph",
    "run_rlaph_stream",
]
