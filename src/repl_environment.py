"""
REPL environment for RLM context manipulation.

Implements: Spec §4 REPL Environment Design
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import TYPE_CHECKING, Any

from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    safer_getattr,
)

from .tokenization import partition_content_by_tokens
from .types import DeferredBatch, DeferredOperation, ExecutionResult, SessionContext

if TYPE_CHECKING:
    from .memory_store import MemoryStore
    from .recursive_handler import RecursiveREPL

# Subprocess allowlist for sandbox
ALLOWED_SUBPROCESSES = frozenset({"ty", "ruff"})

# Blocked builtins that could be dangerous
BLOCKED_BUILTINS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
    }
)


class RLMSecurityError(Exception):
    """Raised when sandbox security is violated."""

    pass


class _REPLPrintCollector:
    """Print collector for RestrictedPython that captures output to a buffer."""

    def __init__(self, buffer: list[str]):
        self._buffer = buffer

    def _call_print(self, *args: Any, **kwargs: Any) -> None:
        """Called by RestrictedPython for print() statements."""
        output = " ".join(str(arg) for arg in args)
        self._buffer.append(output)

    def write(self, text: str) -> None:
        """Write method for compatibility."""
        self._buffer.append(text)


class RLMEnvironment:
    """
    Sandboxed REPL for context manipulation.

    Implements: Spec §4.1 Sandbox Architecture

    The REPL provides:
    - Context variables (conversation, files, tool_outputs, working_memory)
    - Helper functions (peek, search, summarize, recursive_query)
    - Safe stdlib (re, json, string operations)
    - Extended tooling (pydantic, hypothesis, cpmpy)

    Security:
    - Uses RestrictedPython for code compilation
    - Blocks dangerous builtins (open, exec, eval, __import__)
    - Subprocess calls limited to allowlist (ty, ruff)
    """

    def __init__(
        self,
        context: SessionContext,
        recursive_handler: RecursiveREPL | None = None,
        use_restricted: bool = True,
    ):
        """
        Initialize REPL environment with context.

        Args:
            context: Session context to externalize
            recursive_handler: Handler for recursive calls (depth>0)
            use_restricted: Whether to use RestrictedPython (default True)

        Implements: Spec §4.1 Sandbox Architecture
        """
        self.context = context
        self.recursive_handler = recursive_handler
        self.use_restricted = use_restricted

        # Build safe builtins
        self._safe_builtins = self._build_safe_builtins()

        # Build globals with context and helpers
        self.globals: dict[str, Any] = {
            # Module-level names needed for class definitions
            "__name__": "__rlm_repl__",
            "__doc__": None,
            # Builtins
            "__builtins__": self._safe_builtins,
            # RestrictedPython guards
            "_getiter_": default_guarded_getiter,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_getattr_": safer_getattr,
            # Context variables
            "conversation": [
                {"role": m.role.value, "content": m.content} for m in context.messages
            ],
            "files": context.files.copy(),
            "tool_outputs": [
                {"tool": o.tool_name, "content": o.content} for o in context.tool_outputs
            ],
            "working_memory": context.working_memory.copy(),
            # Helper functions
            "peek": self._peek,
            "search": self._search,
            "summarize": self._summarize,
            "recursive_query": self._recursive_query,
            "recursive_llm": self._recursive_query,
            "llm": self._recursive_query,  # Shorthand alias
            "llm_batch": self._llm_batch,  # Parallel LLM calls
            # Advanced REPL functions (SPEC-01)
            "map_reduce": self._map_reduce,
            "find_relevant": self._find_relevant,
            "extract_functions": self._extract_functions,
            # Safe subprocess execution
            "run_tool": self._run_tool,
            # Standard library (safe modules)
            "re": re,
            "json": json,
        }

        # Add extended tooling
        self._add_extended_tooling()

        # Local namespace for user variables
        self.locals: dict[str, Any] = {}

        # Execution history for debugging
        self.history: list[dict[str, Any]] = []

        # Pending async operations (deferred until after sync execution)
        self.pending_operations: list[DeferredOperation] = []
        self.pending_batches: list[DeferredBatch] = []
        self._operation_counter = 0

        # Memory store (initialized via enable_memory())
        self._memory_store: MemoryStore | None = None

    def _build_safe_builtins(self) -> dict[str, Any]:
        """
        Build restricted builtins dict.

        Implements: Spec §4.1 Security Constraints
        """
        # Start with RestrictedPython's safe_builtins
        builtins = dict(safe_builtins)

        # Add common safe functions
        safe_additions = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "range": range,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "pow": pow,
            "divmod": divmod,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "hasattr": hasattr,
            "getattr": getattr,
            "type": type,
            "repr": repr,
            "hash": hash,
            "id": id,
            "ord": ord,
            "chr": chr,
            "hex": hex,
            "bin": bin,
            "oct": oct,
            "format": format,
            "slice": slice,
            "print": self._safe_print,  # Captured print
        }

        builtins.update(safe_additions)

        # Explicitly remove dangerous builtins
        for blocked in BLOCKED_BUILTINS:
            builtins.pop(blocked, None)

        return builtins

    def _add_extended_tooling(self) -> None:
        """
        Add pydantic, hypothesis, and cpmpy to globals.

        Implements: Spec §4.1.1 Extended Python Tooling
        """
        try:
            import pydantic

            self.globals["pydantic"] = pydantic
            # Common pydantic imports for convenience
            self.globals["BaseModel"] = pydantic.BaseModel
            self.globals["Field"] = pydantic.Field
        except ImportError:
            pass

        try:
            import hypothesis
            from hypothesis import strategies as st

            self.globals["hypothesis"] = hypothesis
            self.globals["given"] = hypothesis.given
            self.globals["st"] = st
        except ImportError:
            pass

        try:
            import cpmpy

            self.globals["cp"] = cpmpy
            self.globals["cpmpy"] = cpmpy
        except ImportError:
            pass

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute code in sandbox.

        Implements: Spec §4.1 Sandbox Architecture

        Uses eval-first approach: tries to evaluate code as an expression first
        to capture return values (e.g., `1+1` returns `2`). Falls back to exec
        mode for statements (e.g., `x = 1`).

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with output or error
        """
        start_time = time.time()
        output_capture: list[str] = []

        # Set up print capture
        self._print_buffer = output_capture

        # Create print collector for RestrictedPython
        print_collector = _REPLPrintCollector(output_capture)
        self.globals["_print"] = print_collector
        self.globals["_print_"] = lambda _: print_collector

        try:
            result = None

            if self.use_restricted:
                # Try eval mode first for expression values
                try:
                    byte_code = compile_restricted(
                        code,
                        filename="<repl>",
                        mode="eval",
                    )
                    result = eval(byte_code, self.globals, self.locals)
                except SyntaxError:
                    # Fall back to exec mode for statements
                    byte_code = compile_restricted(
                        code,
                        filename="<repl>",
                        mode="exec",
                    )
                    exec(byte_code, self.globals, self.locals)
            else:
                # Fallback: use regular compile (for testing)
                try:
                    compiled = compile(code, "<repl>", "eval")
                    result = eval(compiled, self.globals, self.locals)
                except SyntaxError:
                    # Fall back to exec mode for statements
                    compiled = compile(code, "<repl>", "exec")
                    exec(compiled, self.globals, self.locals)

            # Use eval result if we got one, otherwise check for _ or print output
            output = result
            if output is None:
                output = self.locals.get("_")
            if output is None and output_capture:
                output = "\n".join(output_capture)

            execution_time = (time.time() - start_time) * 1000

            # Record in history
            self.history.append(
                {
                    "code": code,
                    "success": True,
                    "output": output,
                    "time_ms": execution_time,
                }
            )

            return ExecutionResult(
                success=True,
                output=output,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            # Record in history
            self.history.append(
                {
                    "code": code,
                    "success": False,
                    "error": str(e),
                    "time_ms": execution_time,
                }
            )

            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )
        finally:
            self._print_buffer = []

    def _safe_print(self, *args: Any, **_kwargs: Any) -> None:
        """Capture print output instead of writing to stdout."""
        output = " ".join(str(arg) for arg in args)
        if hasattr(self, "_print_buffer"):
            self._print_buffer.append(output)

    def _peek(self, var: Any, start: int = 0, end: int = 1000) -> Any:
        """
        View a portion of a context variable.

        Implements: Spec §3.2 Root Prompt Structure

        Args:
            var: Variable to peek into
            start: Start index
            end: End index

        Returns:
            Sliced portion of the input - str for strings, list slice for lists,
            dict subset for dicts, str representation slice for other types.
        """
        if isinstance(var, (str, list)):
            return var[start:end]
        elif isinstance(var, dict):
            items = list(var.items())[start:end]
            return dict(items)
        else:
            return str(var)[start:end]

    def _search(self, var: Any, pattern: str, regex: bool = False) -> list[dict[str, Any]]:
        """
        Search for patterns in context variable.

        Implements: Spec §3.2 Root Prompt Structure

        Args:
            var: Variable to search in
            pattern: Pattern to search for
            regex: If True, treat pattern as regex

        Returns:
            List of match results with location info
        """
        results: list[dict[str, Any]] = []

        if regex:
            compiled = re.compile(pattern)

            def matcher(s: str) -> bool:
                return bool(compiled.search(s))
        else:

            def matcher(s: str) -> bool:
                return pattern.lower() in s.lower()

        if isinstance(var, str):
            if matcher(var):
                results.append({"match": var[:500], "type": "string"})
        elif isinstance(var, list):
            for i, item in enumerate(var):
                content = str(item) if not isinstance(item, str) else item
                if matcher(content):
                    results.append({"index": i, "match": content[:200]})
        elif isinstance(var, dict):
            for key, value in var.items():
                content = str(value) if not isinstance(value, str) else value
                if matcher(content) or matcher(str(key)):
                    results.append({"key": key, "match": content[:200]})

        return results

    def _summarize(self, var: Any, max_tokens: int = 500) -> DeferredOperation:
        """
        Summarize context variable via sub-call.

        Implements: Spec §4.2 Recursive Call Implementation

        Args:
            var: Variable to summarize
            max_tokens: Max tokens for summary

        Returns:
            DeferredOperation that will be resolved by orchestrator
        """
        self._operation_counter += 1
        op_id = f"sum_{self._operation_counter}"

        content = str(var)
        if len(content) > max_tokens * 4:
            content = content[: max_tokens * 4] + "..."

        op = DeferredOperation(
            operation_id=op_id,
            operation_type="summarize",
            query=f"Summarize this content in {max_tokens} tokens or less",
            context=content,
            spawn_repl=False,
        )
        self.pending_operations.append(op)
        return op

    def _recursive_query(
        self,
        query: str,
        context: Any = None,
        spawn_repl: bool = False,
    ) -> DeferredOperation:
        """
        Spawn a recursive sub-query.

        Implements: Spec §4.2 Recursive Call Implementation

        Args:
            query: Query string for sub-call
            context: Context to pass to sub-call (optional)
            spawn_repl: If True, child gets its own REPL

        Returns:
            DeferredOperation that will be resolved by orchestrator
        """
        self._operation_counter += 1
        op_id = f"rq_{self._operation_counter}"

        # Convert context to string if not None
        context_str = str(context)[:8000] if context is not None else ""

        op = DeferredOperation(
            operation_id=op_id,
            operation_type="recursive_query",
            query=query,
            context=context_str,
            spawn_repl=spawn_repl,
        )
        self.pending_operations.append(op)
        return op

    def _llm_batch(
        self,
        queries: list[tuple[str, Any]],
        spawn_repl: bool = False,
    ) -> DeferredBatch:
        """
        Execute multiple LLM queries in parallel.

        Implements: Spec §4.2 Parallel Sub-Calls

        Args:
            queries: List of (query, context) tuples
            spawn_repl: If True, each query gets its own REPL

        Returns:
            DeferredBatch that will be resolved by orchestrator

        Example:
            results = llm_batch([
                ("Analyze auth module", auth_code),
                ("Analyze db module", db_code),
                ("Analyze api module", api_code),
            ])
        """
        self._operation_counter += 1
        batch_id = f"batch_{self._operation_counter}"

        batch = DeferredBatch(batch_id=batch_id)

        for query, context in queries:
            self._operation_counter += 1
            op_id = f"rq_{self._operation_counter}"
            context_str = str(context)[:8000] if context is not None else ""

            op = DeferredOperation(
                operation_id=op_id,
                operation_type="recursive_query",
                query=query,
                context=context_str,
                spawn_repl=spawn_repl,
            )
            batch.operations.append(op)

        self.pending_batches.append(batch)
        return batch

    def _map_reduce(
        self,
        content: str,
        map_prompt: str,
        reduce_prompt: str,
        n_chunks: int = 4,
        model: str = "auto",
    ) -> DeferredBatch:
        """
        Apply map-reduce pattern to large content.

        Implements: Spec SPEC-01.01 through SPEC-01.06

        Args:
            content: Content to process
            map_prompt: Prompt to apply to each chunk (can use {chunk} placeholder)
            reduce_prompt: Prompt to combine map results
            n_chunks: Number of chunks to split content into
            model: Model tier to use ("fast", "balanced", "powerful", "auto")

        Returns:
            DeferredBatch with map operations and reduce metadata
        """
        # Validate model parameter (SPEC-01.05)
        valid_models = {"fast", "balanced", "powerful", "auto"}
        if model not in valid_models:
            raise ValueError(f"Invalid model: {model}. Must be one of: {valid_models}")

        # Handle empty content (SPEC-01.06 graceful handling)
        if not content:
            content = ""

        # Partition content into n_chunks (SPEC-01.02)
        chunks = self._partition_content(content, n_chunks)

        # Create batch of map operations (SPEC-01.03)
        self._operation_counter += 1
        batch_id = f"mapreduce_{self._operation_counter}"

        batch = DeferredBatch(batch_id=batch_id)

        for _i, chunk in enumerate(chunks):
            self._operation_counter += 1
            op_id = f"map_{self._operation_counter}"

            # Format map_prompt with chunk if it has {chunk} placeholder
            if "{chunk}" in map_prompt:
                formatted_prompt = map_prompt.format(chunk=chunk)
            else:
                formatted_prompt = f"{map_prompt}\n\nContent:\n{chunk}"

            op = DeferredOperation(
                operation_id=op_id,
                operation_type="map",
                query=formatted_prompt,
                context=chunk,
                spawn_repl=False,
            )
            batch.operations.append(op)

        # Store reduce_prompt in batch metadata for later processing
        # The orchestrator will use this to create the reduce operation
        batch.metadata["reduce_prompt"] = reduce_prompt
        batch.metadata["model"] = model

        self.pending_batches.append(batch)
        return batch

    def _partition_content(self, content: str, n_chunks: int) -> list[str]:
        """
        Partition content into roughly equal token-sized chunks.

        Implements: Spec SPEC-01.02

        Uses token-aware chunking (SPEC-01.01) for accurate LLM context handling.
        Chunks at semantic boundaries (function/class definitions) when possible.

        Args:
            content: Content to partition
            n_chunks: Target number of chunks

        Returns:
            List of content chunks
        """
        return partition_content_by_tokens(content, n_chunks)

    def _find_relevant(
        self,
        content: str,
        query: str,
        top_k: int = 5,
        use_llm_scoring: bool = False,
    ) -> list[tuple[str, float]]:
        """
        Find sections most relevant to query.

        Implements: Spec SPEC-01.07 through SPEC-01.12

        Args:
            content: Content to search
            query: Query to find relevant sections for
            top_k: Number of top results to return
            use_llm_scoring: Whether to use LLM for scoring (if candidates > top_k * 2)

        Returns:
            List of (chunk, score) tuples sorted by relevance descending
        """
        if not content:
            return []

        # Partition into ~50-line chunks with 5-line overlap (SPEC-01.08)
        chunks = self._chunk_with_overlap(content, chunk_lines=50, overlap_lines=5)

        if not chunks:
            return []

        # Extract query keywords for filtering (SPEC-01.09)
        query_lower = query.lower()
        keywords = set(re.findall(r"\b\w{3,}\b", query_lower))

        # Keyword pre-filtering
        scored_chunks: list[tuple[str, float]] = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in chunk_lower)
            if matches > 0 or not keywords:
                # Score based on keyword density
                score = matches / max(len(keywords), 1)
                scored_chunks.append((chunk, score))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Optional LLM scoring if candidates exceed threshold (SPEC-01.10)
        if use_llm_scoring and len(scored_chunks) > top_k * 2:
            # Create deferred batch for LLM scoring
            # For now, we just use keyword scoring
            # LLM scoring would be implemented by creating a DeferredBatch
            # and processing results in the orchestrator
            pass

        # Return top_k results (SPEC-01.11)
        return scored_chunks[:top_k]

    def _chunk_with_overlap(
        self, content: str, chunk_lines: int = 50, overlap_lines: int = 5
    ) -> list[str]:
        """
        Split content into chunks with line overlap.

        Args:
            content: Content to chunk
            chunk_lines: Target lines per chunk
            overlap_lines: Lines to overlap between chunks

        Returns:
            List of overlapping chunks
        """
        lines = content.split("\n")
        if not lines:
            return []

        chunks = []
        start = 0

        while start < len(lines):
            end = min(start + chunk_lines, len(lines))
            chunk = "\n".join(lines[start:end])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            start = end - overlap_lines

            # Ensure we make progress and don't loop forever
            if end >= len(lines):
                break  # Reached end of content

        return chunks

    def _extract_functions(
        self,
        content: str,
        language: str = "python",
    ) -> list[dict[str, Any]]:
        """
        Extract function definitions from code.

        Implements: Spec SPEC-01.13 through SPEC-01.17

        Args:
            content: Source code content
            language: Programming language ("python", "go", "javascript", "typescript")

        Returns:
            List of dicts with keys: "name", "signature", "start_line", "end_line"
        """
        # Validate language (SPEC-01.14)
        valid_languages = {"python", "go", "javascript", "typescript"}
        if language not in valid_languages:
            raise ValueError(f"Invalid language: {language}. Must be one of: {valid_languages}")

        # Handle empty/malformed input gracefully (SPEC-01.17)
        if not content:
            return []

        # Get regex patterns for the language (SPEC-01.16)
        patterns = self._get_function_patterns(language)

        functions: list[dict[str, Any]] = []
        lines = content.split("\n")

        for pattern_info in patterns:
            pattern = pattern_info["pattern"]
            for match in re.finditer(pattern, content, re.MULTILINE):
                try:
                    name = match.group("name")
                    signature = match.group(0).strip()

                    # Calculate line numbers
                    # Note: \s* in pattern can match leading newlines, so we need
                    # to find where the actual function keyword starts
                    match_text = match.group(0)
                    leading_ws = len(match_text) - len(match_text.lstrip())
                    start_pos = match.start() + leading_ws
                    start_line = content[:start_pos].count("\n") + 1

                    # Estimate end line (simple heuristic)
                    end_line = self._estimate_function_end(lines, start_line - 1, language)

                    functions.append(
                        {
                            "name": name,
                            "signature": signature,
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    )
                except (IndexError, AttributeError):
                    # Handle malformed input gracefully (SPEC-01.17)
                    continue

        # Remove duplicates (same function might match multiple patterns)
        seen = set()
        unique_functions = []
        for func in functions:
            key = (func["name"], func["start_line"])
            if key not in seen:
                seen.add(key)
                unique_functions.append(func)

        # Sort by start_line
        unique_functions.sort(key=lambda x: x["start_line"])
        return unique_functions

    def _get_function_patterns(self, language: str) -> list[dict[str, Any]]:
        """
        Get regex patterns for function extraction by language.

        Implements: Spec SPEC-01.16
        """
        patterns: dict[str, list[dict[str, Any]]] = {
            "python": [
                {
                    "pattern": r"^\s*def\s+(?P<name>\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:",
                    "type": "function",
                },
                {
                    "pattern": r"^\s*async\s+def\s+(?P<name>\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:",
                    "type": "async_function",
                },
            ],
            "go": [
                {
                    "pattern": r"^func\s+(?P<name>\w+)\s*\([^)]*\)",
                    "type": "function",
                },
                {
                    "pattern": r"^func\s+\([^)]+\)\s+(?P<name>\w+)\s*\([^)]*\)",
                    "type": "method",
                },
            ],
            "javascript": [
                {
                    "pattern": r"^\s*function\s+(?P<name>\w+)\s*\([^)]*\)",
                    "type": "function",
                },
                {
                    "pattern": r"^\s*async\s+function\s+(?P<name>\w+)\s*\([^)]*\)",
                    "type": "async_function",
                },
                {
                    "pattern": r"^\s*(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
                    "type": "arrow_function",
                },
                {
                    "pattern": r"^\s*(?:const|let|var)\s+(?P<name>\w+)\s*=\s*function\s*\([^)]*\)",
                    "type": "function_expression",
                },
            ],
            "typescript": [
                {
                    "pattern": r"^\s*function\s+(?P<name>\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)",
                    "type": "function",
                },
                {
                    "pattern": r"^\s*async\s+function\s+(?P<name>\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)",
                    "type": "async_function",
                },
                {
                    "pattern": r"^\s*(?:const|let|var)\s+(?P<name>\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
                    "type": "arrow_function",
                },
                {
                    "pattern": r"^\s*(?:const|let|var)\s+(?P<name>\w+)\s*(?::\s*[^=]+)?\s*=\s*function\s*\([^)]*\)",
                    "type": "function_expression",
                },
            ],
        }
        return patterns.get(language, [])

    def _estimate_function_end(self, lines: list[str], start_idx: int, language: str) -> int:
        """
        Estimate the end line of a function.

        Simple heuristic based on indentation/braces.
        """
        if start_idx >= len(lines):
            return start_idx + 1

        if language == "python":
            # Find end by indentation
            start_line = lines[start_idx]
            base_indent = len(start_line) - len(start_line.lstrip())

            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                if not line.strip():  # Skip empty lines
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and line.strip():
                    return i  # Found line at same or lower indent
            return len(lines)
        else:
            # For brace-based languages, count braces
            brace_count = 0
            started = False

            for i in range(start_idx, len(lines)):
                line = lines[i]
                brace_count += line.count("{") - line.count("}")
                if "{" in line:
                    started = True
                if started and brace_count <= 0:
                    return i + 1

            return len(lines)

    def has_pending_operations(self) -> bool:
        """Check if there are pending async operations to process."""
        return bool(self.pending_operations) or bool(self.pending_batches)

    def get_pending_operations(self) -> tuple[list[DeferredOperation], list[DeferredBatch]]:
        """Get all pending operations for processing by orchestrator."""
        ops = self.pending_operations.copy()
        batches = self.pending_batches.copy()
        return ops, batches

    def clear_pending_operations(self) -> None:
        """Clear pending operations after they've been processed."""
        self.pending_operations.clear()
        self.pending_batches.clear()

    def resolve_operation(self, op_id: str, result: Any) -> None:
        """
        Resolve a pending operation with its result.

        Args:
            op_id: Operation ID to resolve
            result: Result to inject
        """
        # Check individual operations
        for op in self.pending_operations:
            if op.operation_id == op_id:
                op.resolved = True
                op.result = result
                # Inject into working memory for REPL access
                self.globals["working_memory"][op_id] = result
                return

        # Check batch operations
        for batch in self.pending_batches:
            for op in batch.operations:
                if op.operation_id == op_id:
                    op.resolved = True
                    op.result = result
                    self.globals["working_memory"][op_id] = result
                    return

    def resolve_batch(self, batch_id: str, results: list[Any]) -> None:
        """
        Resolve a batch of operations with their results.

        Args:
            batch_id: Batch ID to resolve
            results: List of results in order
        """
        for batch in self.pending_batches:
            if batch.batch_id == batch_id:
                batch.resolved = True
                batch.results = results
                # Resolve individual operations
                for op, result in zip(batch.operations, results, strict=False):
                    op.resolved = True
                    op.result = result
                    self.globals["working_memory"][op.operation_id] = result
                # Also store full batch results
                self.globals["working_memory"][batch_id] = results
                return

    def _run_tool(
        self,
        tool: str,
        *args: str,
        timeout: float = 30.0,
        stdin_input: str | None = None,
    ) -> dict[str, Any]:
        """
        Run an allowed subprocess tool.

        Implements: Spec §4.1.1 CLI Tools (via subprocess in sandbox)

        Args:
            tool: Tool name (must be in ALLOWED_SUBPROCESSES)
            *args: Arguments to pass to tool
            timeout: Timeout in seconds
            stdin_input: Optional input to pass via stdin

        Returns:
            Dict with stdout, stderr, returncode

        Raises:
            RLMSecurityError: If tool is not in allowlist
        """
        if tool not in ALLOWED_SUBPROCESSES:
            raise RLMSecurityError(
                f"Tool '{tool}' not allowed. Allowed: {sorted(ALLOWED_SUBPROCESSES)}"
            )

        try:
            result = subprocess.run(
                [tool, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin_input,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Tool '{tool}' timed out after {timeout}s",
                "returncode": -1,
                "success": False,
            }
        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": f"Tool '{tool}' not found",
                "returncode": -1,
                "success": False,
            }

    def enable_memory(self, store: MemoryStore) -> None:
        """
        Enable memory functions in the REPL environment.

        Implements: Spec SPEC-02.27-31

        Args:
            store: MemoryStore instance to use for persistence
        """
        self._memory_store = store

        # Add memory functions to globals
        self.globals["memory_query"] = self._memory_query
        self.globals["memory_add_fact"] = self._memory_add_fact
        self.globals["memory_add_experience"] = self._memory_add_experience
        self.globals["memory_get_context"] = self._memory_get_context
        self.globals["memory_relate"] = self._memory_relate

    def _memory_query(self, query: str, limit: int = 10) -> list[Any]:
        """
        Search for nodes matching a query.

        Implements: Spec SPEC-02.27

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching Node objects
        """
        if self._memory_store is None:
            return []

        # Query nodes that contain the search terms
        all_nodes = self._memory_store.query_nodes(limit=limit * 3)

        # Filter by query terms
        query_lower = query.lower()
        keywords = set(query_lower.split())

        matching = []
        for node in all_nodes:
            content_lower = node.content.lower()
            if any(kw in content_lower for kw in keywords):
                matching.append(node)

        # Sort by confidence descending and limit
        matching.sort(key=lambda n: n.confidence, reverse=True)
        return matching[:limit]

    def _memory_add_fact(self, content: str, confidence: float = 0.5) -> str:
        """
        Add a fact to memory.

        Implements: Spec SPEC-02.28

        Args:
            content: Fact content
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            Node ID of created fact
        """
        if self._memory_store is None:
            raise RuntimeError("Memory not enabled. Call enable_memory() first.")

        return self._memory_store.create_node(
            node_type="fact",
            content=content,
            confidence=confidence,
        )

    def _memory_add_experience(self, content: str, outcome: str, success: bool) -> str:
        """
        Add an experience to memory.

        Implements: Spec SPEC-02.29

        Args:
            content: Experience description
            outcome: Outcome description
            success: Whether the experience was successful

        Returns:
            Node ID of created experience
        """
        if self._memory_store is None:
            raise RuntimeError("Memory not enabled. Call enable_memory() first.")

        return self._memory_store.create_node(
            node_type="experience",
            content=content,
            metadata={"outcome": outcome, "success": success},
        )

    def _memory_get_context(self, limit: int = 10) -> list[Any]:
        """
        Get recent/relevant context nodes.

        Implements: Spec SPEC-02.30

        Args:
            limit: Maximum number of nodes to return

        Returns:
            List of Node objects sorted by confidence
        """
        if self._memory_store is None:
            return []

        # Get nodes sorted by confidence (high to low)
        nodes = self._memory_store.query_nodes(limit=limit * 2)

        # Sort by confidence descending
        nodes.sort(key=lambda n: n.confidence, reverse=True)
        return nodes[:limit]

    def _memory_relate(self, label: str, node_id1: str, node_id2: str) -> str:
        """
        Create a relation between two nodes.

        Implements: Spec SPEC-02.31

        Args:
            label: Relation label (e.g., "implies", "causes", "connects")
            node_id1: First node ID
            node_id2: Second node ID

        Returns:
            Edge ID of created relation
        """
        if self._memory_store is None:
            raise RuntimeError("Memory not enabled. Call enable_memory() first.")

        return self._memory_store.create_edge(
            edge_type="relation",
            label=label,
            members=[
                {"node_id": node_id1, "role": "subject", "position": 0},
                {"node_id": node_id2, "role": "object", "position": 1},
            ],
        )

    def get_context_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current context.

        Implements: Spec §3.1 Context Variable Schema

        Returns:
            Dict with context statistics
        """
        conv = self.globals.get("conversation", [])
        files = self.globals.get("files", {})
        tool_outputs = self.globals.get("tool_outputs", [])

        conv_chars = sum(len(m.get("content", "")) for m in conv)
        file_chars = sum(len(content) for content in files.values())
        output_chars = sum(len(o.get("content", "")) for o in tool_outputs)

        return {
            "conversation_count": len(conv),
            "conversation_tokens": conv_chars // 4,
            "file_count": len(files),
            "file_tokens": file_chars // 4,
            "tool_output_count": len(tool_outputs),
            "tool_output_tokens": output_chars // 4,
            "total_tokens": (conv_chars + file_chars + output_chars) // 4,
        }

    def inject_result(self, name: str, value: Any) -> None:
        """
        Inject a result back into the REPL namespace.

        Args:
            name: Variable name
            value: Value to inject
        """
        self.locals[name] = value

    def get_variable(self, name: str) -> Any:
        """
        Get a variable from the REPL namespace.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found
        """
        # Check locals first
        if name in self.locals:
            return self.locals[name]
        # Then globals
        if name in self.globals:
            return self.globals[name]
        # Check working_memory
        working_memory = self.globals.get("working_memory", {})
        if name in working_memory:
            return working_memory[name]
        raise KeyError(f"Variable '{name}' not found")

    def get_execution_history(self) -> list[dict[str, Any]]:
        """Get execution history for debugging."""
        return self.history.copy()


async def typecheck_snippet(code: str, timeout: float = 30.0) -> dict[str, Any]:
    """
    Run ty type checker on a code snippet.

    Implements: Spec §4.1.1 CLI Tools

    Args:
        code: Python code to type check
        timeout: Timeout in seconds

    Returns:
        TypeCheckResult with success status and errors
    """
    env = RLMEnvironment(SessionContext())
    result = env._run_tool("ty", "check", "-", timeout=timeout, stdin_input=code)

    return {
        "success": result["success"],
        "errors": result["stderr"] if not result["success"] else "",
        "output": result["stdout"],
    }


async def lint_snippet(code: str, timeout: float = 30.0) -> dict[str, Any]:
    """
    Run ruff linter on a code snippet.

    Implements: Spec §4.1.1 CLI Tools

    Args:
        code: Python code to lint
        timeout: Timeout in seconds

    Returns:
        LintResult with issues found
    """
    env = RLMEnvironment(SessionContext())
    result = env._run_tool(
        "ruff",
        "check",
        "--stdin-filename=snippet.py",
        "-",
        timeout=timeout,
        stdin_input=code,
    )

    return {
        "success": result["success"],
        "issues": result["stdout"],
        "errors": result["stderr"],
    }


__all__ = [
    "RLMEnvironment",
    "RLMSecurityError",
    "ALLOWED_SUBPROCESSES",
    "typecheck_snippet",
    "lint_snippet",
]
