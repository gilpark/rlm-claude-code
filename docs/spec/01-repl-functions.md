# SPEC-01: Advanced REPL Functions

## Overview

Add three new REPL helper functions from recurse: `map_reduce()`, `find_relevant()`, and `extract_functions()`.

## Requirements

### map_reduce()

[SPEC-01.01] The system SHALL provide a `map_reduce(content, map_prompt, reduce_prompt, n_chunks, model)` function.

[SPEC-01.02] `map_reduce()` SHALL partition content into `n_chunks` roughly equal parts.

[SPEC-01.03] `map_reduce()` SHALL apply `map_prompt` to each chunk in parallel using `llm_batch()`.

[SPEC-01.04] `map_reduce()` SHALL combine map results and apply `reduce_prompt` to synthesize final output.

[SPEC-01.05] `map_reduce()` SHALL support optional `model` parameter with values: "fast", "balanced", "powerful", "auto" (default).

[SPEC-01.06] `map_reduce()` SHALL handle content exceeding 1M characters without failure.

### find_relevant()

[SPEC-01.07] The system SHALL provide a `find_relevant(content, query, top_k, use_llm_scoring)` function.

[SPEC-01.08] `find_relevant()` SHALL partition content into ~50-line chunks with 5-line overlap.

[SPEC-01.09] `find_relevant()` SHALL perform keyword pre-filtering to identify candidate chunks.

[SPEC-01.10] `find_relevant()` SHALL optionally use LLM scoring when `use_llm_scoring=True` and candidates exceed `top_k * 2`.

[SPEC-01.11] `find_relevant()` SHALL return a list of `(chunk, score)` tuples sorted by relevance descending.

[SPEC-01.12] `find_relevant()` SHALL complete within 2 seconds for content under 100K characters (without LLM scoring).

### extract_functions()

[SPEC-01.13] The system SHALL provide an `extract_functions(content, language)` function.

[SPEC-01.14] `extract_functions()` SHALL support languages: "python", "go", "javascript", "typescript".

[SPEC-01.15] `extract_functions()` SHALL return a list of dicts with keys: "name", "signature", "start_line", "end_line".

[SPEC-01.16] `extract_functions()` SHALL use regex patterns appropriate for each language.

[SPEC-01.17] `extract_functions()` SHALL handle malformed input gracefully, returning partial results.

## Interface

```python
def map_reduce(
    content: str,
    map_prompt: str,
    reduce_prompt: str,
    n_chunks: int = 4,
    model: str = "auto"
) -> str:
    """Apply map-reduce pattern to large content."""

def find_relevant(
    content: str,
    query: str,
    top_k: int = 5,
    use_llm_scoring: bool = False
) -> list[tuple[str, float]]:
    """Find sections most relevant to query."""

def extract_functions(
    content: str,
    language: str = "python"
) -> list[dict]:
    """Extract function definitions from code."""
```

## Core REPL Functions

The REPL also provides these core functions (from Phase 4 RLAPH refactoring):

```python
def llm(query: str, context: str = "") -> str:
    """
    Recursive LLM call - returns actual result immediately.

    Note: As of Phase 4 (RLAPH refactoring), this returns a string
    synchronously, not a DeferredOperation. Use like:

        result = llm("What is 2+2?")
        print(result)  # Prints: "4"
    """

def peek(var: Any, start: int = 0, end: int = 1000) -> Any:
    """View a slice of any context variable."""

def search(var: Any, pattern: str, regex: bool = False) -> list:
    """Find patterns in context. Returns list of matches with location info."""

def summarize(var: Any, max_tokens: int = 500) -> str:
    """LLM-powered summarization via sub-call."""
```

## Security

[SPEC-01.18] All functions SHALL execute within the existing RestrictedPython sandbox.

[SPEC-01.19] `map_reduce()` and `find_relevant()` SHALL respect existing budget limits for LLM calls.

## Testing Requirements

[SPEC-01.20] Unit tests SHALL cover all functions with edge cases (empty content, single chunk, malformed input).

[SPEC-01.21] Property tests SHALL verify `map_reduce()` produces consistent results across different chunking strategies.

[SPEC-01.22] Integration tests SHALL verify functions work within REPL environment.
