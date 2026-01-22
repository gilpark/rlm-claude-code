"""
Token-aware chunking for massive context handling.

Implements: SPEC-01.01 (Phase 4 - Massive Context)

Provides token-based chunking with semantic boundary detection,
replacing character-based partitioning for better LLM context handling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# Lazy import tiktoken to avoid startup cost
_tiktoken_encoding = None


def _get_encoding():
    """Get tiktoken encoding, lazily loaded."""
    global _tiktoken_encoding
    if _tiktoken_encoding is None:
        try:
            import tiktoken

            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _tiktoken_encoding = None
    return _tiktoken_encoding


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken cl100k_base encoding.

    Falls back to character-based estimation (4 chars/token) if tiktoken unavailable.

    Args:
        text: Text to count tokens for

    Returns:
        Token count
    """
    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))
    # Fallback: ~4 chars per token
    return len(text) // 4


def estimate_tokens(text: str) -> int:
    """Alias for count_tokens for backwards compatibility."""
    return count_tokens(text)


@dataclass
class Chunk:
    """A chunk of content with token metadata."""

    content: str
    token_count: int
    start_offset: int  # Character offset in original content
    end_offset: int
    boundary_type: str = "none"  # "function", "class", "paragraph", "line", "none"

    @property
    def char_count(self) -> int:
        """Character count of content."""
        return len(self.content)


@dataclass
class ChunkingConfig:
    """Configuration for token-aware chunking."""

    max_tokens: int = 4000
    overlap_tokens: int = 200
    prefer_semantic_boundaries: bool = True
    language: str | None = None  # "python", "javascript", "typescript", etc.

    # Semantic boundary patterns by language
    boundary_patterns: dict[str, list[re.Pattern]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default boundary patterns."""
        if not self.boundary_patterns:
            self.boundary_patterns = get_default_boundary_patterns()


def get_default_boundary_patterns() -> dict[str, list[re.Pattern]]:
    """
    Get default semantic boundary patterns for various languages.

    Returns:
        Dict mapping language to list of compiled regex patterns
    """
    return {
        "python": [
            re.compile(r"^class\s+\w+", re.MULTILINE),  # Class definitions
            re.compile(r"^def\s+\w+", re.MULTILINE),  # Function definitions
            re.compile(r"^async\s+def\s+\w+", re.MULTILINE),  # Async functions
            re.compile(r"^\s*@\w+", re.MULTILINE),  # Decorators (chunk before)
        ],
        "javascript": [
            re.compile(r"^(?:export\s+)?class\s+\w+", re.MULTILINE),
            re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+", re.MULTILINE),
            re.compile(r"^(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\(", re.MULTILINE),
            re.compile(r"^(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?function", re.MULTILINE),
        ],
        "typescript": [
            re.compile(r"^(?:export\s+)?class\s+\w+", re.MULTILINE),
            re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+\w+", re.MULTILINE),
            re.compile(r"^(?:export\s+)?const\s+\w+\s*=\s*(?:async\s+)?\(", re.MULTILINE),
            re.compile(r"^(?:export\s+)?interface\s+\w+", re.MULTILINE),
            re.compile(r"^(?:export\s+)?type\s+\w+", re.MULTILINE),
        ],
        "generic": [
            re.compile(r"^\n\n+", re.MULTILINE),  # Paragraph breaks
            re.compile(r"^#{1,6}\s+", re.MULTILINE),  # Markdown headers
            re.compile(r"^---+$", re.MULTILINE),  # Horizontal rules
        ],
    }


def detect_language(content: str) -> str | None:
    """
    Detect programming language from content.

    Args:
        content: Source code content

    Returns:
        Detected language or None
    """
    # Simple heuristics - could be enhanced with more sophisticated detection
    if re.search(r"^\s*def\s+\w+\s*\(|^\s*class\s+\w+.*:", content, re.MULTILINE):
        return "python"
    if re.search(
        r"^\s*(?:export\s+)?(?:function|const|let|var|interface|type)\s+", content, re.MULTILINE
    ):
        if re.search(r":\s*(?:string|number|boolean|any|\w+\[\]|Record<)", content):
            return "typescript"
        return "javascript"
    return None


def find_semantic_boundaries(content: str, language: str | None = None) -> list[int]:
    """
    Find semantic boundary positions in content.

    Args:
        content: Source code content
        language: Programming language (auto-detected if None)

    Returns:
        Sorted list of character positions where semantic boundaries occur
    """
    if language is None:
        language = detect_language(content)

    patterns = get_default_boundary_patterns()
    language_patterns = patterns.get(language or "generic", patterns["generic"])

    boundaries: set[int] = {0}  # Always include start

    for pattern in language_patterns:
        for match in pattern.finditer(content):
            # Find start of line containing match
            line_start = content.rfind("\n", 0, match.start()) + 1
            boundaries.add(line_start)

    return sorted(boundaries)


def _build_char_to_token_map(content: str) -> list[int]:
    """
    Build a mapping from character index to cumulative token count.

    This allows O(1) token counting for any substring once built.

    Returns:
        List where result[i] = number of tokens in content[:i]
    """
    enc = _get_encoding()
    if enc is None:
        # Fallback: estimate 4 chars per token
        return [i // 4 for i in range(len(content) + 1)]

    # Tokenize the entire content once
    tokens = enc.encode(content)

    # Build character-to-token cumulative count
    # We need to know which character index corresponds to which token
    cumulative = [0] * (len(content) + 1)

    char_idx = 0
    for token_idx, token in enumerate(tokens):
        # Decode this token to get its character length
        token_str = enc.decode([token])
        token_len = len(token_str)

        # All characters in this token map to token_idx + 1 cumulative tokens
        for _ in range(token_len):
            if char_idx < len(content):
                char_idx += 1
                cumulative[char_idx] = token_idx + 1

    # Fill any remaining (shouldn't happen, but safety)
    for i in range(char_idx + 1, len(content) + 1):
        cumulative[i] = len(tokens)

    return cumulative


def token_aware_chunk(
    content: str,
    max_tokens: int = 4000,
    overlap_tokens: int = 200,
    boundary_patterns: list[str] | None = None,
    language: str | None = None,
) -> list[tuple[str, int]]:
    """
    Chunk content based on token count with semantic boundary awareness.

    Primary chunking function that replaces character-based partitioning.

    Args:
        content: Content to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks for context continuity
        boundary_patterns: Optional list of regex patterns for boundaries
        language: Programming language for semantic detection

    Returns:
        List of (chunk_content, token_count) tuples
    """
    if not content:
        return [("", 0)]

    total_tokens = count_tokens(content)

    # If content fits in one chunk, return as-is
    if total_tokens <= max_tokens:
        return [(content, total_tokens)]

    # Find semantic boundaries (character positions)
    boundaries = find_semantic_boundaries(content, language)

    # Add custom boundary patterns if provided
    if boundary_patterns:
        for pattern_str in boundary_patterns:
            try:
                pattern = re.compile(pattern_str, re.MULTILINE)
                for match in pattern.finditer(content):
                    line_start = content.rfind("\n", 0, match.start()) + 1
                    boundaries.append(line_start)
            except re.error:
                pass  # Skip invalid patterns
        boundaries = sorted(set(boundaries))

    # Also add newline positions as potential boundaries
    newline_boundaries = set()
    for i, c in enumerate(content):
        if c == "\n":
            newline_boundaries.add(i + 1)  # Position after newline

    all_boundaries = sorted(set(boundaries) | newline_boundaries)

    # Build character-to-token cumulative map for O(1) lookups
    cumulative_tokens = _build_char_to_token_map(content)

    def tokens_in_range(start: int, end: int) -> int:
        """Get token count for content[start:end] in O(1)."""
        return (
            cumulative_tokens[min(end, len(content))] - cumulative_tokens[min(start, len(content))]
        )

    # Build chunks
    chunks: list[tuple[str, int]] = []
    current_start = 0

    while current_start < len(content):
        # Find the best end position that fits within max_tokens
        best_end = len(content)  # Default to end

        # Binary search through boundaries to find optimal end
        valid_ends = [b for b in all_boundaries if b > current_start]
        valid_ends.append(len(content))  # Always include end of content

        # Find largest boundary that keeps us within budget
        best_boundary_end = current_start
        for boundary in valid_ends:
            tok_count = tokens_in_range(current_start, boundary)
            if tok_count <= max_tokens:
                best_boundary_end = boundary
            else:
                break  # Boundaries are sorted, no point continuing

        if best_boundary_end > current_start:
            best_end = best_boundary_end
        else:
            # No boundary works, must cut mid-content
            # Find approximate character position for max_tokens
            target_tokens = cumulative_tokens[current_start] + max_tokens
            # Binary search for character position
            low, high = current_start, len(content)
            while high - low > 1:
                mid = (low + high) // 2
                if cumulative_tokens[mid] <= target_tokens:
                    low = mid
                else:
                    high = mid
            best_end = low

        # Ensure we make progress
        if best_end <= current_start:
            best_end = min(current_start + 100, len(content))

        chunk_content = content[current_start:best_end]
        chunk_tokens = tokens_in_range(current_start, best_end)

        if chunk_content.strip():  # Only add non-empty chunks
            chunks.append((chunk_content, chunk_tokens))

        # Calculate next start with overlap
        if best_end >= len(content):
            break

        # Find overlap start position
        if overlap_tokens > 0:
            # Find position that gives us overlap_tokens before best_end
            target_overlap_start_tokens = cumulative_tokens[best_end] - overlap_tokens
            target_overlap_start_tokens = max(
                target_overlap_start_tokens, cumulative_tokens[current_start]
            )

            # Binary search for overlap start
            low, high = current_start, best_end
            while high - low > 1:
                mid = (low + high) // 2
                if cumulative_tokens[mid] < target_overlap_start_tokens:
                    low = mid
                else:
                    high = mid

            # Try to align to a boundary
            overlap_start = high
            for boundary in all_boundaries:
                if overlap_start <= boundary < best_end:
                    overlap_start = boundary
                    break

            current_start = overlap_start
        else:
            current_start = best_end

    return chunks if chunks else [("", 0)]


def chunk_by_tokens(
    content: str,
    max_tokens: int = 4000,
    overlap_tokens: int = 0,
) -> list[Chunk]:
    """
    Chunk content with full metadata.

    Higher-level API returning Chunk objects with metadata.

    Args:
        content: Content to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks

    Returns:
        List of Chunk objects with metadata
    """
    raw_chunks = token_aware_chunk(
        content,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    chunks = []
    current_offset = 0

    for chunk_content, token_count in raw_chunks:
        # Find actual position in original content
        start = content.find(chunk_content, current_offset)
        if start == -1:
            start = current_offset
        end = start + len(chunk_content)

        # Detect boundary type
        boundary_type = _detect_boundary_type(chunk_content)

        chunks.append(
            Chunk(
                content=chunk_content,
                token_count=token_count,
                start_offset=start,
                end_offset=end,
                boundary_type=boundary_type,
            )
        )

        current_offset = start + 1  # Allow for overlap

    return chunks


def _detect_boundary_type(content: str) -> str:
    """Detect what type of boundary starts a chunk."""
    first_line = content.split("\n", 1)[0].strip()

    if re.match(r"^(async\s+)?def\s+\w+", first_line):
        return "function"
    if re.match(r"^class\s+\w+", first_line):
        return "class"
    if re.match(r"^(export\s+)?(async\s+)?function\s+\w+", first_line):
        return "function"
    if re.match(r"^(export\s+)?class\s+\w+", first_line):
        return "class"
    if re.match(r"^#{1,6}\s+", first_line):
        return "paragraph"
    if first_line == "" or re.match(r"^---+$", first_line):
        return "paragraph"

    return "line"


def iter_chunks(
    content: str,
    max_tokens: int = 4000,
    overlap_tokens: int = 200,
) -> Iterator[Chunk]:
    """
    Iterate over chunks lazily.

    Memory-efficient chunking for very large content.

    Args:
        content: Content to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks

    Yields:
        Chunk objects
    """
    # For now, this is not truly lazy since we need full content for boundaries
    # Future: could use streaming tokenization with tree-sitter
    yield from chunk_by_tokens(content, max_tokens, overlap_tokens)


def partition_content_by_tokens(content: str, n_chunks: int) -> list[str]:
    """
    Partition content into n roughly equal token-sized chunks.

    Drop-in replacement for character-based _partition_content().

    Args:
        content: Content to partition
        n_chunks: Target number of chunks

    Returns:
        List of content chunks
    """
    if not content:
        return [""]

    n_chunks = max(1, n_chunks)
    total_tokens = count_tokens(content)

    if total_tokens == 0:
        return [content]

    # For very small content, don't over-partition.
    # If we have fewer tokens than requested chunks, or if partitioning would
    # create chunks smaller than 2 tokens each, just return the content as-is.
    # This prevents content loss from whitespace filtering in token_aware_chunk.
    if total_tokens < n_chunks * 2:
        return [content]

    # Calculate target tokens per chunk using ceiling division
    # This ensures we don't create more than n_chunks
    tokens_per_chunk = -(-total_tokens // n_chunks)  # Ceiling division

    # Apply minimum chunk size only if it won't prevent creating n_chunks
    # This ensures we respect the n_chunks parameter for smaller content
    min_chunk_size = 100
    if total_tokens >= n_chunks * min_chunk_size:
        tokens_per_chunk = max(tokens_per_chunk, min_chunk_size)
    # No else needed - ceiling division already handles small content

    # Use token_aware_chunk with no overlap
    chunks = token_aware_chunk(
        content,
        max_tokens=tokens_per_chunk,
        overlap_tokens=0,
    )

    result = [chunk_content for chunk_content, _ in chunks]

    # Ensure we have at least one chunk
    return result if result else [""]
