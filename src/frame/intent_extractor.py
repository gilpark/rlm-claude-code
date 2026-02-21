"""Extract canonical task from user query - language/framework agnostic."""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Callable

from .canonical_task import CanonicalTask

if TYPE_CHECKING:
    from ..llm.llm_client import LLMClient


# -----------------------------------------------------------------------------
# Common intent verbs and their canonical task_type + scope
# Language-agnostic: works for Python, TypeScript, Go, Rust, etc.
# ORDER MATTERS: More specific patterns must come first
# -----------------------------------------------------------------------------
INTENT_VERBS: list[tuple[str, tuple[str, str]]] = [
    # Security / performance (must be checked first - they override other scopes)
    (r"\bsecurity\b", ("analyze", "security")),
    (r"\bvulnerability\b", ("analyze", "security")),
    (r"\bexploit\b", ("analyze", "security")),
    (r"\bperformance\b", ("analyze", "performance")),
    (r"\bslow\b", ("analyze", "performance")),
    (r"\boptimize\b", ("analyze", "performance")),
    (r"\bscale\b", ("analyze", "performance")),
    (r"\barchitecture\b", ("analyze", "architecture")),
    (r"\bdesign\b", ("analyze", "architecture")),
    (r"\bstructure\b", ("analyze", "architecture")),

    # Debugging / correctness
    (r"\bdebug\b", ("debug", "correctness")),
    (r"\bfix\b", ("debug", "correctness")),
    (r"\bsolve\b", ("debug", "correctness")),
    (r"\btroubleshoot\b", ("debug", "correctness")),
    (r"\berror\b", ("debug", "correctness")),
    (r"\bbug\b", ("debug", "correctness")),
    (r"\bissue\b", ("debug", "correctness")),
    (r"\bwrong\b", ("debug", "correctness")),
    (r"\bbroken\b", ("debug", "correctness")),
    (r"\bnot working\b", ("debug", "correctness")),

    # Verification / testing
    (r"\btest\b", ("verify", "correctness")),
    (r"\bverify\b", ("verify", "correctness")),
    (r"\bvalidate\b", ("verify", "correctness")),
    (r"\bcheck if\b", ("verify", "correctness")),
    (r"\bmake sure\b", ("verify", "correctness")),

    # Implementation / change
    (r"\bimplement\b", ("implement", "functionality")),
    (r"\badd\b", ("implement", "functionality")),
    (r"\bcreate\b", ("implement", "functionality")),
    (r"\bbuild\b", ("implement", "functionality")),
    (r"\bwrite\b", ("implement", "functionality")),
    (r"\bmake\b", ("implement", "functionality")),
    (r"\brefactor\b", ("refactor", "structure")),
    (r"\bimprove\b", ("refactor", "structure")),
    (r"\bclean up\b", ("refactor", "structure")),
    (r"\bdocument\b", ("document", "documentation")),
    (r"\bcomment\b", ("document", "documentation")),
    (r"\bdoc\b", ("document", "documentation")),
    (r"\breadme\b", ("document", "documentation")),

    # Analysis / understanding (most general - check last)
    (r"\banalyse\b", ("analyze", "overview")),
    (r"\banalyze\b", ("analyze", "overview")),
    (r"\breview\b", ("analyze", "overview")),
    (r"\bcheck\b", ("analyze", "overview")),
    (r"\binspect\b", ("analyze", "overview")),
    (r"\bexamine\b", ("analyze", "overview")),
    (r"\blook at\b", ("analyze", "overview")),
    (r"\bstudy\b", ("analyze", "overview")),
    (r"\bexplain\b", ("explain", "overview")),
    (r"\bdescribe\b", ("explain", "overview")),
    (r"\bhow does\b", ("explain", "overview")),
    (r"\bwhat is\b", ("explain", "overview")),
    (r"\btell me about\b", ("explain", "overview")),
    (r"\bsummarize\b", ("summarize", "overview")),
    (r"\bsummary\b", ("summarize", "overview")),
    (r"\boverview\b", ("summarize", "overview")),
    (r"\btl;?dr\b", ("summarize", "overview")),
]

# -----------------------------------------------------------------------------
# Target extraction patterns (file, dir, module, function, etc.)
# Extracts ACTUAL target from query, not hardcoded names
# ORDER MATTERS: More specific patterns must come first
# -----------------------------------------------------------------------------
TARGET_PATTERNS: list[tuple[str, Callable]] = [
    # Direct file mention with extension: "auth.py", "server.ts", "main.rs", "user.service.ts"
    # Must handle multi-dot names like "user.service.ts"
    (r"(?:^|\s)([a-zA-Z0-9_-][a-zA-Z0-9_.-]*\.(?:py|ts|js|jsx|tsx|go|rs|java|cpp|c|cs|rb|scala|kt))(?:\s|$|[,;.!?])",
     lambda m: [m.group(1)]),

    # Directory or module: "src/auth", "controllers/user", "lib/mdns"
    (r"\b(src|lib|app|controllers|services|utils|pkg|internal|cmd)/([a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*)\b",
     lambda m: [f"{m.group(1)}/{m.group(2)}"]),

    # Generic "the X file/module": "the auth file", "the user service"
    (r"\b(the|this|my|our)\s+([a-zA-Z0-9_-]+)\s+(file|module|service|controller|component|endpoints?|api)\b",
     lambda m: [f"**/*{m.group(2)}*"]),

    # Function/method/class: "login function", "UserService class"
    (r"\b([A-Za-z0-9_]+)\s+(function|method|func|class|struct|interface|handler)\b",
     lambda m: [f"**/*{m.group(1)}*"]),

    # "in X" pattern: "in auth", "in the api"
    (r"\b(in|inside)\s+(?:the\s+)?([a-zA-Z0-9_-]+)\b",
     lambda m: [f"**/*{m.group(2)}*"]),
]


def extract_canonical_task(
    query: str,
    llm_client: "LLMClient | None" = None,
    use_llm_fallback: bool = True,
) -> CanonicalTask:
    """
    Extract canonical task from user query.

    Uses fast deterministic rules first (70-80% hit rate), then LLM fallback.

    Design:
    - Intent-first: focuses on VERB + TARGET patterns
    - Language-agnostic: works for .py, .ts, .go, .rs, etc.
    - Framework-agnostic: no hardcoded module names

    Args:
        query: User query string
        llm_client: Optional LLM client for fallback
        use_llm_fallback: Whether to use LLM if rules don't match

    Returns:
        CanonicalTask with normalized intent
    """
    q_lower = query.lower()

    # Step 1: Determine task_type and scope from intent verbs
    task_type = "analyze"  # default fallback
    analysis_scope = "overview"

    for pattern, (tt, scope) in INTENT_VERBS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            task_type = tt
            analysis_scope = scope
            break

    # Step 2: Extract target dynamically
    target: str | list[str] | None = None

    for pattern, extractor in TARGET_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            target = extractor(match)
            break

    # Step 3: If we got both task and target -> confident rule-based result
    if target is not None:
        return CanonicalTask(
            task_type=task_type,
            target=target,
            analysis_scope=analysis_scope,
            params={"original_query": query[:200]},
        )

    # Step 4: LLM fallback for everything else
    if llm_client and use_llm_fallback:
        try:
            canonical = _extract_with_llm(query, llm_client)
            if canonical:
                return canonical
        except Exception:
            pass  # Fall through to ultimate fallback

    # Ultimate fallback: analyze whole codebase
    return CanonicalTask(
        task_type="analyze",
        target=["**/*"],
        analysis_scope="overview",
        params={"original_query": query[:200]},
    )


def _extract_with_llm(query: str, llm_client: "LLMClient") -> CanonicalTask | None:
    """Use LLM to extract canonical task (fallback for complex queries)."""
    prompt = f'''Extract canonical task from user query: "{query}"

Output JSON only:
{{
  "task_type": "analyze|debug|summarize|implement|refactor|document|verify|explain",
  "target": ["file pattern or list of files"],
  "analysis_scope": "overview|correctness|architecture|performance|security|documentation",
  "params": {{}}
}}

Be precise. Use glob patterns like "src/auth/*.ts" if appropriate. Target should be a list.'''

    result = llm_client.call(prompt)

    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', result, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group(1) if json_match.lastindex else json_match.group())
            # Ensure target is a list
            target = data.get("target", ["**/*"])
            if isinstance(target, str):
                target = [target]

            return CanonicalTask(
                task_type=data.get("task_type", "analyze"),
                target=target,
                analysis_scope=data.get("analysis_scope", "overview"),
                params=data.get("params", {}),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    return None
