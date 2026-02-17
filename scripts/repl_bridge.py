#!/usr/bin/env python3
"""
REPL Bridge for rlm-orchestrator agent.

This script provides a CLI interface to the real Python REPL implementation
in src/repl_environment.py. The rlm-orchestrator agent calls this via Bash
to execute actual REPL operations.

Implements: Spec §3.1 Context Variable Schema, §4.1 REPL Environment

Usage:
    python scripts/repl_bridge.py --op peek --args '{"var": "conversation", "start": 0, "end": 5}'
    python scripts/repl_bridge.py --op search --args '{"var": "files", "pattern": "def auth"}'
    python scripts/repl_bridge.py --op summarize --args '{"content": "...", "max_tokens": 500}'
    python scripts/repl_bridge.py --op llm --args '{"query": "Analyze this"}'
    python scripts/repl_bridge.py --op map_reduce --args '{"content": "...", "map_prompt": "...", "reduce_prompt": "..."}'
    python scripts/repl_bridge.py --op find_relevant --args '{"content": "...", "query": "..."}'

Context is loaded from ~/.claude/rlm-state/context.json (updated by sync_context hook).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_state_dir() -> Path:
    """Get RLM state directory."""
    state_dir = Path.home() / ".claude" / "rlm-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def load_config() -> dict[str, Any]:
    """Load RLM config from ~/.claude/rlm-config.json.

    Implements: Spec §5.3 Router Configuration
    """
    config_file = Path.home() / ".claude" / "rlm-config.json"
    if config_file.exists():
        try:
            return json.loads(config_file.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "depth": {"default": 2, "max": 3},
        "activation": {"mode": "complexity"},
    }


# Load config at module level
_config = load_config()
_max_depth = _config.get("depth", {}).get("max", 3)
_depth = 0


def load_context() -> dict[str, Any]:
    """Load context from state file."""
    ctx_file = get_state_dir() / "context.json"
    if ctx_file.exists():
        try:
            return json.loads(ctx_file.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "conversation": [],
        "files": {},
        "tool_outputs": [],
        "working_memory": {},
    }


def save_context(context: dict[str, Any]) -> None:
    """Save context to state file."""
    ctx_file = get_state_dir() / "context.json"
    ctx_file.write_text(json.dumps(context, indent=2, default=str))


def create_repl_environment():
    """Create RLM environment with loaded context.

    Implements: Spec §4.1 Sandbox Architecture
    """
    try:
        from src.repl_environment import RLMEnvironment
        from src.types import SessionContext

        context_data = load_context()
        session_ctx = SessionContext(
            messages=context_data.get("conversation", []),
            files=context_data.get("files", {}),
            tool_outputs=context_data.get("tool_outputs", []),
        )
        return RLMEnvironment(session_ctx)
    except ImportError as e:
        return None, str(e)


def op_peek(args: dict[str, Any]) -> dict[str, Any]:
    """Execute peek operation.

    Implements: Spec §3.1 peek() helper function
    """
    var_name = args.get("var", "conversation")
    start = args.get("start", 0)
    end = args.get("end", -1)

    context = load_context()

    # Get the variable
    if var_name not in context:
        return {"error": f"Variable '{var_name}' not found. Available: {list(context.keys())}"}

    var = context[var_name]

    # Try to use real RLMEnvironment._peek if available
    try:
        from src.repl_environment import RLMEnvironment
        from src.types import SessionContext

        session_ctx = SessionContext(
            messages=context.get("conversation", []),
            files=context.get("files", {}),
            tool_outputs=context.get("tool_outputs", []),
        )
        repl = RLMEnvironment(session_ctx)
        result = repl._peek(var, start, end)
        return {"result": result, "type": type(result).__name__, "source": "rlm_environment"}
    except (ImportError, AttributeError):
        pass

    # Fallback implementation
    if isinstance(var, list):
        if end == -1:
            end = len(var)
        result = var[start:end]
    elif isinstance(var, str):
        if end == -1:
            end = len(var)
        result = var[start:end]
    elif isinstance(var, dict):
        keys = list(var.keys())[start:end] if end != -1 else list(var.keys())[start:]
        result = {k: var[k] for k in keys}
    else:
        result = str(var)

    return {"result": result, "type": type(result).__name__}


def op_search(args: dict[str, Any]) -> dict[str, Any]:
    """Execute search operation."""
    import re

    var_name = args.get("var", "files")
    pattern = args.get("pattern", "")
    use_regex = args.get("regex", False)

    context = load_context()

    if var_name not in context:
        return {"error": f"Variable '{var_name}' not found"}

    var = context[var_name]
    matches = []

    if isinstance(var, dict):
        # Search in dict values (e.g., files)
        for key, value in var.items():
            if isinstance(value, str):
                if use_regex:
                    if re.search(pattern, value):
                        matches.append({"file": key, "matches": re.findall(pattern, value)})
                else:
                    if pattern in value:
                        matches.append({"file": key, "count": value.count(pattern)})
    elif isinstance(var, list):
        # Search in list items
        for i, item in enumerate(var):
            item_str = json.dumps(item) if not isinstance(item, str) else item
            if use_regex:
                if re.search(pattern, item_str):
                    matches.append({"index": i, "match": re.findall(pattern, item_str)})
            else:
                if pattern in item_str:
                    matches.append({"index": i})
    elif isinstance(var, str):
        if use_regex:
            matches = re.findall(pattern, var)
        else:
            # Find all positions
            pos = 0
            while True:
                pos = var.find(pattern, pos)
                if pos == -1:
                    break
                matches.append({"position": pos, "context": var[max(0, pos-20):pos+len(pattern)+20]})
                pos += 1

    return {"matches": matches, "count": len(matches)}


def op_summarize(args: dict[str, Any]) -> dict[str, Any]:
    """Execute summarize operation using LLM."""
    try:
        from src.repl_environment import RLMEnvironment
        from src.types import SessionContext

        content = args.get("content", "")
        max_tokens = args.get("max_tokens", 500)

        context = load_context()
        session_ctx = SessionContext(
            messages=context.get("conversation", []),
            files=context.get("files", {}),
            tool_outputs=context.get("tool_outputs", []),
        )
        repl = RLMEnvironment(session_ctx)

        # Use the real summarize function
        result = repl.summarize(content, max_tokens=max_tokens)
        return {"summary": result}

    except ImportError as e:
        # Fallback: simple truncation
        content = args.get("content", "")
        max_tokens = args.get("max_tokens", 500)
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        if len(content) > max_chars:
            return {"summary": content[:max_chars] + "...", "truncated": True}
        return {"summary": content, "truncated": False}
    except Exception as e:
        return {"error": str(e)}


def op_llm(args: dict[str, Any]) -> dict[str, Any]:
    """Execute recursive LLM query.

    Implements: Spec §3.1 llm() helper function, §4.1 Deferred operations
    """
    try:
        from src.repl_environment import RLMEnvironment
        from src.types import SessionContext

        query = args.get("query", "")
        context_content = args.get("context")

        context = load_context()
        session_ctx = SessionContext(
            messages=context.get("conversation", []),
            files=context.get("files", {}),
            tool_outputs=context.get("tool_outputs", []),
        )
        repl = RLMEnvironment(session_ctx)

        # _recursive_query returns a DeferredOperation
        deferred = repl._recursive_query(query, context=context_content)

        return {
            "status": "deferred",
            "operation_id": deferred.operation_id,
            "operation": "llm",
            "query": query[:100] + "..." if len(query) > 100 else query,
            "note": "LLM query requires execution. Use orchestrator to resolve.",
        }

    except ImportError as e:
        return {"error": f"LLM function not available: {e}"}
    except Exception as e:
        return {"error": str(e)}


def op_context(args: dict[str, Any]) -> dict[str, Any]:
    """Get or update context."""
    action = args.get("action", "get")

    if action == "get":
        context = load_context()
        return {"context": context}
    elif action == "set":
        updates = args.get("updates", {})
        context = load_context()
        context.update(updates)
        save_context(context)
        return {"status": "updated", "keys": list(updates.keys())}
    elif action == "keys":
        context = load_context()
        return {"keys": list(context.keys())}
    else:
        return {"error": f"Unknown action: {action}"}


def op_map_reduce(args: dict[str, Any]) -> dict[str, Any]:
    """Execute map-reduce operation over content.

    Implements: Spec §3.1, §4.1 - Partition and aggregate large contexts
    """
    try:
        from src.repl_environment import RLMEnvironment
        from src.types import SessionContext

        content = args.get("content", "")
        map_prompt = args.get("map_prompt", "Summarize this section")
        reduce_prompt = args.get("reduce_prompt", "Combine these summaries")
        n_chunks = args.get("n_chunks", 4)

        context = load_context()
        session_ctx = SessionContext(
            messages=context.get("conversation", []),
            files=context.get("files", {}),
            tool_outputs=context.get("tool_outputs", []),
        )
        repl = RLMEnvironment(session_ctx)

        # _map_reduce returns a DeferredBatch
        batch = repl._map_reduce(content, map_prompt, reduce_prompt, n_chunks)

        return {
            "status": "deferred",
            "batch_id": batch.batch_id,
            "operation": "map_reduce",
            "n_chunks": n_chunks,
            "n_operations": len(batch.operations),
            "note": "Map-reduce requires LLM execution. Use orchestrator to resolve.",
        }

    except ImportError as e:
        return {"error": f"RLM environment not available: {e}"}
    except Exception as e:
        return {"error": str(e)}


def op_find_relevant(args: dict[str, Any]) -> dict[str, Any]:
    """Find relevant sections in content.

    Implements: Spec §3.1, §4.1 - Semantic search over context
    """
    try:
        from src.repl_environment import RLMEnvironment
        from src.types import SessionContext

        content = args.get("content", "")
        query = args.get("query", "")
        top_k = args.get("top_k", 5)

        context = load_context()
        session_ctx = SessionContext(
            messages=context.get("conversation", []),
            files=context.get("files", {}),
            tool_outputs=context.get("tool_outputs", []),
        )
        repl = RLMEnvironment(session_ctx)

        # _find_relevant returns list of (chunk, score) tuples directly
        results = repl._find_relevant(content, query, top_k)

        return {
            "status": "complete",
            "results": [
                {"chunk": chunk, "score": score}
                for chunk, score in results
            ],
            "count": len(results),
            "operation": "find_relevant",
            "top_k": top_k,
        }

    except ImportError as e:
        return {"error": f"RLM environment not available: {e}"}
    except Exception as e:
        return {"error": str(e)}


def op_extract_functions(args: dict[str, Any]) -> dict[str, Any]:
    """Extract function definitions from code.

    Implements: Spec §4.1 - Parse code structure
    """
    try:
        from src.repl_environment import RLMEnvironment
        from src.types import SessionContext

        content = args.get("content", "")
        language = args.get("language", "python")

        context = load_context()
        session_ctx = SessionContext(
            messages=context.get("conversation", []),
            files=context.get("files", {}),
            tool_outputs=context.get("tool_outputs", []),
        )
        repl = RLMEnvironment(session_ctx)

        # _extract_functions returns list of function info
        functions = repl._extract_functions(content, language)

        return {
            "functions": functions,
            "count": len(functions),
            "language": language,
        }

    except ImportError as e:
        return {"error": f"RLM environment not available: {e}"}
    except Exception as e:
        return {"error": str(e)}


def op_conversation(args: dict[str, Any]) -> dict[str, Any]:
    """Load conversation from transcript file.

    Implements: Access to session transcript for rlm-orchestrator agent.
    Reads from the transcript_path captured by capture_session_context.sh.
    """
    state_dir = get_state_dir()
    metadata_file = state_dir / "session-metadata.json"

    if not metadata_file.exists():
        return {"error": "No session metadata. Start a new session."}

    metadata = json.loads(metadata_file.read_text())
    transcript_path = metadata.get("transcript_path")

    if not transcript_path:
        return {"error": "No transcript_path in session metadata"}

    transcript_file = Path(transcript_path)
    if not transcript_file.exists():
        return {"error": f"Transcript file not found: {transcript_path}"}

    # Read JSONL transcript
    messages = []
    with open(transcript_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "message":
                    messages.append({
                        "role": entry.get("role", "unknown"),
                        "content": entry.get("content", ""),
                    })
            except json.JSONDecodeError:
                continue

    # Pagination
    start = args.get("start", 0)
    end = args.get("end", -1)
    total = len(messages)
    if end == -1 or end > total:
        end = total

    return {
        "messages": messages[start:end],
        "total": total,
        "range": [start, end],
        "session_id": metadata.get("session_id"),
        "transcript_path": transcript_path,
    }


def op_stats(args: dict[str, Any]) -> dict[str, Any]:
    """Get context statistics.

    Implements: Spec §3.1 context_stats
    """
    context = load_context()

    # Calculate token estimates
    conversation_tokens = sum(
        len(str(m)) // 4 for m in context.get("conversation", [])
    )
    file_tokens = sum(
        len(str(v)) // 4 for v in context.get("files", {}).values()
    )
    tool_output_tokens = sum(
        len(str(o)) // 4 for o in context.get("tool_outputs", [])
    )
    total_tokens = conversation_tokens + file_tokens + tool_output_tokens

    return {
        "total_tokens": total_tokens,
        "conversation_tokens": conversation_tokens,
        "file_tokens": file_tokens,
        "tool_output_tokens": tool_output_tokens,
        "n_messages": len(context.get("conversation", [])),
        "n_files": len(context.get("files", {})),
        "n_tool_outputs": len(context.get("tool_outputs", [])),
        "max_depth": _max_depth,
        "config": _config.get("activation", {}),
    }


def op_memory_query(args: dict[str, Any]) -> dict[str, Any]:
    """Query memory store.

    Implements: Spec §4.1 memory_* functions, uses rlm-memory.db
    """
    try:
        from src.memory_store import MemoryStore

        query = args.get("query", "")
        limit = args.get("limit", 10)

        # Memory store at ~/.claude/rlm-memory.db
        db_path = Path.home() / ".claude" / "rlm-memory.db"
        store = MemoryStore(db_path=str(db_path))

        # Search nodes
        results = store.query_nodes(query, limit=limit)

        return {
            "results": results,
            "count": len(results),
            "query": query,
        }

    except ImportError as e:
        return {"error": f"Memory store not available: {e}"}
    except Exception as e:
        return {"error": str(e)}


def op_memory_add(args: dict[str, Any]) -> dict[str, Any]:
    """Add fact or experience to memory.

    Implements: Spec §4.1 memory_add_fact, memory_add_experience
    """
    try:
        from src.memory_store import MemoryStore

        node_type = args.get("type", "fact")  # fact, experience, procedure, goal
        content = args.get("content", "")
        confidence = args.get("confidence", 0.9)

        db_path = Path.home() / ".claude" / "rlm-memory.db"
        store = MemoryStore(db_path=str(db_path))

        node_id = store.create_node(
            node_type=node_type,
            content=content,
            confidence=confidence,
        )

        return {
            "status": "created",
            "node_id": node_id,
            "type": node_type,
        }

    except ImportError as e:
        return {"error": f"Memory store not available: {e}"}
    except Exception as e:
        return {"error": str(e)}


OPERATIONS = {
    # Core operations (Spec §3.1)
    "peek": op_peek,
    "search": op_search,
    "summarize": op_summarize,
    "llm": op_llm,
    # Extended operations (Spec §4.1)
    "map_reduce": op_map_reduce,
    "find_relevant": op_find_relevant,
    "extract_functions": op_extract_functions,
    # Session access
    "conversation": op_conversation,
    # Memory operations (uses rlm-memory.db)
    "memory_query": op_memory_query,
    "memory_add": op_memory_add,
    # Context management
    "context": op_context,
    "stats": op_stats,
}


def main():
    parser = argparse.ArgumentParser(
        description="REPL Bridge for rlm-orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Operations (Spec §3.1, §4.1):
  Core:
    peek              View slice of context variable
    search            Find patterns in context
    summarize         LLM-based summarization (returns deferred)
    llm               Recursive LLM query (returns deferred)

  Extended:
    map_reduce        Partition and aggregate large content
    find_relevant     Semantic search over content
    extract_functions Parse function definitions from code

  Session:
    conversation      Load conversation from session transcript

  Memory (rlm-memory.db):
    memory_query      Search stored facts/experiences
    memory_add        Store new fact or experience

  Context:
    context           Get/set context state
    stats             Get token counts and config

Examples:
  Peek at conversation:
    python scripts/repl_bridge.py --op peek --args '{"var": "conversation", "start": 0, "end": 5}'

  Load session transcript:
    python scripts/repl_bridge.py --op conversation --args '{"start": 0, "end": 5}'

  Search files:
    python scripts/repl_bridge.py --op search --args '{"var": "files", "pattern": "def auth"}'

  Query memory:
    python scripts/repl_bridge.py --op memory_query --args '{"query": "authentication"}'

  Get stats:
    python scripts/repl_bridge.py --op stats --args '{}'
        """,
    )
    parser.add_argument(
        "--op",
        required=True,
        choices=list(OPERATIONS.keys()),
        help="Operation to execute",
    )
    parser.add_argument(
        "--args",
        default="{}",
        help="JSON arguments for the operation",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    try:
        op_args = json.loads(args.args)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON args: {e}"}))
        sys.exit(1)

    op_func = OPERATIONS.get(args.op)
    if not op_func:
        print(json.dumps({"error": f"Unknown operation: {args.op}"}))
        sys.exit(1)

    try:
        result = op_func(op_args)
        if args.pretty:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(json.dumps(result, default=str))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
