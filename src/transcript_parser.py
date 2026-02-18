"""
Claude Code transcript parser for extracting full session context.

Reads the JSONL transcript files that Claude Code maintains at:
~/.claude/projects/{project_hash}/{session_id}.jsonl

This provides complete access to:
- User messages
- Assistant responses
- Tool calls and their inputs
- Tool results with full output

Implements: SPEC-XX (Transcript-based Context Capture)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    """A tool call from the transcript."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    result: str | None = None
    result_content: list[Any] | None = None


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: str | None = None


@dataclass
class TranscriptData:
    """Parsed transcript data."""

    session_id: str
    project_path: str | None = None
    git_branch: str | None = None
    turns: list[ConversationTurn] = field(default_factory=list)
    tool_results: dict[str, ToolCall] = field(default_factory=dict)  # tool_use_id -> ToolCall
    files_read: dict[str, str] = field(default_factory=dict)  # path -> content
    files_written: list[str] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class TranscriptParser:
    """
    Parse Claude Code transcript files.

    The transcript is a JSONL file where each line is a JSON object.
    Key entry types:
    - "user": User message or tool result
    - "assistant": Assistant response with possible tool calls
    - "progress": Hook and tool execution progress
    - "system": System messages
    """

    def __init__(self, transcript_path: str | Path):
        self.transcript_path = Path(transcript_path)
        self._entries: list[dict[str, Any]] = []
        self._parsed = False

    def _load_entries(self) -> None:
        """Load all entries from transcript file."""
        if self._parsed:
            return

        if not self.transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {self.transcript_path}")

        with open(self.transcript_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        self._parsed = True

    def parse(self) -> TranscriptData:
        """
        Parse the transcript and extract all relevant data.

        Returns:
            TranscriptData with conversation turns and tool information
        """
        self._load_entries()

        # Extract session info from first entry
        session_id = ""
        project_path = None
        git_branch = None

        for entry in self._entries:
            if "sessionId" in entry:
                session_id = entry.get("sessionId", "")
                project_path = entry.get("cwd")
                git_branch = entry.get("gitBranch")
                break

        data = TranscriptData(
            session_id=session_id,
            project_path=project_path,
            git_branch=git_branch,
        )

        # Track pending tool calls (tool_use_id -> ToolCall)
        pending_tool_calls: dict[str, ToolCall] = {}

        for entry in self._entries:
            entry_type = entry.get("type", "")

            if entry_type == "user":
                self._process_user_entry(entry, data, pending_tool_calls)

            elif entry_type == "assistant":
                self._process_assistant_entry(entry, data, pending_tool_calls)

        return data

    def _process_user_entry(
        self,
        entry: dict[str, Any],
        data: TranscriptData,
        pending_tool_calls: dict[str, ToolCall],
    ) -> None:
        """Process a user entry (message or tool result)."""
        message = entry.get("message", {})
        content = message.get("content", "")

        if isinstance(content, list):
            # Check for tool_result blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")

                    if block_type == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        result_content = block.get("content", "")

                        # Convert result content to string
                        if isinstance(result_content, list):
                            result_str = self._extract_text_from_blocks(result_content)
                        else:
                            result_str = str(result_content) if result_content else ""

                        # Update pending tool call with result
                        if tool_use_id in pending_tool_calls:
                            tool_call = pending_tool_calls[tool_use_id]
                            tool_call.result = result_str
                            tool_call.result_content = result_content if isinstance(result_content, list) else None

                            # Track specific tool results
                            self._track_tool_result(tool_call, data)

                            # Move to completed tool results
                            data.tool_results[tool_use_id] = tool_call
                            del pending_tool_calls[tool_use_id]

                    elif block_type == "text":
                        text_parts.append(block.get("text", ""))

                elif isinstance(block, str):
                    text_parts.append(block)

            if text_parts:
                data.turns.append(ConversationTurn(
                    role="user",
                    content="\n".join(text_parts),
                ))

        elif isinstance(content, str) and content.strip():
            data.turns.append(ConversationTurn(
                role="user",
                content=content,
            ))

    def _process_assistant_entry(
        self,
        entry: dict[str, Any],
        data: TranscriptData,
        pending_tool_calls: dict[str, ToolCall],
    ) -> None:
        """Process an assistant entry (response with possible tool calls)."""
        message = entry.get("message", {})
        content = message.get("content", "")

        if isinstance(content, list):
            text_parts = []
            tool_calls = []

            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")

                    if block_type == "text":
                        text_parts.append(block.get("text", ""))

                    elif block_type == "tool_use":
                        tool_call = ToolCall(
                            tool_use_id=block.get("id", ""),
                            tool_name=block.get("name", "unknown"),
                            tool_input=block.get("input", {}),
                        )
                        tool_calls.append(tool_call)
                        pending_tool_calls[tool_call.tool_use_id] = tool_call

                        # Track specific tool calls
                        self._track_tool_call(tool_call, data)

            if text_parts or tool_calls:
                data.turns.append(ConversationTurn(
                    role="assistant",
                    content="\n".join(text_parts),
                    tool_calls=tool_calls,
                ))

        elif isinstance(content, str) and content.strip():
            data.turns.append(ConversationTurn(
                role="assistant",
                content=content,
            ))

    def _extract_text_from_blocks(self, blocks: list[Any]) -> str:
        """Extract text from content blocks."""
        parts = []
        for block in blocks:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)

    def _track_tool_call(self, tool_call: ToolCall, data: TranscriptData) -> None:
        """Track specific tool calls for context extraction."""
        tool_name = tool_call.tool_name
        tool_input = tool_call.tool_input

        if tool_name == "Read":
            path = tool_input.get("file_path", "")
            if path:
                # Will be populated when we get the result
                pass

        elif tool_name == "Write":
            path = tool_input.get("file_path", "")
            if path:
                data.files_written.append(path)

        elif tool_name == "Edit":
            path = tool_input.get("file_path", "")
            if path:
                data.files_edited.append(path)

        elif tool_name == "Bash":
            command = tool_input.get("command", "")
            if command:
                data.commands_run.append(command)

    def _track_tool_result(self, tool_call: ToolCall, data: TranscriptData) -> None:
        """Track tool results for context extraction."""
        tool_name = tool_call.tool_name
        tool_input = tool_call.tool_input
        result = tool_call.result or ""

        if tool_name == "Read":
            path = tool_input.get("file_path", "")
            if path and result:
                data.files_read[path] = result

        elif tool_name == "Bash":
            # Check for errors in output
            if "error" in result.lower() or "failed" in result.lower():
                data.errors.append(f"Bash: {tool_input.get('command', '')[:100]}")

    def get_recent_messages(self, limit: int = 50) -> list[dict[str, str]]:
        """
        Get recent conversation messages.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of {"role": str, "content": str} dicts
        """
        data = self.parse()
        messages = []

        for turn in data.turns[-limit:]:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        return messages

    def get_all_tool_outputs(self) -> dict[str, str]:
        """
        Get all tool outputs indexed by tool_use_id.

        Returns:
            Dict of tool_use_id -> result content
        """
        data = self.parse()
        return {
            tool_use_id: tc.result
            for tool_use_id, tc in data.tool_results.items()
            if tc.result
        }

    def get_files_context(self) -> dict[str, str]:
        """
        Get all file contents that were read.

        Returns:
            Dict of file_path -> content
        """
        data = self.parse()
        return data.files_read


def find_transcript_path(session_id: str | None = None) -> Path | None:
    """
    Find the transcript file for a session.

    Args:
        session_id: Session ID to find (uses env var if not provided)

    Returns:
        Path to transcript file or None if not found
    """
    # Check environment variable first
    transcript_path = os.environ.get("CLAUDE_TRANSCRIPT_PATH")
    if transcript_path and Path(transcript_path).exists():
        return Path(transcript_path)

    # Try to find based on session ID
    session_id = session_id or os.environ.get("CLAUDE_SESSION_ID")
    cwd = os.environ.get("CLAUDE_CWD", os.getcwd())

    if session_id and cwd:
        claude_projects = Path.home() / ".claude" / "projects"

        # Claude uses sanitized path with dashes
        sanitized = cwd.replace("/", "-").replace(" ", "-")
        potential_path = claude_projects / sanitized / f"{session_id}.jsonl"

        if potential_path.exists():
            return potential_path

    return None


def load_session_context() -> dict[str, Any]:
    """
    Load full session context from transcript.

    This is the main entry point for getting all context data.

    Returns:
        Dict with:
        - messages: List of conversation messages
        - files: Dict of file_path -> content
        - tool_outputs: Dict of tool_use_id -> result
        - metadata: Session metadata
    """
    transcript_path = find_transcript_path()

    if not transcript_path:
        return {
            "messages": [],
            "files": {},
            "tool_outputs": {},
            "metadata": {},
            "error": "Transcript not found",
        }

    try:
        parser = TranscriptParser(transcript_path)
        data = parser.parse()

        return {
            "messages": [
                {"role": turn.role, "content": turn.content}
                for turn in data.turns
            ],
            "files": data.files_read,
            "tool_outputs": {
                tid: tc.result
                for tid, tc in data.tool_results.items()
                if tc.result
            },
            "metadata": {
                "session_id": data.session_id,
                "project_path": data.project_path,
                "git_branch": data.git_branch,
                "files_written": data.files_written,
                "files_edited": data.files_edited,
                "commands_run": data.commands_run,
                "errors": data.errors,
            },
        }

    except Exception as e:
        return {
            "messages": [],
            "files": {},
            "tool_outputs": {},
            "metadata": {},
            "error": str(e),
        }


__all__ = [
    "TranscriptParser",
    "TranscriptData",
    "ConversationTurn",
    "ToolCall",
    "find_transcript_path",
    "load_session_context",
]
