"""Tests for causal_router - CausalFrame command router."""

from unittest.mock import patch

import pytest

from src.skills.causal_router import (
    parse_flags,
    generate_help_text,
    COMMANDS,
    get_session_id,
    cmd_status,
    cmd_tree,
    cmd_resume,
    cmd_clear_cache,
)


class TestParseFlags:
    """Test suite for parse_flags function."""

    def test_parse_empty_string(self):
        """Parse empty string returns empty positional list."""
        result = parse_flags("")
        assert result == {"_positional": []}

    def test_parse_none_returns_empty(self):
        """Parse None returns empty positional list."""
        result = parse_flags(None)
        assert result == {"_positional": []}

    def test_parse_positional_only(self):
        """Parse string with only positional arguments."""
        result = parse_flags("analyze src/auth.py")
        assert result["_positional"] == ["analyze", "src/auth.py"]
        assert len([k for k in result if k != "_positional"]) == 0

    def test_parse_single_flag_with_value(self):
        """Parse string with --flag value syntax."""
        result = parse_flags("analyze src/auth.py --scope security")
        assert result["_positional"] == ["analyze", "src/auth.py"]
        assert result["scope"] == "security"

    def test_parse_boolean_flag(self):
        """Parse string with boolean flag (--flag with no value)."""
        result = parse_flags("analyze --verbose")
        assert result["_positional"] == ["analyze"]
        assert result["verbose"] is True

    def test_parse_multiple_flags(self):
        """Parse string with multiple flags."""
        result = parse_flags("analyze src/auth.py --scope security --depth 5 --verbose")
        assert result["_positional"] == ["analyze", "src/auth.py"]
        assert result["scope"] == "security"
        assert result["depth"] == "5"
        assert result["verbose"] is True

    def test_parse_flag_value_is_another_flag(self):
        """Parse when next token starts with -- (treat as boolean flag)."""
        result = parse_flags("analyze --verbose --scope security")
        assert result["_positional"] == ["analyze"]
        assert result["verbose"] is True
        assert result["scope"] == "security"

    def test_parse_quoted_strings(self):
        """Parse string with quoted arguments using shlex."""
        result = parse_flags('analyze "some file.py" --scope "security analysis"')
        assert result["_positional"] == ["analyze", "some file.py"]
        assert result["scope"] == "security analysis"

    def test_parse_numeric_values_as_strings(self):
        """Parse numeric flag values as strings."""
        result = parse_flags("analyze --depth 10 --threshold 3.14")
        assert result["depth"] == "10"
        assert result["threshold"] == "3.14"

    def test_parse_mixed_order(self):
        """Parse flags in various positions relative to positional args."""
        result = parse_flags("--scope security analyze src/auth.py --verbose")
        assert result["_positional"] == ["analyze", "src/auth.py"]
        assert result["scope"] == "security"
        assert result["verbose"] is True


class TestGenerateHelpText:
    """Test suite for generate_help_text function."""

    def test_help_text_contains_command_table(self):
        """Help text includes command table with headers."""
        help_text = generate_help_text()
        assert "## /causal Commands" in help_text
        assert "| Command | Description | Example |" in help_text
        assert "|---------|-------------|----------|" in help_text

    def test_help_text_includes_all_commands(self):
        """Help text includes entries for all registered commands."""
        help_text = generate_help_text()
        for cmd in COMMANDS:
            assert f"| {cmd} |" in help_text

    def test_help_text_includes_descriptions(self):
        """Help text includes command descriptions."""
        help_text = generate_help_text()
        assert "Run detailed analysis on target" in help_text
        assert "Quick summary of target" in help_text
        assert "Debug issues in target" in help_text
        assert "Show valid/invalidated frames" in help_text

    def test_help_text_includes_examples(self):
        """Help text includes command examples."""
        help_text = generate_help_text()
        assert "/causal analyze" in help_text
        assert "/causal summarize" in help_text
        assert "/causal debug" in help_text
        assert "/causal status" in help_text
        assert "/causal resume" in help_text
        assert "/causal tree" in help_text
        assert "/causal clear-cache" in help_text
        assert "/causal help" in help_text

    def test_help_text_contains_flags_section(self):
        """Help text includes flags documentation."""
        help_text = generate_help_text()
        assert "### Flags" in help_text
        assert "| Flag | Effect |" in help_text
        assert "|------|--------|" in help_text

    def test_help_text_lists_common_flags(self):
        """Help text documents common flags."""
        help_text = generate_help_text()
        assert "--verbose" in help_text
        assert "--depth" in help_text
        assert "--scope" in help_text
        assert "--last" in help_text
        assert "--session" in help_text

    def test_help_text_describes_flag_effects(self):
        """Help text describes what each flag does."""
        help_text = generate_help_text()
        assert "recursion logs" in help_text or "recursion" in help_text.lower()
        assert "depth" in help_text.lower()
        assert "scope" in help_text.lower() or "analysis" in help_text.lower()
        assert "session" in help_text.lower()


class TestCommandsRegistry:
    """Test suite for COMMANDS registry."""

    def test_all_commands_have_required_fields(self):
        """All commands in registry have description, example, and agent."""
        for cmd, info in COMMANDS.items():
            assert "description" in info
            assert "example" in info
            assert "agent" in info

    def test_special_commands_have_none_agent(self):
        """Special handler commands have agent=None."""
        special_commands = ["status", "resume", "tree", "clear-cache", "help"]
        for cmd in special_commands:
            assert COMMANDS[cmd]["agent"] is None

    def test_agent_commands_have_agent_set(self):
        """Agent-based commands have agent configured."""
        agent_commands = ["analyze", "summarize", "debug"]
        for cmd in agent_commands:
            assert COMMANDS[cmd]["agent"] is not None

    def test_command_names_are_lowercase(self):
        """All command names are lowercase for consistency."""
        for cmd in COMMANDS:
            assert cmd == cmd.lower()


class TestCmdTree:
    """Test suite for cmd_tree handler (Task 66)."""

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_tree_no_frames(self, mock_find_session, mock_index_load):
        """cmd_tree returns message when no frames found."""
        from src.frame.frame_index import FrameIndex

        mock_find_session.return_value = "test_session"
        mock_index_load.return_value = FrameIndex()  # Empty index

        result = await cmd_tree(args={})

        assert "No frames found" in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_tree_single_frame(self, mock_find_session, mock_index_load):
        """cmd_tree displays single root frame."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        mock_find_session.return_value = "test_session"

        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        frame = CausalFrame(
            frame_id="abc123def456",
            depth=0,
            parent_id=None,
            children=[],
            query="Root query",
            context_slice=context,
            evidence=[],
            conclusion="Root conclusion",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        result = await cmd_tree(args={})

        assert "## Frame Tree" in result
        assert "test_session" in result
        assert "✓" in result  # Completed icon
        assert "abc123de" in result  # Truncated frame_id
        assert "depth=0" in result
        assert "Root query" in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_tree_with_children(self, mock_find_session, mock_index_load):
        """cmd_tree displays parent-child relationships."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        mock_find_session.return_value = "test_session"

        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        parent = CausalFrame(
            frame_id="parent12345",  # Need at least 8 chars
            depth=0,
            parent_id=None,
            children=["child45678"],
            query="Parent query",
            context_slice=context,
            evidence=[],
            conclusion="Parent conclusion",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        child = CausalFrame(
            frame_id="child45678",  # Need at least 8 chars
            depth=1,
            parent_id="parent12345",
            children=[],
            query="Child query",
            context_slice=context,
            evidence=[],
            conclusion="Child conclusion",
            confidence=0.85,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(parent)
        index.add(child)
        mock_index_load.return_value = index

        result = await cmd_tree(args={})

        # Frame IDs are truncated to 8 chars in output
        assert "parent12" in result
        assert "child456" in result
        assert "└──" in result  # Tree connector
        assert "depth=0" in result
        assert "depth=1" in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_tree_status_icons(self, mock_find_session, mock_index_load):
        """cmd_tree displays correct status icons."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        mock_find_session.return_value = "test_session"

        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        # Create frames with different statuses
        completed_frame = CausalFrame(
            frame_id="completed",
            depth=0,
            parent_id=None,
            children=[],
            query="Completed",
            context_slice=context,
            evidence=[],
            conclusion="Done",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        invalidated_frame = CausalFrame(
            frame_id="invalidated",
            depth=0,
            parent_id=None,
            children=[],
            query="Invalidated",
            context_slice=context,
            evidence=[],
            conclusion="Old",
            confidence=0.7,
            invalidation_condition={},
            status=FrameStatus.INVALIDATED,
            created_at=datetime.now(),
        )

        suspended_frame = CausalFrame(
            frame_id="suspended",
            depth=0,
            parent_id=None,
            children=[],
            query="Suspended",
            context_slice=context,
            evidence=[],
            conclusion=None,
            confidence=0.0,
            invalidation_condition={},
            status=FrameStatus.SUSPENDED,
            created_at=datetime.now(),
        )

        running_frame = CausalFrame(
            frame_id="running",
            depth=0,
            parent_id=None,
            children=[],
            query="Running",
            context_slice=context,
            evidence=[],
            conclusion=None,
            confidence=0.0,
            invalidation_condition={},
            status=FrameStatus.RUNNING,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        for f in [completed_frame, invalidated_frame, suspended_frame, running_frame]:
            index.add(f)
        mock_index_load.return_value = index

        result = await cmd_tree(args={})

        assert "✓" in result  # COMPLETED
        assert "✗" in result  # INVALIDATED
        assert "⏸" in result  # SUSPENDED
        assert "🔄" in result  # RUNNING

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_tree_truncates_long_queries(self, mock_find_session, mock_index_load):
        """cmd_tree truncates long query text."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        mock_find_session.return_value = "test_session"

        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        long_query = "This is a very long query that should be truncated" * 3

        frame = CausalFrame(
            frame_id="long123",
            depth=0,
            parent_id=None,
            children=[],
            query=long_query,
            context_slice=context,
            evidence=[],
            conclusion="Conclusion",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        result = await cmd_tree(args={})

        # Query should be truncated with "..."
        assert "..." in result
        # But not the full query
        assert long_query not in result


class TestCmdResume:
    """Test suite for cmd_resume handler (Task 67)."""

    @patch("src.skills.causal_router.analyzer_agent.run")
    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_resume_no_frames(self, mock_find_session, mock_index_load, mock_agent_run):
        """cmd_resume returns message when no frames found."""
        from src.frame.frame_index import FrameIndex

        mock_find_session.return_value = "test_session"
        mock_index_load.return_value = None

        result = await cmd_resume(frame_id="abc123", args={})

        assert "No frames found" in result
        mock_agent_run.assert_not_called()

    @patch("src.skills.causal_router.analyzer_agent.run")
    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_resume_empty_index(self, mock_find_session, mock_index_load, mock_agent_run):
        """cmd_resume returns message when index is empty."""
        from src.frame.frame_index import FrameIndex

        mock_find_session.return_value = "test_session"
        mock_index_load.return_value = FrameIndex()

        result = await cmd_resume(frame_id="nonexistent", args={})

        assert "No frames found" in result
        mock_agent_run.assert_not_called()

    @patch("src.skills.causal_router.analyzer_agent.run")
    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_resume_frame_not_found_in_index(self, mock_find_session, mock_index_load, mock_agent_run):
        """cmd_resume returns message when frame ID not found in index."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        mock_find_session.return_value = "test_session"

        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        # Add a different frame to the index
        frame = CausalFrame(
            frame_id="existing123",
            depth=0,
            parent_id=None,
            children=[],
            query="Existing frame",
            context_slice=context,
            evidence=[],
            conclusion="Result",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        result = await cmd_resume(frame_id="nonexistent", args={})

        assert "not found" in result
        mock_agent_run.assert_not_called()

    @patch("src.skills.causal_router.analyzer_agent.run")
    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_resume_auto_detect_invalidated(self, mock_find_session, mock_index_load, mock_agent_run):
        """cmd_resume auto-detects most recent invalidated frame."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex
        from src.repl.rlaph_loop import RLPALoopResult

        mock_find_session.return_value = "test_session"

        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        invalidated_frame = CausalFrame(
            frame_id="invalidated123",
            depth=0,
            parent_id=None,
            children=[],
            query="Old analysis",
            context_slice=context,
            evidence=[],
            conclusion="Old result",
            confidence=0.7,
            invalidation_condition={"description": "File changed"},
            status=FrameStatus.INVALIDATED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(invalidated_frame)
        mock_index_load.return_value = index

        mock_result = RLPALoopResult(
            answer="New analysis result",
            iterations=1,
            depth_used=1,
            tokens_used=100,
            execution_time_ms=100.0,
            history=[],
        )
        mock_agent_run.return_value = mock_result

        result = await cmd_resume(frame_id=None, args={})

        assert "Resumed Frame" in result
        assert "invalida" in result  # Truncated frame_id
        mock_agent_run.assert_called_once()

    @patch("src.skills.causal_router.analyzer_agent.run")
    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_resume_with_frame_id(self, mock_find_session, mock_index_load, mock_agent_run):
        """cmd_resume resumes specific frame by ID."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex
        from src.repl.rlaph_loop import RLPALoopResult

        mock_find_session.return_value = "test_session"

        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        frame = CausalFrame(
            frame_id="target123",
            depth=0,
            parent_id=None,
            children=[],
            query="Target analysis",
            context_slice=context,
            evidence=[],
            conclusion="Old result",
            confidence=0.7,
            invalidation_condition={"description": "Changed"},
            status=FrameStatus.INVALIDATED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        mock_result = RLPALoopResult(
            answer="New analysis",
            iterations=1,
            depth_used=1,
            tokens_used=100,
            execution_time_ms=100.0,
            history=[],
        )
        mock_agent_run.return_value = mock_result

        result = await cmd_resume(frame_id="target", args={})

        assert "Resumed Frame" in result
        assert "target" in result
        mock_agent_run.assert_called_once()

    @patch("src.skills.causal_router.analyzer_agent.run")
    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_resume_with_canonical_task(self, mock_find_session, mock_index_load, mock_agent_run):
        """cmd_resume uses canonical_task for query if available."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.canonical_task import CanonicalTask
        from src.frame.frame_index import FrameIndex
        from src.repl.rlaph_loop import RLPALoopResult

        mock_find_session.return_value = "test_session"

        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        canonical_task = CanonicalTask(
            task_type="analyze",
            target=["src/auth.py"],
            analysis_scope="security",
        )

        frame = CausalFrame(
            frame_id="canonical123",
            depth=0,
            parent_id=None,
            children=[],
            query="Original query",
            context_slice=context,
            evidence=[],
            conclusion="Result",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.INVALIDATED,
            created_at=datetime.now(),
            canonical_task=canonical_task,
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        mock_result = RLPALoopResult(
            answer="Resumed analysis",
            iterations=1,
            depth_used=1,
            tokens_used=100,
            execution_time_ms=100.0,
            history=[],
        )
        mock_agent_run.return_value = mock_result

        await cmd_resume(frame_id="canonical", args={})

        # Check that canonical task was used in query
        call_args = mock_agent_run.call_args
        assert "analyze" in call_args[1]["query"]
        assert "src/auth.py" in call_args[1]["query"]

    @patch("src.skills.causal_router.analyzer_agent.run")
    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    @pytest.mark.asyncio
    async def test_cmd_resume_no_invalidated_frames(self, mock_find_session, mock_index_load, mock_agent_run):
        """cmd_resume returns message when no invalidated frames available."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        mock_find_session.return_value = "test_session"

        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        # Only completed frames, no invalidated
        frame = CausalFrame(
            frame_id="completed123",
            depth=0,
            parent_id=None,
            children=[],
            query="Completed analysis",
            context_slice=context,
            evidence=[],
            conclusion="Result",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        result = await cmd_resume(frame_id=None, args={})

        assert "No invalidated frames to resume" in result
        mock_agent_run.assert_not_called()


class TestCmdClearCache:
    """Test suite for cmd_clear_cache handler (Task 68)."""

    def test_cmd_clear_cache_returns_success(self):
        """cmd_clear_cache returns success message."""
        result = cmd_clear_cache()

        assert "✓" in result or "OK" in result or "Success" in result
        assert "ContextMap" in result or "cache" in result.lower()
        assert "cleared" in result.lower() or "next run" in result.lower()


class TestCmdStatus:
    """Test suite for cmd_status handler (Task 65)."""

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_no_frames(self, mock_config_load, mock_index_load):
        """cmd_status returns message when no frames found."""
        from src.frame.frame_index import FrameIndex

        # Setup mocks
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 5
        mock_config.status_icons = True

        mock_index_load.return_value = FrameIndex()  # Empty index

        result = await cmd_status(topic=None, args={})

        assert "No frames found" in result
        assert "/causal analyze" in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_with_completed_frames(self, mock_config_load, mock_index_load):
        """cmd_status shows completed frames table."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        # Setup config mock
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 5
        mock_config.status_icons = True

        # Create test frames
        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        frame1 = CausalFrame(
            frame_id="abc123def456",
            depth=0,
            parent_id=None,
            children=[],
            query="Analyze authentication flow",
            context_slice=context,
            evidence=[],
            conclusion="Auth flow is secure",
            confidence=0.95,
            invalidation_condition={"description": "File changes"},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        frame2 = CausalFrame(
            frame_id="def789ghi012",
            depth=1,
            parent_id="abc123def456",
            children=[],
            query="Check session management",
            context_slice=context,
            evidence=[],
            conclusion="Sessions timeout correctly",
            confidence=0.88,
            invalidation_condition={"description": "Config changes"},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame1)
        index.add(frame2)
        mock_index_load.return_value = index

        result = await cmd_status(topic=None, args={})

        # Check dashboard structure
        assert "## CausalFrame Status" in result
        assert "### Summary" in result
        assert "Valid frames: 2" in result
        assert "### Valid Frames" in result
        assert "abc123de" in result  # Truncated frame_id
        assert "0.9" in result or "0.8" in result  # Confidence values

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_with_invalidated_frames(self, mock_config_load, mock_index_load):
        """cmd_status shows invalidated frames with reasons."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        # Setup config mock
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 5
        mock_config.status_icons = True

        # Create invalidated frame
        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        frame = CausalFrame(
            frame_id="bad111bad222",
            depth=0,
            parent_id=None,
            children=[],
            query="Analyze deprecated API",
            context_slice=context,
            evidence=[],
            conclusion="API is deprecated",
            confidence=0.70,
            invalidation_condition={"description": "src/api.py was modified"},
            status=FrameStatus.INVALIDATED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        result = await cmd_status(topic=None, args={})

        assert "Invalidated frames: 1" in result
        assert "### Invalidated Frames" in result
        assert "src/api.py was modified" in result
        assert "/causal resume" in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_with_suspended_frames(self, mock_config_load, mock_index_load):
        """cmd_status shows suspended branches."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        # Setup config mock
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 5
        mock_config.status_icons = True

        # Create suspended frame
        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        frame = CausalFrame(
            frame_id="sus333sus444",
            depth=1,
            parent_id="parent123",
            children=[],
            query="Deep investigation of bug",
            context_slice=context,
            evidence=[],
            conclusion=None,
            confidence=0.0,
            invalidation_condition={},
            status=FrameStatus.SUSPENDED,
            created_at=datetime.now(),
            branched_from="parent123",
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        result = await cmd_status(topic=None, args={})

        assert "Suspended branches: 1" in result
        assert "### Suspended Branches" in result
        assert "sus333su" in result  # Truncated frame_id

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_respects_limit(self, mock_config_load, mock_index_load):
        """cmd_status limits number of frames shown per status."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        # Setup config with limit=2
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 2
        mock_config.status_icons = True

        # Create 5 completed frames
        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        index = FrameIndex()
        for i in range(5):
            frame = CausalFrame(
                frame_id=f"frame{i:04d}" + "x" * 12,
                depth=0,
                parent_id=None,
                children=[],
                query=f"Query {i}",
                context_slice=context,
                evidence=[],
                conclusion=f"Conclusion {i}",
                confidence=0.8 + i * 0.02,
                invalidation_condition={},
                status=FrameStatus.COMPLETED,
                created_at=datetime.now(),
            )
            index.add(frame)

        mock_index_load.return_value = index

        result = await cmd_status(topic=None, args={})

        # Should show "2 more" message
        assert "and 3 more" in result or "and 3" in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_icons_disabled(self, mock_config_load, mock_index_load):
        """cmd_status uses text icons when disabled in config."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        # Setup config with icons disabled
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 5
        mock_config.status_icons = False

        # Create test frame
        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        frame = CausalFrame(
            frame_id="test123test456",
            depth=0,
            parent_id=None,
            children=[],
            query="Test query",
            context_slice=context,
            evidence=[],
            conclusion="Test conclusion",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        result = await cmd_status(topic=None, args={})

        # Should use text icons
        assert "[OK]" in result
        assert "✓" not in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_filters_by_topic(self, mock_config_load, mock_index_load):
        """cmd_status filters frames by topic when provided."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        # Setup config
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 5
        mock_config.status_icons = True

        # Create frames with different topics
        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        frame1 = CausalFrame(
            frame_id="auth123auth456",
            depth=0,
            parent_id=None,
            children=[],
            query="Analyze authentication module",
            context_slice=context,
            evidence=[],
            conclusion="Auth is secure",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        frame2 = CausalFrame(
            frame_id="db123db456",
            depth=0,
            parent_id=None,
            children=[],
            query="Check database schema",
            context_slice=context,
            evidence=[],
            conclusion="Schema is valid",
            confidence=0.85,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame1)
        index.add(frame2)
        mock_index_load.return_value = index

        # Filter by "auth"
        result = await cmd_status(topic="auth", args={})

        # Should only show auth frame
        assert "Valid frames: 1" in result
        assert "authentication" in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_wildcard_topic(self, mock_config_load, mock_index_load):
        """cmd_status shows all frames when topic is **/*."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        # Setup config
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 5
        mock_config.status_icons = True

        # Create test frames
        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        index = FrameIndex()
        for i in range(3):
            frame = CausalFrame(
                frame_id=f"frame{i:04d}" + "x" * 12,
                depth=0,
                parent_id=None,
                children=[],
                query=f"Query {i}",
                context_slice=context,
                evidence=[],
                conclusion=f"Conclusion {i}",
                confidence=0.8,
                invalidation_condition={},
                status=FrameStatus.COMPLETED,
                created_at=datetime.now(),
            )
            index.add(frame)

        mock_index_load.return_value = index

        # Use wildcard topic
        result = await cmd_status(topic="**/*", args={})

        # Should show all frames
        assert "Valid frames: 3" in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_truncates_long_queries(self, mock_config_load, mock_index_load):
        """cmd_status truncates long query text in tables."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        # Setup config
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 5
        mock_config.status_icons = True

        # Create frame with long query
        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        long_query = "This is a very long query that should be truncated " * 3

        frame = CausalFrame(
            frame_id="long123long456",
            depth=0,
            parent_id=None,
            children=[],
            query=long_query,
            context_slice=context,
            evidence=[],
            conclusion="Conclusion",
            confidence=0.9,
            invalidation_condition={},
            status=FrameStatus.COMPLETED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        result = await cmd_status(topic=None, args={})

        # Query should be truncated with "..."
        assert "..." in result
        # But not the full query
        assert long_query not in result

    @patch("src.skills.causal_router.FrameIndex.load")
    @patch("src.skills.causal_router.CFConfig.load")
    @pytest.mark.asyncio
    async def test_cmd_status_suggestions_section(self, mock_config_load, mock_index_load):
        """cmd_status includes suggestions section."""
        from datetime import datetime
        from src.frame.causal_frame import CausalFrame, FrameStatus
        from src.frame.context_slice import ContextSlice
        from src.frame.frame_index import FrameIndex

        # Setup config
        mock_config = mock_config_load.return_value
        mock_config.status_limit = 5
        mock_config.status_icons = True

        # Create invalidated frame
        context = ContextSlice(
            files={},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )

        frame = CausalFrame(
            frame_id="inv123inv456",
            depth=0,
            parent_id=None,
            children=[],
            query="Invalidated query",
            context_slice=context,
            evidence=[],
            conclusion="Old conclusion",
            confidence=0.7,
            invalidation_condition={"description": "File changed"},
            status=FrameStatus.INVALIDATED,
            created_at=datetime.now(),
        )

        index = FrameIndex()
        index.add(frame)
        mock_index_load.return_value = index

        result = await cmd_status(topic=None, args={})

        assert "**Suggestions:**" in result
        assert "/causal resume inv123in" in result
        assert "--full" in result
        assert "/causal tree" in result


class TestGetSessionId:
    """Test suite for get_session_id function."""

    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    def test_session_flag_takes_priority(self, mock_find_session):
        """--session flag takes priority over other flags."""
        mock_find_session.return_value = "recent_session"

        args = {"session": "specific_session", "last": True}
        result = get_session_id(args)

        assert result == "specific_session"
        # Should not call find_most_recent_session when --session is provided
        mock_find_session.assert_not_called()

    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    def test_last_flag_uses_most_recent(self, mock_find_session):
        """--last flag uses most recent session."""
        mock_find_session.return_value = "recent_session"

        args = {"last": True}
        result = get_session_id(args)

        assert result == "recent_session"
        mock_find_session.assert_called_once()

    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    def test_no_session_flags_defaults_to_most_recent(self, mock_find_session):
        """Without --session or --last, defaults to most recent session."""
        mock_find_session.return_value = "recent_session"

        args = {"verbose": True}
        result = get_session_id(args)

        assert result == "recent_session"
        mock_find_session.assert_called_once()

    @patch("src.skills.causal_router.FrameStore.find_most_recent_session")
    def test_empty_args_defaults_to_most_recent(self, mock_find_session):
        """Empty args dict defaults to most recent session."""
        mock_find_session.return_value = "recent_session"

        args = {}
        result = get_session_id(args)

        assert result == "recent_session"
        mock_find_session.assert_called_once()
