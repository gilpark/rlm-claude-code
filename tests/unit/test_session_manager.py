"""
Unit tests for session_manager module.

Implements: Session State Consolidation Plan Phase 1
Implements: Spec §5.2 Hook Integration tests
"""

import json
import pytest
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.session_manager import SessionManager, get_session_manager
from src.session_schema import (
    ActivationMode,
    CellIndex,
    SessionState,
)


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager with temp directory."""
        return SessionManager(base_dir=temp_sessions_dir)

    def test_init_creates_base_dir(self, temp_sessions_dir):
        """Initialization creates base directory."""
        manager = SessionManager(base_dir=temp_sessions_dir / "new-dir")
        assert manager.base_dir.exists()

    def test_create_session(self, session_manager, temp_sessions_dir):
        """Can create a new session."""
        state = session_manager.create_session(
            session_id="test-session-1",
            cwd="/test/path",
            transcript_path="/path/to/transcript.jsonl",
        )

        assert state.metadata.session_id == "test-session-1"
        assert state.metadata.cwd == "/test/path"
        assert state.metadata.claude_transcript_path == "/path/to/transcript.jsonl"
        assert session_manager.current_session is state

        # Check directory structure
        session_dir = temp_sessions_dir / "test-session-1"
        assert session_dir.exists()
        assert (session_dir / "session.json").exists()
        assert (session_dir / "cells").exists()
        assert (session_dir / "reasoning").exists()

    def test_create_session_with_metadata(self, session_manager):
        """Can create session with additional metadata."""
        state = session_manager.create_session(
            session_id="test-session-2",
            cwd="/test/path",
            metadata={
                "tags": ["important", "test"],
                "description": "Test session",
            },
        )

        assert "important" in state.metadata.tags
        assert "test" in state.metadata.tags
        assert state.metadata.description == "Test session"

    def test_current_symlink_updated(self, session_manager, temp_sessions_dir):
        """Current symlink is updated on session creation."""
        session_manager.create_session("session-a", cwd="/path/a")
        session_manager.create_session("session-b", cwd="/path/b")

        current_link = temp_sessions_dir / "current"
        assert current_link.is_symlink()
        assert current_link.resolve().name == "session-b"

    def test_cell_index_initialized(self, session_manager, temp_sessions_dir):
        """Cell index is initialized on session creation."""
        session_manager.create_session("test-session-3", cwd="/test")

        cell_index_file = temp_sessions_dir / "test-session-3" / "cells" / "index.json"
        assert cell_index_file.exists()

        with open(cell_index_file) as f:
            index_data = json.load(f)

        assert index_data["session_id"] == "test-session-3"
        assert index_data["cells"] == {}

    def test_load_session(self, session_manager):
        """Can load an existing session."""
        # Create session
        original = session_manager.create_session(
            session_id="load-test",
            cwd="/original/path",
        )
        original.activation.rlm_active = True
        original.budget.total_tokens_used = 5000
        session_manager.save_session()

        # Clear current and reload
        session_manager._current_session = None
        session_manager._current_session_dir = None

        loaded = session_manager.load_session("load-test")

        assert loaded.metadata.session_id == "load-test"
        assert loaded.metadata.cwd == "/original/path"
        assert loaded.activation.rlm_active is True
        assert loaded.budget.total_tokens_used == 5000

    def test_load_nonexistent_raises(self, session_manager):
        """Load nonexistent session raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            session_manager.load_session("nonexistent")

    def test_save_session(self, session_manager, temp_sessions_dir):
        """Can save session state."""
        state = session_manager.create_session("save-test", cwd="/test")
        state.activation.current_depth = 3
        state.budget.total_tokens_used = 10000

        saved_path = session_manager.save_session()
        assert saved_path.exists()

        # Verify saved content
        with open(saved_path) as f:
            data = json.load(f)

        assert data["activation"]["current_depth"] == 3
        assert data["budget"]["total_tokens_used"] == 10000

    def test_session_exists(self, session_manager):
        """session_exists returns correct status."""
        assert not session_manager.session_exists("new-session")

        session_manager.create_session("new-session", cwd="/test")

        assert session_manager.session_exists("new-session")

    def test_delete_session(self, session_manager, temp_sessions_dir):
        """Can delete a session."""
        session_manager.create_session("delete-test", cwd="/test")
        assert session_manager.session_exists("delete-test")

        session_manager.delete_session("delete-test")

        assert not session_manager.session_exists("delete-test")
        assert not (temp_sessions_dir / "delete-test").exists()

    def test_delete_current_session_clears_state(self, session_manager):
        """Deleting current session clears in-memory state."""
        session_manager.create_session("delete-current", cwd="/test")
        assert session_manager.current_session is not None

        session_manager.delete_session("delete-current")

        assert session_manager.current_session is None

    def test_list_sessions(self, session_manager):
        """Can list all sessions."""
        session_manager.create_session("session-a", cwd="/a")
        session_manager.create_session("session-b", cwd="/b")
        session_manager.create_session("session-c", cwd="/c")

        sessions = session_manager.list_sessions()

        assert len(sessions) == 3
        assert "session-a" in sessions
        assert "session-b" in sessions
        assert "session-c" in sessions

    def test_list_sessions_with_type_filter(self, session_manager):
        """Can filter sessions by type."""
        session_manager.create_session("default-session", cwd="/")
        session_manager.create_session(
            "migrated-session",
            cwd="/",
            metadata={"session_type": "migrated"},
        )

        all_sessions = session_manager.list_sessions()
        assert len(all_sessions) == 2

        migrated_sessions = session_manager.list_sessions(session_type="migrated")
        assert len(migrated_sessions) == 1
        assert "migrated-session" in migrated_sessions

    def test_list_sessions_with_tags_filter(self, session_manager):
        """Can filter sessions by tags."""
        session_manager.create_session(
            "tagged-session",
            cwd="/",
            metadata={"tags": ["important", "test"]},
        )
        session_manager.create_session(
            "other-session",
            cwd="/",
            metadata={"tags": ["other"]},
        )

        important_sessions = session_manager.list_sessions(tags=["important"])
        assert len(important_sessions) == 1
        assert "tagged-session" in important_sessions

        # Multiple tags - session must have ALL tags
        multi_tag_sessions = session_manager.list_sessions(tags=["important", "test"])
        assert len(multi_tag_sessions) == 1

    def test_get_current_session_id(self, session_manager):
        """Can get current session ID from symlink."""
        assert session_manager.get_current_session_id() is None

        session_manager.create_session("current-session", cwd="/test")

        assert session_manager.get_current_session_id() == "current-session"

    def test_archive_session(self, session_manager, temp_sessions_dir):
        """Can archive a session."""
        session_manager.create_session("archive-test", cwd="/test")

        archived_path = session_manager.archive_session("archive-test")

        assert archived_path.exists()
        assert archived_path == temp_sessions_dir / "archive" / "archive-test"
        assert not session_manager.session_exists("archive-test")

    def test_fork_session(self, session_manager):
        """Can fork a session."""
        original = session_manager.create_session(
            "original-session",
            cwd="/original",
            metadata={"tags": ["parent"]},
        )
        original.context.working_memory["key"] = "value"
        session_manager.save_session()

        forked = session_manager.fork_session("original-session", "forked-session")

        assert forked.metadata.session_id == "forked-session"
        assert forked.metadata.parent_session_id == "original-session"
        assert forked.metadata.session_type == "fork"
        assert "parent" in forked.metadata.tags
        assert forked.context.working_memory["key"] == "value"

        # Verify both exist
        assert session_manager.session_exists("original-session")
        assert session_manager.session_exists("forked-session")

    def test_update_activation(self, session_manager):
        """Can update activation status."""
        session_manager.create_session("activation-test", cwd="/test")

        session_manager.update_activation(True, reason="Complex task")

        assert session_manager.current_session.activation.rlm_active is True
        assert session_manager.current_session.activation.activation_reason == "Complex task"

    def test_update_depth(self, session_manager):
        """Can update recursion depth."""
        session_manager.create_session("depth-test", cwd="/test")

        session_manager.update_depth(2)

        assert session_manager.current_session.activation.current_depth == 2

    def test_add_tokens(self, session_manager):
        """Can add tokens and costs."""
        session_manager.create_session("tokens-test", cwd="/test")

        session_manager.add_tokens(1000, model="claude-sonnet", cost=0.01)

        assert session_manager.current_session.budget.total_tokens_used == 1000
        assert session_manager.current_session.budget.cost_usd == 0.01
        assert "claude-sonnet" in session_manager.current_session.budget.by_model

    def test_increment_recursive_calls(self, session_manager):
        """Can increment recursive call counter."""
        session_manager.create_session("calls-test", cwd="/test")

        session_manager.increment_recursive_calls(2)
        session_manager.increment_recursive_calls(3)

        assert session_manager.current_session.budget.total_recursive_calls == 5

    def test_update_working_memory(self, session_manager):
        """Can update working memory."""
        session_manager.create_session("memory-test", cwd="/test")

        session_manager.update_working_memory("key1", "value1")
        session_manager.update_working_memory("key2", {"nested": True})

        assert session_manager.current_session.context.working_memory["key1"] == "value1"
        assert session_manager.current_session.context.working_memory["key2"] == {"nested": True}

    def test_end_session(self, session_manager):
        """Can mark session as ended."""
        session_manager.create_session("end-test", cwd="/test")

        assert session_manager.current_session.metadata.ended_at is None

        session_manager.end_session()

        assert session_manager.current_session.metadata.ended_at is not None

    def test_get_session_dir(self, session_manager, temp_sessions_dir):
        """get_session_dir returns correct path."""
        session_dir = session_manager.get_session_dir("test-id")

        assert session_dir == temp_sessions_dir / "test-id"

    def test_get_session_file(self, session_manager, temp_sessions_dir):
        """get_session_file returns correct path."""
        session_file = session_manager.get_session_file("test-id")

        assert session_file == temp_sessions_dir / "test-id" / "session.json"


class TestSessionManagerAtomicWrites:
    """Tests for atomic write behavior."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager with temp directory."""
        return SessionManager(base_dir=temp_sessions_dir)

    def test_atomic_write_no_temp_files_left(self, session_manager, temp_sessions_dir):
        """No temp files left after successful write."""
        session_manager.create_session("atomic-test", cwd="/test")
        session_manager.save_session()

        session_dir = temp_sessions_dir / "atomic-test"
        temp_files = list(session_dir.glob("*.tmp"))

        assert len(temp_files) == 0

    def test_concurrent_saves_preserve_data(self, session_manager):
        """Multiple saves preserve data integrity."""
        session_manager.create_session("concurrent-test", cwd="/test")

        # Make multiple rapid saves
        for i in range(10):
            session_manager.current_session.budget.total_tokens_used = i * 100
            session_manager.save_session()

        # Reload and verify
        session_manager._current_session = None
        loaded = session_manager.load_session("concurrent-test")

        assert loaded.budget.total_tokens_used == 900

    def test_session_json_valid(self, session_manager, temp_sessions_dir):
        """session.json is valid JSON after save."""
        session_manager.create_session("json-test", cwd="/test")
        session_manager.current_session.context.working_memory["test"] = "value"
        session_manager.save_session()

        session_file = temp_sessions_dir / "json-test" / "session.json"

        with open(session_file) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "activation" in data
        assert "context" in data
        assert "budget" in data


class TestGetSessionManager:
    """Tests for get_session_manager function."""

    def test_returns_singleton(self, monkeypatch):
        """get_session_manager returns same instance."""
        import src.session_manager as sm

        # Reset global
        sm._session_manager = None

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))

            m1 = get_session_manager()
            m2 = get_session_manager()

            assert m1 is m2

            # Clean up
            sm._session_manager = None


class TestSessionStateSchema:
    """Tests for SessionState Pydantic model."""

    def test_session_state_default_values(self):
        """SessionState has expected default values."""
        from src.session_schema import (
            SessionActivation,
            SessionBudget,
            SessionContextData,
            SessionMetadata,
        )

        state = SessionState(
            metadata=SessionMetadata(
                session_id="test",
                created_at=time.time(),
                updated_at=time.time(),
                cwd="/test",
            ),
        )

        assert state.activation.rlm_active is False
        assert state.activation.activation_mode == "complexity"
        assert state.budget.total_tokens_used == 0
        assert state.context.files == {}
        assert state.context.tool_outputs == []

    def test_session_state_validation(self):
        """SessionState validates input."""
        from src.session_schema import Cell, CellInput, CellType

        # Invalid cell_id pattern should fail (must match cell_{hex8})
        with pytest.raises(Exception):  # Pydantic ValidationError
            Cell(
                cell_id="invalid-id",  # Doesn't match pattern
                created_at=time.time(),
                type=CellType.REPL,
                input=CellInput(source="test", operation="test"),
            )

    def test_session_state_json_serialization(self):
        """SessionState can be serialized to JSON."""
        from src.session_schema import (
            SessionActivation,
            SessionMetadata,
        )

        state = SessionState(
            metadata=SessionMetadata(
                session_id="json-test",
                created_at=1000.0,
                updated_at=2000.0,
                cwd="/test",
            ),
            activation=SessionActivation(rlm_active=True),
        )

        json_str = state.model_dump_json()
        data = json.loads(json_str)

        assert data["metadata"]["session_id"] == "json-test"
        assert data["activation"]["rlm_active"] is True

    def test_session_state_from_dict(self):
        """SessionState can be created from dict."""
        data = {
            "metadata": {
                "session_id": "dict-test",
                "created_at": 1000.0,
                "updated_at": 2000.0,
                "cwd": "/test",
                "claude_transcript_path": None,
                "rlm_version": "0.1.0",
                "parent_session_id": None,
                "session_type": "default",
                "tags": [],
                "description": None,
                "ended_at": None,
            },
            "activation": {
                "rlm_active": True,
                "activation_mode": "manual",
                "activation_reason": "Test",
                "complexity_score": 0.8,
                "current_depth": 1,
                "max_depth": 2,
            },
            "context": {
                "files": {"test.py": {"hash": None, "size_bytes": None, "first_access": None, "last_access": None}},
                "tool_outputs": [],
                "working_memory": {"key": "value"},
            },
            "budget": {
                "total_tokens_used": 1000,
                "total_recursive_calls": 5,
                "max_recursive_calls": 10,
                "cost_usd": 0.01,
                "by_model": {},
            },
            "cells": {
                "count": 0,
                "index_path": "cells/index.json",
            },
            "trajectory": {
                "events_count": 0,
                "export_path": None,
            },
        }

        state = SessionState(**data)

        assert state.metadata.session_id == "dict-test"
        assert state.activation.rlm_active is True
        assert state.context.working_memory["key"] == "value"


# Causal Frame Tests


class TestCausalFrameStorage:
    """Tests for CausalFrame storage in SessionManager."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager with temp directory."""
        return SessionManager(base_dir=temp_sessions_dir)

    def test_save_frame_stores_causal_frame(self, session_manager, temp_sessions_dir):
        """save_frame should serialize and store a CausalFrame."""
        from datetime import datetime

        from src.causal_frame import CausalFrame, FrameStatus
        from src.context_slice import ContextSlice

        session_manager.create_session("test-session", cwd="/tmp")

        frame = CausalFrame(
            frame_id="frame001",
            depth=0,
            parent_id=None,
            children=[],
            query="test query",
            context_slice=ContextSlice(
                files={}, memory_refs=[], tool_outputs={}, token_budget=1000
            ),
            evidence=[],
            conclusion="done",
            confidence=0.9,
            invalidation_condition="never",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

        frame_id = session_manager.save_frame(frame)
        assert frame_id == "frame001"

        # Verify frame is in session
        frames = session_manager.get_session_frames()
        assert len(frames) == 1
        assert frames[0].frame_id == "frame001"

    def test_frame_persistence_round_trip(self, session_manager, temp_sessions_dir):
        """Frame should survive save/load cycle."""
        from datetime import datetime

        from src.causal_frame import CausalFrame, FrameStatus
        from src.context_slice import ContextSlice

        # Create and save
        session_manager.create_session("persist-test", cwd="/tmp")

        frame = CausalFrame(
            frame_id="persist001",
            depth=0,
            parent_id=None,
            children=[],
            query="persistence test",
            context_slice=ContextSlice(
                files={"/path/to/file.py": "abc123"},
                memory_refs=["mem1"],
                tool_outputs={"tool1": "output1"},
                token_budget=2000,
            ),
            evidence=["evidence1"],
            conclusion="persisted",
            confidence=0.95,
            invalidation_condition="file changes",
            status=FrameStatus.VERIFIED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

        session_manager.save_frame(frame)

        # Load in new manager instance
        manager2 = SessionManager(base_dir=temp_sessions_dir)
        manager2.load_session("persist-test")

        loaded = manager2.load_frame("persist001")
        assert loaded is not None
        assert loaded.frame_id == "persist001"
        assert loaded.query == "persistence test"
        assert loaded.status == FrameStatus.VERIFIED
        assert loaded.context_slice.files == {"/path/to/file.py": "abc123"}

    def test_save_frame_updates_parent_children(self, session_manager, temp_sessions_dir):
        """Saving a child frame should update parent's children list."""
        from datetime import datetime

        from src.causal_frame import CausalFrame, FrameStatus
        from src.context_slice import ContextSlice

        session_manager.create_session("parent-test", cwd="/tmp")

        # Create parent frame
        parent = CausalFrame(
            frame_id="parent001",
            depth=0,
            parent_id=None,
            children=[],
            query="parent query",
            context_slice=ContextSlice(
                files={}, memory_refs=[], tool_outputs={}, token_budget=1000
            ),
            evidence=[],
            conclusion="parent done",
            confidence=0.9,
            invalidation_condition="",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )
        session_manager.save_frame(parent)

        # Create child frame
        child = CausalFrame(
            frame_id="child001",
            depth=1,
            parent_id="parent001",
            children=[],
            query="child query",
            context_slice=ContextSlice(
                files={}, memory_refs=[], tool_outputs={}, token_budget=1000
            ),
            evidence=["parent001"],
            conclusion="child done",
            confidence=0.8,
            invalidation_condition="",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )
        session_manager.save_frame(child)

        # Verify parent's children list was updated
        updated_parent = session_manager.load_frame("parent001")
        assert updated_parent is not None
        assert "child001" in updated_parent.children

