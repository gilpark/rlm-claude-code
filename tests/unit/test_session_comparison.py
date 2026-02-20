"""Tests for session comparison."""

from datetime import datetime

from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.frame.frame_index import FrameIndex
from src.session.session_artifacts import FileRecord, SessionArtifacts
from src.session.session_comparison import SessionDiff, compare_sessions


def make_artifacts(session_id: str, prompt: str, files: dict) -> SessionArtifacts:
    """Helper to create SessionArtifacts for testing."""
    return SessionArtifacts(
        session_id=session_id,
        initial_prompt=prompt,
        files=files,
        root_frame_id=f"root-{session_id}",
        conversation_log=f"/path/{session_id}.json"
    )


def test_compare_sessions_same_task():
    """Same prompt should indicate same task."""
    current = make_artifacts(
        "curr",
        "Fix auth bug",
        {"auth.py": FileRecord("auth.py", "hash1", "read")}
    )
    prior = make_artifacts(
        "prior",
        "Fix auth bug",
        {"auth.py": FileRecord("auth.py", "hash1", "read")}
    )

    diff = compare_sessions(current, prior)
    assert diff.same_task is True
    assert len(diff.changed_files) == 0


def test_compare_sessions_detects_file_changes():
    """Different file hashes should be detected."""
    current = make_artifacts(
        "curr",
        "Fix auth bug",
        {
            "auth.py": FileRecord("auth.py", "hash2", "read"),  # changed
            "new.py": FileRecord("new.py", "hash3", "created"),  # new
        }
    )
    prior = make_artifacts(
        "prior",
        "Fix auth bug",
        {"auth.py": FileRecord("auth.py", "hash1", "read")}
    )

    diff = compare_sessions(current, prior)
    assert "auth.py" in diff.changed_files
    assert "new.py" in diff.changed_files  # new files count as changed


def test_compare_sessions_different_task():
    """Different prompt should indicate different task."""
    current = make_artifacts(
        "curr",
        "Fix auth bug",
        {"auth.py": FileRecord("auth.py", "hash1", "read")}
    )
    prior = make_artifacts(
        "prior",
        "Add new feature",  # different prompt
        {"auth.py": FileRecord("auth.py", "hash1", "read")}
    )

    diff = compare_sessions(current, prior)
    assert diff.same_task is False


def test_compare_sessions_returns_session_diff():
    """compare_sessions should return SessionDiff dataclass."""
    current = make_artifacts(
        "curr",
        "Test",
        {}
    )
    prior = make_artifacts(
        "prior",
        "Test",
        {}
    )

    diff = compare_sessions(current, prior)
    assert isinstance(diff, SessionDiff)
    assert hasattr(diff, "same_task")
    assert hasattr(diff, "changed_files")
    assert hasattr(diff, "invalidated_frame_ids")
    assert hasattr(diff, "resumable_frames")


def make_frame(frame_id: str, files: dict[str, str], status: FrameStatus = FrameStatus.COMPLETED) -> CausalFrame:
    """Helper to create a CausalFrame for testing."""
    context = ContextSlice(files=files, memory_refs=[], tool_outputs={}, token_budget=1000)
    return CausalFrame(
        frame_id=frame_id,
        depth=0,
        parent_id=None,
        children=[],
        query=f"query for {frame_id}",
        context_slice=context,
        evidence=[],
        conclusion="done",
        confidence=0.9,
        invalidation_condition="",
        status=status,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now()
    )


def test_compare_sessions_with_frame_index_finds_invalidated():
    """compare_sessions with FrameIndex should find invalidated frames."""
    current = make_artifacts(
        "curr",
        "Fix auth bug",
        {"auth.py": FileRecord("auth.py", "hash2", "modified")}
    )
    prior = make_artifacts(
        "prior",
        "Fix auth bug",
        {"auth.py": FileRecord("auth.py", "hash1", "read")}
    )

    # Frame that references auth.py - should be invalidated
    frame = make_frame("frame1", files={"auth.py": "hash1"})

    index = FrameIndex()
    index.add(frame)

    diff = compare_sessions(current, prior, index=index)
    assert "frame1" in diff.invalidated_frame_ids


def test_compare_sessions_with_frame_index_no_invalidated():
    """compare_sessions with FrameIndex should not invalidate unchanged files."""
    current = make_artifacts(
        "curr",
        "Fix auth bug",
        {"auth.py": FileRecord("auth.py", "hash1", "read")}
    )
    prior = make_artifacts(
        "prior",
        "Fix auth bug",
        {"auth.py": FileRecord("auth.py", "hash1", "read")}
    )

    # Frame that references auth.py - not invalidated since file didn't change
    frame = make_frame("frame1", files={"auth.py": "hash1"})

    index = FrameIndex()
    index.add(frame)

    diff = compare_sessions(current, prior, index=index)
    assert len(diff.invalidated_frame_ids) == 0


def test_compare_sessions_with_frame_index_finds_resumable():
    """compare_sessions with FrameIndex should find resumable suspended frames."""
    current = make_artifacts(
        "curr",
        "Fix auth bug",
        {}
    )
    prior = make_artifacts(
        "prior",
        "Fix auth bug",
        {}
    )

    # Suspended frame that might be resumable
    suspended_frame = make_frame("suspended1", files={}, status=FrameStatus.SUSPENDED)
    completed_frame = make_frame("completed1", files={}, status=FrameStatus.COMPLETED)

    index = FrameIndex()
    index.add(suspended_frame)
    index.add(completed_frame)

    diff = compare_sessions(current, prior, index=index)
    assert "suspended1" in diff.resumable_frames
    assert "completed1" not in diff.resumable_frames


def test_compare_sessions_different_task_no_resumable():
    """compare_sessions should not suggest resuming frames for different tasks."""
    current = make_artifacts(
        "curr",
        "New task",
        {}
    )
    prior = make_artifacts(
        "prior",
        "Different task",
        {}
    )

    suspended_frame = make_frame("suspended1", files={}, status=FrameStatus.SUSPENDED)

    index = FrameIndex()
    index.add(suspended_frame)

    diff = compare_sessions(current, prior, index=index)
    assert len(diff.resumable_frames) == 0
