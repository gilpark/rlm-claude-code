"""Tests for session comparison."""

from src.session_artifacts import FileRecord, SessionArtifacts
from src.session_comparison import SessionDiff, compare_sessions


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
