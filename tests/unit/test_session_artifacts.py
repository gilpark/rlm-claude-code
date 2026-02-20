"""Tests for SessionArtifacts and FileRecord."""

from src.session.session_artifacts import FileRecord, SessionArtifacts


def test_file_record_creation():
    """FileRecord should be creatable with all fields."""
    record = FileRecord(
        path="/path/to/file.py",
        hash="abc123",
        role="read"
    )
    assert record.path == "/path/to/file.py"
    assert record.hash == "abc123"
    assert record.role == "read"


def test_file_record_roles():
    """FileRecord should accept all role types."""
    read_record = FileRecord(path="a.py", hash="h1", role="read")
    modified_record = FileRecord(path="b.py", hash="h2", role="modified")
    created_record = FileRecord(path="c.py", hash="h3", role="created")

    assert read_record.role == "read"
    assert modified_record.role == "modified"
    assert created_record.role == "created"


def test_session_artifacts_creation():
    """SessionArtifacts should be creatable with all fields."""
    artifacts = SessionArtifacts(
        session_id="session-001",
        initial_prompt="Fix the bug in auth module",
        files={
            "auth.py": FileRecord(path="auth.py", hash="hash1", role="read"),
            "auth_test.py": FileRecord(path="auth_test.py", hash="hash2", role="modified"),
        },
        root_frame_id="frame-root",
        conversation_log="/path/to/transcript.json"
    )
    assert artifacts.session_id == "session-001"
    assert artifacts.initial_prompt == "Fix the bug in auth module"
    assert len(artifacts.files) == 2
    assert artifacts.files["auth.py"].role == "read"
    assert artifacts.files["auth_test.py"].role == "modified"
    assert artifacts.root_frame_id == "frame-root"
    assert artifacts.conversation_log == "/path/to/transcript.json"
