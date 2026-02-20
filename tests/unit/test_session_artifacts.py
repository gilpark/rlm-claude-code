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


class TestSessionArtifactsPersistence:
    """Tests for SessionArtifacts save/load."""

    def test_save_to_file(self, tmp_path):
        """SessionArtifacts.save persists to JSON file."""
        artifacts = SessionArtifacts(
            session_id="test-session",
            initial_prompt="Fix the bug",
            files={"auth.py": FileRecord("auth.py", "abc123", "read")},
            root_frame_id="frame-001",
            conversation_log="/path/to/log.json",
        )

        save_path = artifacts.save(base_dir=tmp_path)

        assert save_path.exists()
        assert save_path.name == "artifacts.json"

    def test_load_from_file(self, tmp_path):
        """SessionArtifacts.load reconstructs from JSON file."""
        # First save
        artifacts = SessionArtifacts(
            session_id="test-session",
            initial_prompt="Fix the bug",
            files={"auth.py": FileRecord("auth.py", "abc123", "read")},
            root_frame_id="frame-001",
            conversation_log="/path/to/log.json",
        )
        artifacts.save(base_dir=tmp_path)

        # Then load
        loaded = SessionArtifacts.load("test-session", base_dir=tmp_path)

        assert loaded.session_id == "test-session"
        assert loaded.initial_prompt == "Fix the bug"
        assert "auth.py" in loaded.files
        assert loaded.files["auth.py"].hash == "abc123"

    def test_load_nonexistent_returns_none(self, tmp_path):
        """SessionArtifacts.load returns None if file doesn't exist."""
        loaded = SessionArtifacts.load("nonexistent", base_dir=tmp_path)
        assert loaded is None
