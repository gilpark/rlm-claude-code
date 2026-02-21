"""Tests for canonical task extraction - language/framework agnostic."""
import pytest

from src.frame.canonical_task import CanonicalTask
from src.frame.intent_extractor import extract_canonical_task


# -----------------------------------------------------------------------------
# CanonicalTask hash tests
# -----------------------------------------------------------------------------

def test_canonical_task_hash_stable():
    """Same canonical task should produce same hash."""
    task1 = CanonicalTask(task_type="analyze", target=["auth.py"], analysis_scope="correctness")
    task2 = CanonicalTask(task_type="analyze", target=["auth.py"], analysis_scope="correctness")
    assert task1.to_hash() == task2.to_hash()


def test_canonical_task_hash_different():
    """Different tasks should produce different hashes."""
    task1 = CanonicalTask(task_type="analyze", target=["auth.py"])
    task2 = CanonicalTask(task_type="debug", target=["auth.py"])
    assert task1.to_hash() != task2.to_hash()


def test_canonical_task_serialization():
    """Should serialize/deserialize correctly."""
    task = CanonicalTask(
        task_type="analyze",
        target=["auth.py"],
        analysis_scope="security",
        params={"key": "value"},
    )

    data = task.to_dict()
    restored = CanonicalTask.from_dict(data)

    assert restored.task_type == task.task_type
    assert restored.target == task.target
    assert restored.analysis_scope == task.analysis_scope
    assert restored.params == task.params


# -----------------------------------------------------------------------------
# Language-agnostic tests (work for any stack)
# -----------------------------------------------------------------------------

def test_extract_python_file():
    """Should extract Python file from query."""
    task = extract_canonical_task("Analyze auth.py for security issues")
    assert task.task_type == "analyze"
    assert "auth.py" in task.target


def test_extract_typescript_file():
    """Should extract TypeScript file from query."""
    task = extract_canonical_task("Debug the user.service.ts error")
    assert task.task_type == "debug"
    assert "user.service.ts" in task.target or "service.ts" in task.target


def test_extract_go_file():
    """Should extract Go file from query."""
    task = extract_canonical_task("Explain how main.go works")
    assert task.task_type == "explain"
    assert "main.go" in task.target


def test_extract_rust_file():
    """Should extract Rust file from query."""
    task = extract_canonical_task("Review lib.rs for performance")
    assert task.task_type == "analyze"
    assert "lib.rs" in task.target
    assert task.analysis_scope == "performance"


def test_extract_directory_target():
    """Should extract directory path from query."""
    task = extract_canonical_task("Summarize src/auth module")
    assert task.task_type == "summarize"
    assert any("auth" in t for t in task.target)


def test_extract_generic_module():
    """Should extract module name with glob pattern."""
    task = extract_canonical_task("Implement the cache service")
    assert task.task_type == "implement"
    assert any("cache" in t.lower() for t in task.target)


def test_security_scope_override():
    """Should detect security scope."""
    task = extract_canonical_task("Analyze security vulnerabilities in the code")
    assert task.task_type == "analyze"
    assert task.analysis_scope == "security"


def test_ultimate_fallback():
    """Should fallback to whole codebase when no target found."""
    task = extract_canonical_task("What do you think?")
    assert task.task_type == "analyze"
    assert task.target == ["**/*"]


def test_extract_debug_task():
    """Should extract debug task from debug query."""
    task = extract_canonical_task("Debug the login function")
    assert task.task_type == "debug"


def test_extract_verify_task():
    """Should extract verify task."""
    task = extract_canonical_task("Verify the auth.py implementation")
    assert task.task_type == "verify"


def test_extract_refactor_task():
    """Should extract refactor task."""
    task = extract_canonical_task("Refactor the auth module")
    assert task.task_type == "refactor"


def test_extract_document_task():
    """Should extract document task."""
    task = extract_canonical_task("Document the auth.py file")
    assert task.task_type == "document"


def test_extract_architecture_scope():
    """Should detect architecture scope."""
    task = extract_canonical_task("Analyze the architecture of auth.py")
    assert task.task_type == "analyze"
    assert task.analysis_scope == "architecture"


def test_str_representation():
    """Should have human-readable string representation."""
    task = CanonicalTask(
        task_type="analyze",
        target=["auth.py"],
        analysis_scope="security",
    )
    s = str(task)
    assert "analyze" in s
    assert "security" in s


def test_canonical_task_frozen():
    """CanonicalTask should be frozen (immutable)."""
    task = CanonicalTask(task_type="analyze", target=["auth.py"])
    with pytest.raises(Exception):  # FrozenInstanceError
        task.task_type = "debug"
