"""Tests for frame serialization."""

from datetime import datetime

from src.causal_frame import CausalFrame, FrameStatus
from src.context_slice import ContextSlice
from src.frame_serialization import deserialize_frame, serialize_frame


def test_serialize_frame():
    """serialize_frame should convert CausalFrame to dict."""
    context = ContextSlice(
        files={"test.py": "abc123"},
        memory_refs=["mem1"],
        tool_outputs={"grep": "output"},
        token_budget=1000
    )
    frame = CausalFrame(
        frame_id="test123",
        depth=0,
        parent_id=None,
        children=["child1"],
        query="test query",
        context_slice=context,
        evidence=["evidence1"],
        conclusion="test conclusion",
        confidence=0.85,
        invalidation_condition="test condition",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime(2026, 2, 18, 12, 0, 0),
        completed_at=datetime(2026, 2, 18, 12, 5, 0)
    )

    serialized = serialize_frame(frame)

    assert isinstance(serialized, dict)
    assert serialized["frame_id"] == "test123"
    assert serialized["depth"] == 0
    assert serialized["query"] == "test query"
    assert serialized["confidence"] == 0.85
    assert serialized["status"] == "completed"
    assert serialized["context_slice"]["files"] == {"test.py": "abc123"}


def test_deserialize_frame():
    """deserialize_frame should convert dict back to CausalFrame."""
    data = {
        "frame_id": "test456",
        "depth": 1,
        "parent_id": "parent",
        "children": [],
        "query": "child query",
        "context_slice": {
            "files": {"child.py": "def456"},
            "memory_refs": [],
            "tool_outputs": {},
            "token_budget": 500,
        },
        "evidence": [],
        "conclusion": "child conclusion",
        "confidence": 0.9,
        "invalidation_condition": "child condition",
        "status": "running",
        "branched_from": None,
        "escalation_reason": None,
        "created_at": "2026-02-18T12:00:00",
        "completed_at": None,
    }

    frame = deserialize_frame(data)

    assert frame.frame_id == "test456"
    assert frame.depth == 1
    assert frame.parent_id == "parent"
    assert frame.query == "child query"
    assert frame.confidence == 0.9
    assert frame.context_slice.files == {"child.py": "def456"}


def test_serialize_deserialize_roundtrip():
    """serialize then deserialize should produce equivalent frame."""
    context = ContextSlice(
        files={"test.py": "abc123"},
        memory_refs=["mem1"],
        tool_outputs={"grep": "output"},
        token_budget=1000
    )
    original = CausalFrame(
        frame_id="test789",
        depth=0,
        parent_id=None,
        children=["child1"],
        query="test query",
        context_slice=context,
        evidence=["evidence1"],
        conclusion="test conclusion",
        confidence=0.85,
        invalidation_condition="test condition",
        status=FrameStatus.VERIFIED,
        branched_from="other",
        escalation_reason="test reason",
        created_at=datetime(2026, 2, 18, 12, 0, 0),
        completed_at=datetime(2026, 2, 18, 12, 5, 0)
    )

    serialized = serialize_frame(original)
    deserialized = deserialize_frame(serialized)

    assert deserialized.frame_id == original.frame_id
    assert deserialized.query == original.query
    assert deserialized.confidence == original.confidence
    assert deserialized.status == original.status
    assert deserialized.context_slice.files == original.context_slice.files
    assert deserialized.branched_from == original.branched_from
    assert deserialized.escalation_reason == original.escalation_reason
