"""Tests for FrameStore - JSONL persistence for CausalFrames."""

import json
import pytest
from pathlib import Path
from datetime import datetime
import tempfile

from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.frame.frame_store import FrameStore


class TestFrameStore:
    """Test suite for FrameStore."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary FrameStore."""
        store_path = tmp_path / "test_session.jsonl"
        return FrameStore(path=store_path)

    @pytest.fixture
    def sample_frame(self):
        """Create a sample CausalFrame for testing."""
        context_slice = ContextSlice(
            files={"test.py": "abc123"},
            memory_refs=[],
            tool_outputs={},
            token_budget=1000,
        )
        return CausalFrame(
            frame_id="test_frame_1",
            depth=0,
            parent_id=None,
            children=[],
            query="test query",
            context_slice=context_slice,
            evidence=[],
            conclusion="test conclusion",
            confidence=0.9,
            invalidation_condition="file changes",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

    def test_frame_store_instantiation(self, tmp_path):
        """FrameStore can be instantiated."""
        store_path = tmp_path / "test.jsonl"
        store = FrameStore(path=store_path)
        assert store.path == store_path

    def test_save_frame(self, temp_store, sample_frame):
        """FrameStore.save persists a frame."""
        temp_store.save(sample_frame)
        assert temp_store.path.exists()

    def test_load_frame(self, temp_store, sample_frame):
        """FrameStore.load retrieves a frame by ID."""
        temp_store.save(sample_frame)
        loaded = temp_store.load(sample_frame.frame_id)
        assert loaded is not None
        assert loaded.frame_id == sample_frame.frame_id
        assert loaded.query == sample_frame.query

    def test_load_nonexistent_frame(self, temp_store):
        """FrameStore.load returns None for nonexistent ID."""
        loaded = temp_store.load("nonexistent")
        assert loaded is None

    def test_list_frames(self, temp_store, sample_frame):
        """FrameStore.list returns all frames."""
        temp_store.save(sample_frame)
        frames = temp_store.list()
        assert len(frames) == 1
        assert frames[0].frame_id == sample_frame.frame_id

    def test_list_empty_store(self, temp_store):
        """FrameStore.list returns empty list when empty."""
        frames = temp_store.list()
        assert frames == []

    def test_find_by_status(self, temp_store, sample_frame):
        """FrameStore.find_by_status filters by status."""
        sample_frame.status = FrameStatus.COMPLETED
        temp_store.save(sample_frame)

        completed = temp_store.find_by_status(FrameStatus.COMPLETED)
        assert len(completed) == 1

        running = temp_store.find_by_status(FrameStatus.RUNNING)
        assert len(running) == 0

    def test_multiple_frames(self, temp_store, sample_frame):
        """FrameStore handles multiple frames."""
        frame2 = CausalFrame(
            frame_id="test_frame_2",
            depth=1,
            parent_id="test_frame_1",
            children=[],
            query="child query",
            context_slice=sample_frame.context_slice,
            evidence=["test_frame_1"],
            conclusion="child conclusion",
            confidence=0.8,
            invalidation_condition="parent invalidates",
            status=FrameStatus.COMPLETED,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now(),
        )

        temp_store.save(sample_frame)
        temp_store.save(frame2)

        frames = temp_store.list()
        assert len(frames) == 2
