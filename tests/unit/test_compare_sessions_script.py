"""Tests for compare_sessions.py script."""

import json
from datetime import datetime
from io import StringIO
from unittest.mock import Mock, patch

from src.frame.causal_frame import CausalFrame, FrameStatus
from src.frame.context_slice import ContextSlice
from src.frame.frame_index import FrameIndex
from src.session.session_artifacts import FileRecord, SessionArtifacts


class TestPrintInvalidatedFramesSummary:
    """Tests for _print_invalidated_frames_summary function."""

    def setup_method(self):
        """Import the module under test."""
        import sys
        from pathlib import Path

        # Add parent to path for imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

        # Import after path manipulation
        from scripts import compare_sessions

        self.compare_sessions = compare_sessions
        self.capture_output = StringIO()

    def make_frame(
        self,
        frame_id: str,
        files: dict[str, str],
        status: FrameStatus = FrameStatus.COMPLETED,
        invalidation_desc: str = "File changed"
    ) -> CausalFrame:
        """Helper to create a CausalFrame for testing."""
        context = ContextSlice(
            files=files,
            memory_refs=[],
            tool_outputs={},
            token_budget=1000
        )
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
            invalidation_condition={"description": invalidation_desc},
            status=status,
            branched_from=None,
            escalation_reason=None,
            created_at=datetime.now(),
            completed_at=datetime.now()
        )

    def test_no_invalidated_frames_prints_nothing(self):
        """When no frames are invalidated, nothing should be printed."""
        with patch('sys.stdout', new=self.capture_output):
            self.compare_sessions._print_invalidated_frames_summary(
                "session-123",
                [],
                None
            )

        output = self.capture_output.getvalue()
        assert output == ""

    def test_prints_single_invalidated_frame(self):
        """Single invalidated frame should print with description."""
        frame = self.make_frame(
            "abc123def456",
            {"test.py": "hash1"},
            invalidation_desc="test.py changed or was deleted"
        )
        index = FrameIndex()
        index.add(frame)

        with patch('sys.stdout', new=self.capture_output):
            self.compare_sessions._print_invalidated_frames_summary(
                "session-123",
                ["abc123def456"],
                index
            )

        output = self.capture_output.getvalue()
        assert "## Invalidated Frames from Prior Session" in output
        assert "Session: `session-123`" in output
        assert "abc123de: test.py changed or was deleted" in output
        assert "Suggestion: Use `/causal resume`" in output

    def test_prints_multiple_invalidated_frames(self):
        """Multiple invalidated frames should all be listed."""
        frame1 = self.make_frame(
            "abc123def456",
            {"test.py": "hash1"},
            invalidation_desc="test.py changed or was deleted"
        )
        frame2 = self.make_frame(
            "xyz789uvw012",
            {"other.py": "hash2"},
            invalidation_desc="other.py changes or is deleted"
        )
        index = FrameIndex()
        index.add(frame1)
        index.add(frame2)

        with patch('sys.stdout', new=self.capture_output):
            self.compare_sessions._print_invalidated_frames_summary(
                "session-456",
                ["abc123def456", "xyz789uvw012"],
                index
            )

        output = self.capture_output.getvalue()
        assert "abc123de: test.py changed or was deleted" in output
        assert "xyz789uv: other.py changes or is deleted" in output

    def test_limits_to_five_frames(self):
        """Should only show first 5 frames when more exist."""
        frames = []
        frame_ids = []
        for i in range(7):
            frame_id = f"frame{i:016x}"
            frame = self.make_frame(
                frame_id,
                {f"file{i}.py": "hash1"},
                invalidation_desc=f"file{i}.py changed"
            )
            frames.append(frame)
            frame_ids.append(frame_id)

        index = FrameIndex()
        for frame in frames:
            index.add(frame)

        with patch('sys.stdout', new=self.capture_output):
            self.compare_sessions._print_invalidated_frames_summary(
                "session-789",
                frame_ids,
                index
            )

        output = self.capture_output.getvalue()
        # First 5 frames should be shown (truncated to 8 chars)
        assert "frame000: file0.py changed" in output
        assert "frame000: file1.py changed" in output
        assert "frame000: file2.py changed" in output
        assert "frame000: file3.py changed" in output
        assert "frame000: file4.py changed" in output
        # Last 2 should not be shown individually
        assert "frame00005:" not in output
        assert "frame00006:" not in output
        # Should show "and 2 more"
        assert "and 2 more" in output

    def test_handles_missing_frame_in_index(self):
        """Should handle frames that are in the list but not in the index."""
        index = FrameIndex()  # Empty index

        with patch('sys.stdout', new=self.capture_output):
            self.compare_sessions._print_invalidated_frames_summary(
                "session-missing",
                ["missing_frame_id"],
                index
            )

        output = self.capture_output.getvalue()
        assert "missing_: (frame not found in index)" in output

    def test_handles_frame_without_invalidation_condition(self):
        """Should handle frames with no invalidation_condition."""
        frame = self.make_frame(
            "abc123def456",
            {"test.py": "hash1"}
        )
        # Remove invalidation_condition
        frame.invalidation_condition = None

        index = FrameIndex()
        index.add(frame)

        with patch('sys.stdout', new=self.capture_output):
            self.compare_sessions._print_invalidated_frames_summary(
                "session-empty",
                ["abc123def456"],
                index
            )

        output = self.capture_output.getvalue()
        assert "abc123de: Unknown reason" in output

    def test_handles_empty_invalidation_condition_dict(self):
        """Should handle frames with empty invalidation_condition dict."""
        frame = self.make_frame(
            "abc123def456",
            {"test.py": "hash1"},
            invalidation_desc=""
        )
        # Set to empty dict without description
        frame.invalidation_condition = {}

        index = FrameIndex()
        index.add(frame)

        with patch('sys.stdout', new=self.capture_output):
            self.compare_sessions._print_invalidated_frames_summary(
                "session-empty-dict",
                ["abc123def456"],
                index
            )

        output = self.capture_output.getvalue()
        assert "abc123de: Unknown reason" in output

    def test_none_index(self):
        """Should handle None index gracefully."""
        with patch('sys.stdout', new=self.capture_output):
            self.compare_sessions._print_invalidated_frames_summary(
                "session-none",
                ["abc123def456"],
                None
            )

        output = self.capture_output.getvalue()
        assert "abc123de: (frame not found in index)" in output
