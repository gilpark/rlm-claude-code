"""Tests for session_schema module."""

from datetime import datetime


def test_session_context_data_accepts_causal_frame():
    """SessionContextData.causal_frames should accept CausalFrame objects."""
    from src.causal_frame import CausalFrame, FrameStatus
    from src.context_slice import ContextSlice
    from src.session_schema import SessionContextData

    frame = CausalFrame(
        frame_id="test123",
        depth=0,
        parent_id=None,
        children=[],
        query="test query",
        context_slice=ContextSlice(files={}, memory_refs=[], tool_outputs={}, token_budget=1000),
        evidence=[],
        conclusion="test conclusion",
        confidence=0.9,
        invalidation_condition="test fails",
        status=FrameStatus.COMPLETED,
        branched_from=None,
        escalation_reason=None,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )

    ctx = SessionContextData(causal_frames=[frame])
    assert len(ctx.causal_frames) == 1
