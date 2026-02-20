"""Tests for query/intent tracking in FrameIndex."""
import pytest
from pathlib import Path
from src.frame.frame_index import FrameIndex


def test_frame_index_stores_initial_query():
    """FrameIndex should store the initial user query."""
    index = FrameIndex(initial_query="Analyze the auth module")

    assert index.initial_query == "Analyze the auth module"


def test_frame_index_query_summary():
    """FrameIndex can store a short query summary."""
    index = FrameIndex(
        initial_query="Analyze the authentication flow in the auth module",
        query_summary="Auth flow analysis"
    )

    assert index.query_summary == "Auth flow analysis"


def test_frame_index_save_load_with_query(tmp_path):
    """Query should persist across save/load."""
    index = FrameIndex(
        initial_query="Debug the login bug",
        query_summary="Login bug debug"
    )

    # Save
    index.save("test_query_session", tmp_path)

    # Load
    loaded = FrameIndex.load("test_query_session", tmp_path)

    assert loaded.initial_query == "Debug the login bug"
    assert loaded.query_summary == "Login bug debug"


def test_frame_index_defaults_empty_query():
    """FrameIndex defaults to empty query strings."""
    index = FrameIndex()

    assert index.initial_query == ""
    assert index.query_summary == ""
