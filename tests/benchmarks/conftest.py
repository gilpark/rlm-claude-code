"""
Benchmark test configuration.

Provides fixtures and configuration for benchmark tests.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def warmup_tiktoken():
    """
    Warm up tiktoken encoding before benchmarks run.

    tiktoken.get_encoding() downloads encoding data on first use,
    which can cause benchmarks to appear to hang. This fixture
    pre-loads the encoding at session startup.

    Issue: https://github.com/rand/rlm-claude-code/issues/1
    """
    try:
        from src.cost_tracker import estimate_tokens

        # Force tiktoken initialization with a small text
        estimate_tokens("warmup")
    except ImportError:
        pass  # tiktoken not available, tests will use fallback
