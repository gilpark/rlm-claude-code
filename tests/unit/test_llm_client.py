"""Tests for LLMClient - simple synchronous LLM calls."""

import pytest
from src.repl.llm_client import LLMClient, LLMError


class TestLLMClient:
    """Test suite for LLMClient."""

    def test_llm_client_instantiation(self):
        """LLMClient can be instantiated with defaults."""
        client = LLMClient()
        assert client is not None

    def test_llm_client_with_api_key(self):
        """LLMClient accepts api_key parameter."""
        client = LLMClient(api_key="test-key")
        assert client.api_key == "test-key"

    def test_get_model_for_depth_root(self):
        """Root depth (0) returns root model (glm-4.6 by default)."""
        client = LLMClient()
        model = client.get_model_for_depth(0)
        # Default config uses glm-4.6 for root
        assert model == "glm-4.6"

    def test_get_model_for_depth_deep(self):
        """Deep depth (3+) returns model from config.recursive_depth_3."""
        client = LLMClient()
        model = client.get_model_for_depth(3)
        # Should return the configured model for depth 3
        assert model.startswith("glm-")  # Valid GLM model

    def test_call_requires_query(self):
        """LLMClient.call requires a query."""
        client = LLMClient()
        with pytest.raises(ValueError):
            client.call("")
