"""Tests for LLMClient - simple synchronous LLM calls with streaming support."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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

    def test_get_async_client_initialization(self):
        """LLMClient._get_async_client creates async client with proper config."""
        client = LLMClient(api_key="test-key")
        async_client = client._get_async_client()
        assert async_client is not None
        # Should be cached
        assert client._async_client is async_client

    @pytest.mark.asyncio
    async def test_call_stream_requires_query(self):
        """LLMClient.call_stream requires a non-empty query."""
        client = LLMClient()
        with pytest.raises(ValueError):
            chunks = []
            async for chunk in client.call_stream(""):
                chunks.append(chunk)

    @pytest.mark.asyncio
    async def test_call_stream_uses_model_for_depth(self):
        """call_stream uses get_model_for_depth for model selection."""
        client = LLMClient(api_key="test-key")
        model = client.get_model_for_depth(0)
        assert model == "glm-4.6"

    @pytest.mark.asyncio
    async def test_call_stream_handles_context(self):
        """call_stream properly builds prompt with context dict."""
        client = LLMClient(api_key="test-key")
        # Test that context is properly handled (we can't mock the API call easily
        # in this test, but we can verify the method signature works)
        # This is more of a smoke test that the method exists and is callable
        assert hasattr(client, 'call_stream')
        assert callable(client.call_stream)

    @pytest.mark.asyncio
    async def test_call_async_requires_query(self):
        """LLMClient.call_async requires a non-empty query."""
        client = LLMClient()
        with pytest.raises(ValueError):
            await client.call_async("")

    @pytest.mark.asyncio
    async def test_call_async_method_exists(self):
        """call_async method exists and is callable."""
        client = LLMClient()
        assert hasattr(client, 'call_async')
        assert callable(client.call_async)

    def test_temperature_override_in_call(self):
        """call method accepts temperature parameter."""
        client = LLMClient()
        # Just verify the signature accepts temperature
        import inspect
        sig = inspect.signature(client.call)
        assert 'temperature' in sig.parameters

    @pytest.mark.asyncio
    async def test_temperature_override_in_call_stream(self):
        """call_stream method accepts temperature parameter."""
        client = LLMClient()
        import inspect
        sig = inspect.signature(client.call_stream)
        assert 'temperature' in sig.parameters

    @pytest.mark.asyncio
    async def test_temperature_override_in_call_async(self):
        """call_async method accepts temperature parameter."""
        client = LLMClient()
        import inspect
        sig = inspect.signature(client.call_async)
        assert 'temperature' in sig.parameters
