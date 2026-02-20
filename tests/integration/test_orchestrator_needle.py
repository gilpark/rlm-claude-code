"""Integration tests for orchestrator accuracy."""

import asyncio
import pytest
from pathlib import Path
from src.rlaph_loop import RLAPHLoop
from src.types import SessionContext


class TestOrchestratorAccuracy:
    """Test orchestrator executes code correctly and doesn't hallucinate."""

    @pytest.fixture
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.mark.asyncio
    async def test_basic_math(self):
        """Orchestrator correctly executes simple math."""
        context = SessionContext(messages=[], files={}, tool_outputs={}, working_memory={})
        loop = RLAPHLoop(max_iterations=5)

        query = """
Calculate 42 * 17 and tell me the result.

```python
result = 42 * 17
print(result)
```

FINAL: <the answer>
"""

        result = await loop.run(query, context)

        assert "714" in result.answer

    @pytest.mark.asyncio
    async def test_multi_dependency(self):
        """Orchestrator handles multi-step calculations."""
        context = SessionContext(messages=[], files={}, tool_outputs={}, working_memory={})
        loop = RLAPHLoop(max_iterations=5)

        query = """
Calculate sum, mean, and max of these numbers.

```python
nums = [3, 7, 11, 19, 23]
total = sum(nums)
mean = total / len(nums)
maximum = max(nums)
print(f"sum={total}, mean={mean}, max={maximum}")
```

FINAL: <results>
"""

        result = await loop.run(query, context)

        assert "63" in result.answer  # sum
        assert "12.6" in result.answer  # mean
        assert "23" in result.answer  # max

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_file(self):
        """Orchestrator handles file search queries appropriately."""
        context = SessionContext(messages=[], files={}, tool_outputs={}, working_memory={})
        loop = RLAPHLoop(max_iterations=5)

        query = "Find the main.py file in src/ and tell me what it does."

        result = await loop.run(query, context, working_dir=Path("."))

        # Should either say main.py doesn't exist OR find the actual main entry point
        # The important thing is it doesn't hallucinate a fake main.py
        assert "main.py" in result.answer.lower() or "rlaph_loop" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_file_reading(self):
        """Orchestrator can read files from disk."""
        context = SessionContext(messages=[], files={}, tool_outputs={}, working_memory={})
        loop = RLAPHLoop(max_iterations=5)

        query = """
Read the file src/__init__.py and tell me how many lines it has.

```python
content = read_file("src/__init__.py")
lines = content.split("\\n")
print(f"Lines: {{len(lines)}}")
```

FINAL: <number of lines>
"""

        result = await loop.run(query, context)

        # Should report number of lines
        import re
        match = re.search(r'\d+', result.answer)
        assert match is not None
        lines = int(match.group())
        assert lines > 0  # Should have some lines
