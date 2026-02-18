#!/usr/bin/env python3
"""Test progress callback for ClaudeHeadlessClient."""

import asyncio
from src.api_client import ClaudeHeadlessClient


async def main():
    progress_events = []

    def on_progress(elapsed: int, timeout: float):
        progress_events.append(elapsed)
        print(f"[LLM] Waiting... {elapsed}s (timeout: {int(timeout)}s)")

    client = ClaudeHeadlessClient(timeout=120.0)

    print("[LLM] Calling sonnet...")
    response = await client.complete(
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        model="sonnet",
        progress_callback=on_progress,
    )

    print(f"\n[LLM] Response received ({len(progress_events)} progress events)")
    print(f"[LLM] Content: {response.content[:100]}...")
    print(f"[LLM] Tokens: {response.input_tokens} in / {response.output_tokens} out")


if __name__ == "__main__":
    asyncio.run(main())
