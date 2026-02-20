#!/usr/bin/env python3
"""Debug LLM output format."""
import os
from src.llm_client import LLMClient
from src.response_parser import ResponseParser

client = LLMClient()

print("Testing glm-4.7 model output format...")
print("=" * 50)

response = client.call(
    query='Write a simple Python code block that prints hello world. Use ```python formatting. Then say FINAL: done',
    model='glm-4.7',
    max_tokens=500
)

print("=== RAW RESPONSE ===")
print(repr(response[:1000]))
print()
print("=== FORMATTED ===")
print(response)
print()
print("=== PARSED ===")
parser = ResponseParser()
parsed = parser.parse(response)
for i, item in enumerate(parsed):
    print(f"{i}: {item.action.value} -> {item.content[:100] if item.content else 'empty'}...")
