"""Demonstrates httpx client leak — TOOLING_API Common Mistake #5.

The Sarvam TOOLING_API explicitly warns:
  'Creating httpx.AsyncClient() without async with context manager'
  causes 'Connection pool exhaustion, intermittent timeout errors.'

This file has 3 planted issues covering patterns from TOOLING_API.md:
  1. httpx.AsyncClient() without context manager (leaked client)
  2. requests.get() in async function (sync HTTP in async)
  3. Raising exception instead of returning ToolOutput (Common Mistake #3)
"""

import httpx
import requests
from typing import Optional


class LeakyHttpTool:
    """Tool that leaks httpx connections — missing async with context manager."""

    async def run(self, url: str) -> dict:
        # ISSUE 1 (line 22): httpx.AsyncClient() without context manager — connection leak
        client = httpx.AsyncClient()
        response = await client.get(url, timeout=30.0)
        # Never calls await client.aclose()
        return {"status": response.status_code, "body": response.text[:100]}


class SyncHttpInAsync:
    """Tool that uses sync requests inside async — TOOLING_API explicitly forbids this."""

    async def run(self, url: str) -> dict:
        # ISSUE 2 (line 32): requests.get() blocks the async event loop
        response = requests.get(url, timeout=30)
        return {"status": response.status_code, "length": len(response.text)}


class ExceptionRaisingTool:
    """Tool that raises exceptions instead of returning ToolOutput — Common Mistake #3."""

    async def run(self, data: str) -> dict:
        # ISSUE 3 (line 41): Raising exception instead of returning error ToolOutput
        if not data:
            raise ValueError("data cannot be empty")
        return {"processed": data.upper()}
