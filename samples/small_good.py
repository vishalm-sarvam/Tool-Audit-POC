"""A well-written async tool — no issues expected."""

import asyncio
import hashlib
from typing import Optional

import httpx
import aiofiles
from langchain.tools import BaseTool


class AsyncWebFetcher(BaseTool):
    name: str = "async_web_fetcher"
    description: str = "Fetches a URL asynchronously and returns the content hash."

    async def _arun(self, url: str, save_path: Optional[str] = None) -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.text

        content_hash = await asyncio.to_thread(hashlib.sha256, content.encode())

        if save_path:
            async with aiofiles.open(save_path, "w") as f:
                await f.write(content)

        return f"Fetched {len(content)} chars, hash={content_hash.hexdigest()}"

    def _run(self, url: str, save_path: Optional[str] = None) -> str:
        raise NotImplementedError("Use async version")


class AsyncDataProcessor(BaseTool):
    name: str = "async_data_processor"
    description: str = "Processes data entries asynchronously."

    async def _arun(self, entries: list[str]) -> dict:
        results = {}
        for entry in entries:
            await asyncio.sleep(0)  # yield control
            results[entry] = len(entry)
        return {"processed": len(results), "results": results}

    def _run(self, entries: list[str]) -> dict:
        raise NotImplementedError("Use async version")
