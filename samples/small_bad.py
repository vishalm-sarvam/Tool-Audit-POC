"""A poorly-written async tool — multiple intentional issues planted."""

import time
import hashlib
import requests
from langchain.tools import BaseTool


class BadWebFetcher(BaseTool):
    name: str = "bad_web_fetcher"
    description: str = "Fetches a URL (badly)."

    async def _arun(self, url: str) -> str:
        # ISSUE 1 (line 15): blocking sleep in async function
        time.sleep(1)

        # ISSUE 2 (line 18): sync HTTP call in async function
        response = requests.get(url, timeout=30)
        data = response.text

        return f"Fetched {len(data)} chars"

    def _run(self, url: str) -> str:
        raise NotImplementedError("Use async version")


class BadFileProcessor(BaseTool):
    name: str = "bad_file_processor"
    description: str = "Processes a file (badly)."

    async def _arun(self, file_path: str) -> str:
        # ISSUE 3 (line 32): open() without context manager
        f = open(file_path)
        content = f.read()
        f.close()

        # ISSUE 4 (line 36): CPU-bound crypto in async without to_thread
        digest = hashlib.sha256(content.encode()).hexdigest()

        return f"Hash: {digest}"

    def _run(self, file_path: str) -> str:
        raise NotImplementedError("Use async version")
