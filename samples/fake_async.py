"""Third-party async integration tool."""

from langchain.tools import BaseTool


class ExternalApiClient:
    """Async HTTP client wrapper for external API calls."""

    async def get(self, url: str) -> str:
        return f"response from {url}"


class WebDataFetcher(BaseTool):
    name: str = "web_data_fetcher"
    description: str = "Fetches data from external web APIs."

    async def _arun(self, url: str) -> str:
        client = ExternalApiClient()
        result = await client.get(url)
        return result

    def _run(self, url: str) -> str:
        raise NotImplementedError("Use async version")
