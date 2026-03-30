"""Rate-limited async HTTP client."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RateLimitedClient:
    """HTTP client with rate limiting and retry logic."""

    def __init__(
        self,
        rate_limit: float = 1.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize rate-limited client.

        Args:
            rate_limit: Minimum delay between requests in seconds.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.
            headers: Default headers for all requests.
        """
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = headers or {}
        self._last_request_time: float = 0
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None

    async def _ensure_rate_limit(self) -> None:
        """Ensure minimum delay between requests."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()

    @asynccontextmanager
    async def session(self) -> AsyncIterator["RateLimitedClient"]:
        """Context manager for HTTP session."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True,
        )
        try:
            yield self
        finally:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make a rate-limited GET request.

        Args:
            url: Request URL.
            params: Query parameters.
            headers: Additional headers.

        Returns:
            HTTP response.

        Raises:
            httpx.HTTPStatusError: On HTTP error status.
        """
        await self._ensure_rate_limit()
        if self._client is None:
            msg = "Client not initialized. Use 'async with session()'"
            raise RuntimeError(msg)

        merged_headers = {**self.headers, **(headers or {})}
        response = await self._client.get(url, params=params, headers=merged_headers)
        response.raise_for_status()
        return response

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make a rate-limited POST request.

        Args:
            url: Request URL.
            json: JSON body.
            data: Form data.
            headers: Additional headers.

        Returns:
            HTTP response.

        Raises:
            httpx.HTTPStatusError: On HTTP error status.
        """
        await self._ensure_rate_limit()
        if self._client is None:
            msg = "Client not initialized. Use 'async with session()'"
            raise RuntimeError(msg)

        merged_headers = {**self.headers, **(headers or {})}
        response = await self._client.post(
            url, json=json, data=data, headers=merged_headers
        )
        response.raise_for_status()
        return response

    async def download(
        self,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        """Download binary content.

        Args:
            url: Download URL.
            headers: Additional headers.

        Returns:
            Downloaded bytes.
        """
        response = await self.get(url, headers=headers)
        return response.content
