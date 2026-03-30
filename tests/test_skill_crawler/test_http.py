"""Tests for rate-limited HTTP client."""

import pytest
from skill_crawler.utils.http import RateLimitedClient


class TestRateLimitedClientInit:
    """Test RateLimitedClient initialization."""

    def test_defaults(self):
        """Test default parameters."""
        client = RateLimitedClient()
        assert client.rate_limit == 1.0
        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.headers == {}

    def test_custom_params(self):
        """Test custom initialization parameters."""
        client = RateLimitedClient(
            rate_limit=0.5,
            timeout=60.0,
            max_retries=5,
            headers={"Authorization": "Bearer test"},
        )
        assert client.rate_limit == 0.5
        assert client.timeout == 60.0
        assert client.max_retries == 5
        assert client.headers == {"Authorization": "Bearer test"}


class TestRateLimitedClientSession:
    """Test session context manager."""

    @pytest.mark.asyncio
    async def test_session_creates_client(self):
        """Test session creates and closes httpx client."""
        client = RateLimitedClient()
        assert client._client is None

        async with client.session() as session:
            assert session._client is not None

        assert client._client is None

    @pytest.mark.asyncio
    async def test_get_without_session_raises(self):
        """Test calling get without session raises RuntimeError."""
        client = RateLimitedClient()
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await client.get("https://example.com")

    @pytest.mark.asyncio
    async def test_post_without_session_raises(self):
        """Test calling post without session raises RuntimeError."""
        client = RateLimitedClient()
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await client.post("https://example.com")


class TestRateLimitedClientRequests:
    """Test HTTP request methods with mocked responses."""

    @pytest.mark.asyncio
    async def test_get_request(self, httpx_mock):
        """Test successful GET request."""
        httpx_mock.add_response(url="https://api.test.com/data", json={"ok": True})

        client = RateLimitedClient(rate_limit=0.0)
        async with client.session():
            response = await client.get("https://api.test.com/data")
        assert response.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_get_with_params(self, httpx_mock):
        """Test GET request with query parameters."""
        httpx_mock.add_response(json={"results": []})

        client = RateLimitedClient(rate_limit=0.0)
        async with client.session():
            response = await client.get(
                "https://api.test.com/search",
                params={"q": "test"},
            )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_with_extra_headers(self, httpx_mock):
        """Test GET request merges extra headers."""
        httpx_mock.add_response(json={})

        client = RateLimitedClient(
            rate_limit=0.0,
            headers={"X-Base": "base"},
        )
        async with client.session():
            await client.get(
                "https://api.test.com/data",
                headers={"X-Extra": "extra"},
            )

        request = httpx_mock.get_request()
        assert request.headers["X-Base"] == "base"
        assert request.headers["X-Extra"] == "extra"

    @pytest.mark.asyncio
    async def test_post_request(self, httpx_mock):
        """Test successful POST request."""
        httpx_mock.add_response(url="https://api.test.com/submit", json={"id": 1})

        client = RateLimitedClient(rate_limit=0.0)
        async with client.session():
            response = await client.post(
                "https://api.test.com/submit",
                json={"name": "test"},
            )
        assert response.json() == {"id": 1}

    @pytest.mark.asyncio
    async def test_download(self, httpx_mock):
        """Test binary download."""
        content = b"binary file content"
        httpx_mock.add_response(url="https://dl.test.com/file.zip", content=content)

        client = RateLimitedClient(rate_limit=0.0)
        async with client.session():
            data = await client.download("https://dl.test.com/file.zip")
        assert data == content
