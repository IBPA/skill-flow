"""Tests for SkillsMP API client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from skill_crawler.crawlers.skillsmp.api_client import SkillsMPAPIClient
from skill_crawler.utils.http import RateLimitedClient


def _mock_response(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    return resp


def _patch_get(**kwargs):
    """Shortcut for patching RateLimitedClient.get."""
    return patch.object(RateLimitedClient, "get", new_callable=AsyncMock, **kwargs)


class TestSkillsMPAPIClient:
    """Test SkillsMP API client methods."""

    def test_source_name(self):
        client = SkillsMPAPIClient()
        assert client.source_name == "skillsmp"

    def test_auth_header_set(self):
        client = SkillsMPAPIClient(api_token="sk_test")
        assert client.api_token == "sk_test"

    @pytest.mark.asyncio
    async def test_list_skills_first_page(self):
        client = SkillsMPAPIClient()
        resp = _mock_response(
            {
                "data": {
                    "skills": [
                        {
                            "id": "s1",
                            "name": "Skill 1",
                            "githubUrl": "https://github.com/a/b",
                        },
                        {"id": "s2", "name": "Skill 2"},
                    ],
                    "pagination": {"hasNext": True},
                }
            }
        )
        with _patch_get(return_value=resp):
            skills, next_cursor = await client.list_skills()
        assert len(skills) == 2
        assert skills[0].id == "s1"
        assert next_cursor == "2"

    @pytest.mark.asyncio
    async def test_list_skills_last_page(self):
        client = SkillsMPAPIClient()
        resp = _mock_response(
            {
                "data": {
                    "skills": [{"id": "s1", "name": "Skill"}],
                    "pagination": {"hasNext": False},
                }
            }
        )
        with _patch_get(return_value=resp):
            skills, next_cursor = await client.list_skills(cursor="5")
        assert len(skills) == 1
        assert next_cursor is None

    @pytest.mark.asyncio
    async def test_get_skill_success(self):
        client = SkillsMPAPIClient()
        resp = _mock_response(
            {
                "id": "my-skill",
                "name": "My Skill",
                "description": "A cool skill",
            }
        )
        with _patch_get(return_value=resp):
            skill = await client.get_skill("my-skill")
        assert skill is not None
        assert skill.name == "My Skill"

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self):
        client = SkillsMPAPIClient()
        with _patch_get(side_effect=RuntimeError("404")):
            result = await client.get_skill("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_search(self):
        client = SkillsMPAPIClient()
        resp = _mock_response(
            {
                "data": {
                    "skills": [
                        {"id": "s1", "name": "Match 1"},
                        {"id": "s2", "name": "Match 2"},
                    ]
                }
            }
        )
        with _patch_get(return_value=resp):
            results = await client.search("test query", limit=10)
        assert len(results) == 2

    def test_parse_skill_with_timestamp(self):
        client = SkillsMPAPIClient()
        skill = client._parse_skill(
            {
                "id": "s1",
                "name": "Skill",
                "updatedAt": 1700000000,
                "tags": ["python", "ai"],
                "stars": 42,
            }
        )
        assert skill.updated_at is not None
        assert skill.tags == ["python", "ai"]
        assert skill.stars == 42

    def test_parse_skill_minimal(self):
        client = SkillsMPAPIClient()
        skill = client._parse_skill({"slug": "fallback-id"})
        assert skill.id == "fallback-id"
        assert skill.name == ""

    @pytest.mark.asyncio
    async def test_aexit_without_session(self):
        """Exiting without entering should not raise."""
        client = SkillsMPAPIClient()
        await client.__aexit__(None, None, None)
