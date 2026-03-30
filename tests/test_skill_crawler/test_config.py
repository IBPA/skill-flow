"""Tests for configuration settings."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic_settings import SettingsConfigDict
from skill_crawler.config import Settings, get_settings

# Keys to remove from env so pydantic-settings doesn't pick them up.
_TOKEN_KEYS = [
    "SKILL_CRAWLER_SKILLSMP_API_TOKEN",
    "SKILL_CRAWLER_GITHUB_TOKEN",
]


def _stripped_env() -> dict[str, str]:
    """Return os.environ without token keys."""
    return {k: v for k, v in os.environ.items() if k not in _TOKEN_KEYS}


class _IsolatedSettings(Settings):
    """Settings subclass that skips .env file for testing."""

    model_config = SettingsConfigDict(
        env_prefix="SKILL_CRAWLER_",
        env_file=None,
        extra="ignore",
    )


def _isolated_settings(**kwargs: Any) -> Settings:
    """Create Settings isolated from .env and real env vars."""
    with patch.dict(os.environ, _stripped_env(), clear=True):
        return _IsolatedSettings(**kwargs)


def _isolated_load(config_path: Path) -> Settings:
    """Load Settings from a config file, isolated from .env."""
    with (
        patch.dict(os.environ, _stripped_env(), clear=True),
        patch("skill_crawler.config.Settings", _IsolatedSettings),
    ):
        return _IsolatedSettings.load(config_path)


class TestSettingsLoad:
    """Test Settings.load from config.json."""

    def test_load_nonexistent_config(self):
        """Test load returns defaults when config.json doesn't exist."""
        settings = Settings.load(Path("/nonexistent/config.json"))
        assert settings.max_workers == 4
        assert settings.rate_limit_delay == 1.0

    def test_load_with_crawler_section(self):
        """Test load applies crawler config section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "crawler": {
                            "max_workers": 8,
                            "rate_limit_delay": 2.0,
                            "request_timeout": 60.0,
                            "max_repo_size_mb": 100.0,
                        }
                    }
                )
            )
            settings = Settings.load(config_path)
            assert settings.max_workers == 8
            assert settings.rate_limit_delay == 2.0
            assert settings.request_timeout == 60.0
            assert settings.max_repo_size_mb == 100.0
            assert settings.crawler.max_workers == 8

    def test_load_with_storage_section(self):
        """Test load applies storage config section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"storage": {"data_dir": "/custom/data"}})
            )
            settings = Settings.load(config_path)
            assert settings.data_dir == Path("/custom/data")

    def test_load_with_tokens_section(self):
        """Test load applies token defaults from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "tokens": {
                            "skillsmp_api_token": "sk_test_123",
                            "github_token": "ghp_test_456",
                        }
                    }
                )
            )
            settings = _isolated_load(config_path)
            assert settings.skillsmp_api_token == "sk_test_123"
            assert settings.github_token == "ghp_test_456"

    def test_load_default_path(self):
        """Test load uses config/config.json by default."""
        settings = Settings.load()
        assert isinstance(settings, Settings)

    def test_load_empty_config(self):
        """Test load handles config with no relevant sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"version": "1.0"}))
            settings = Settings.load(config_path)
            assert settings.max_workers == 4


class TestApplyTokenDefaults:
    """Test _apply_token_defaults method."""

    def test_tokens_not_overridden_when_env_set(self):
        """Test config tokens don't override env-set values."""
        settings = Settings(
            skillsmp_api_token="env_token",
            github_token="env_gh_token",
        )
        settings.tokens.skillsmp_api_token = "config_token"
        settings.tokens.github_token = "config_gh_token"
        settings._apply_token_defaults()
        assert settings.skillsmp_api_token == "env_token"
        assert settings.github_token == "env_gh_token"

    def test_tokens_applied_when_env_not_set(self):
        """Test config tokens applied when env vars are None."""
        settings = _isolated_settings()
        settings.tokens.skillsmp_api_token = "config_token"
        settings.tokens.github_token = "config_gh_token"
        settings._apply_token_defaults()
        assert settings.skillsmp_api_token == "config_token"
        assert settings.github_token == "config_gh_token"

    def test_no_op_when_both_none(self):
        """Test no change when both env and config are None."""
        settings = _isolated_settings()
        settings._apply_token_defaults()
        assert settings.skillsmp_api_token is None
        assert settings.github_token is None


class TestSettingsProperties:
    """Test Settings computed properties."""

    def test_metadata_dir(self):
        """Test metadata_dir property."""
        settings = Settings()
        assert settings.metadata_dir == Path("./data/skills/_metadata")

    def test_ensure_directories(self):
        """Test ensure_directories creates all dirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(data_dir=Path(tmpdir))
            settings.ensure_directories()
            assert settings.skillsmp_dir.exists()
            assert settings.metadata_dir.exists()


class TestGetSettings:
    """Test get_settings function."""

    def test_returns_settings_instance(self):
        """Test get_settings returns a valid Settings object."""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.max_workers == 4
