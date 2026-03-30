"""Configuration settings using Pydantic."""

import json
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CrawlerConfig(BaseModel):
    """Crawler configuration."""

    max_workers: int = Field(default=4, ge=1, le=20)
    rate_limit_delay: float = Field(default=1.0, ge=0.1)
    request_timeout: float = Field(default=30.0, ge=5.0)
    max_repo_size_mb: float = Field(default=50.0, ge=0)


class StorageConfig(BaseModel):
    """Storage configuration."""

    data_dir: str = "./data"
    skills_dir: str = "./data/skills"


class TokensConfig(BaseModel):
    """API tokens configuration."""

    skillsmp_api_token: str | None = None
    github_token: str | None = None


class Settings(BaseSettings):
    """Application settings loaded from environment variables or config.json."""

    model_config = SettingsConfigDict(
        env_prefix="SKILL_CRAWLER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API tokens (env vars take precedence)
    skillsmp_api_token: str | None = Field(
        default=None,
        description="SkillsMP API token (optional - falls back to scraper)",
    )
    github_token: str | None = Field(
        default=None,
        description="GitHub token for higher rate limits",
    )

    # Paths
    data_dir: Path = Field(
        default=Path("./data"),
        description="Base directory for data storage",
    )

    # Crawler settings
    max_workers: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum concurrent workers",
    )
    rate_limit_delay: float = Field(
        default=1.0,
        ge=0.1,
        description="Delay between requests in seconds",
    )
    request_timeout: float = Field(
        default=30.0,
        ge=5.0,
        description="HTTP request timeout in seconds",
    )
    max_repo_size_mb: float = Field(
        default=50.0,
        ge=0,
        description="Maximum repo size in MB (0 = no limit).",
    )

    # Nested config (from config.json)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    tokens: TokensConfig = Field(default_factory=TokensConfig)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Settings":
        """Load settings from config/config.json with env var overrides.

        Args:
            config_path: Path to config.json. Defaults to config/config.json.

        Returns:
            Settings instance with merged config and env vars.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "config.json"

        # Start with base settings (env vars)
        settings = cls()

        # Load and merge config.json if it exists
        if config_path.exists():
            with config_path.open() as f:
                data = json.load(f)

            # Apply nested config sections
            if "crawler" in data:
                settings.crawler = CrawlerConfig.model_validate(data["crawler"])
                # Sync top-level fields with nested config
                settings.max_workers = settings.crawler.max_workers
                settings.rate_limit_delay = settings.crawler.rate_limit_delay
                settings.request_timeout = settings.crawler.request_timeout
                settings.max_repo_size_mb = settings.crawler.max_repo_size_mb

            if "storage" in data:
                settings.storage = StorageConfig.model_validate(data["storage"])
                # Sync data_dir
                settings.data_dir = Path(settings.storage.data_dir)

            if "tokens" in data:
                settings.tokens = TokensConfig.model_validate(data["tokens"])
                # Apply token values only if env vars not set
                settings._apply_token_defaults()

        return settings

    def _apply_token_defaults(self) -> None:
        """Apply token defaults from config.json if env vars not set."""
        if self.skillsmp_api_token is None and self.tokens.skillsmp_api_token:
            self.skillsmp_api_token = self.tokens.skillsmp_api_token

        if self.github_token is None and self.tokens.github_token:
            self.github_token = self.tokens.github_token

    @property
    def skills_dir(self) -> Path:
        """Directory for downloaded skills."""
        return self.data_dir / "skills"

    @property
    def skillsmp_dir(self) -> Path:
        """Directory for SkillsMP skills."""
        return self.skills_dir / "skillsmp"

    @property
    def metadata_dir(self) -> Path:
        """Directory for metadata files."""
        return self.skills_dir / "_metadata"

    def ensure_directories(self) -> None:
        """Create all required directories."""
        self.skillsmp_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
