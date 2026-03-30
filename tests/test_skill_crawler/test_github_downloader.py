"""Tests for GitHub downloader."""

import io
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from skill_crawler.downloaders.github import GitHubDownloader
from skill_crawler.models.skill import SkillMetadata
from skill_crawler.utils.http import RateLimitedClient


def _make_skill(**kwargs) -> SkillMetadata:
    defaults = {"id": "test", "name": "test-skill", "source": "skillsmp"}
    return SkillMetadata(**(defaults | kwargs))


def _make_zip(files: dict[str, str], root: str = "owner-repo-abc123") -> bytes:
    """Create a zip archive mimicking GitHub's format."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(f"{root}/{path}", content)
    return buf.getvalue()


class TestParseGithubUrl:
    """Test URL parsing."""

    def setup_method(self):
        self.dl = GitHubDownloader()

    def test_simple_url(self):
        owner, repo, branch, path = self.dl._parse_github_url(
            "https://github.com/acme/tool"
        )
        assert (owner, repo, branch, path) == ("acme", "tool", None, None)

    def test_url_with_git_suffix(self):
        owner, repo, _, _ = self.dl._parse_github_url(
            "https://github.com/acme/tool.git"
        )
        assert (owner, repo) == ("acme", "tool")

    def test_tree_url(self):
        owner, repo, branch, path = self.dl._parse_github_url(
            "https://github.com/acme/mono/tree/main/.skills/my-skill"
        )
        assert owner == "acme"
        assert repo == "mono"
        assert branch == "main"
        assert path == ".skills/my-skill"

    def test_invalid_url(self):
        result = self.dl._parse_github_url("https://example.com/not-github")
        assert result == (None, None, None, None)

    def test_ssh_style_url(self):
        owner, repo, _, _ = self.dl._parse_github_url("github.com:acme/tool")
        assert (owner, repo) == ("acme", "tool")


class TestResolveFolderName:
    """Test folder name conflict resolution."""

    def test_no_conflict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = GitHubDownloader()
            assert dl._resolve_folder_name(Path(tmpdir), "my-skill") == "my-skill"

    def test_with_conflict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "my-skill").mkdir()
            dl = GitHubDownloader()
            assert dl._resolve_folder_name(Path(tmpdir), "my-skill") == "my-skill-1"

    def test_multiple_conflicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "my-skill").mkdir()
            (Path(tmpdir) / "my-skill-1").mkdir()
            dl = GitHubDownloader()
            assert dl._resolve_folder_name(Path(tmpdir), "my-skill") == "my-skill-2"


class TestExtractArchive:
    """Test archive extraction."""

    @pytest.mark.asyncio
    async def test_basic_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = GitHubDownloader()
            content = _make_zip({"SKILL.md": "# Hello", "README.md": "readme"})
            out = Path(tmpdir) / "skill"
            out.mkdir()

            content_hash = await dl._extract_archive(content, out)
            assert (out / "SKILL.md").read_text() == "# Hello"
            assert (out / "README.md").read_text() == "readme"
            assert len(content_hash) == 16

    @pytest.mark.asyncio
    async def test_extraction_with_skill_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = GitHubDownloader()
            content = _make_zip(
                {
                    ".skills/my-skill/SKILL.md": "# Skill",
                    "README.md": "root readme",
                }
            )
            out = Path(tmpdir) / "skill"
            out.mkdir()

            await dl._extract_archive(content, out, skill_path=".skills/my-skill")
            assert (out / "SKILL.md").read_text() == "# Skill"
            assert not (out / "README.md").exists()

    @pytest.mark.asyncio
    async def test_no_skill_md_returns_empty_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = GitHubDownloader()
            content = _make_zip({"README.md": "no skill file"})
            out = Path(tmpdir) / "skill"
            out.mkdir()

            content_hash = await dl._extract_archive(content, out)
            assert content_hash == ""


class TestDownloadSkill:
    """Test the full download_skill flow."""

    @pytest.mark.asyncio
    async def test_no_github_url(self):
        dl = GitHubDownloader()
        skill = _make_skill(github_url=None)
        success, msg, _folder = await dl.download_skill(skill, Path("/tmp"))
        assert not success
        assert msg == "No GitHub URL"

    @pytest.mark.asyncio
    async def test_invalid_github_url(self):
        dl = GitHubDownloader()
        skill = _make_skill(github_url="https://example.com/bad")
        success, msg, _ = await dl.download_skill(skill, Path("/tmp"))
        assert not success
        assert "Invalid GitHub URL" in (msg or "")

    @pytest.mark.asyncio
    async def test_repo_size_exceeds_limit(self):
        dl = GitHubDownloader(max_repo_size_mb=10.0)
        skill = _make_skill(github_url="https://github.com/acme/big-repo")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"size": 20 * 1024}  # 20MB in KB
        with patch.object(
            RateLimitedClient, "get", new_callable=AsyncMock, return_value=mock_resp
        ):
            success, msg, _ = await dl.download_skill(skill, Path("/tmp"))
        assert not success
        assert "Skipped" in (msg or "")

    @pytest.mark.asyncio
    async def test_successful_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = GitHubDownloader()
            skill = _make_skill(github_url="https://github.com/acme/tool")

            zip_bytes = _make_zip({"SKILL.md": "# My Skill"})
            with patch.object(
                RateLimitedClient,
                "download",
                new_callable=AsyncMock,
                return_value=zip_bytes,
            ):
                success, content_hash, folder = await dl.download_skill(
                    skill, Path(tmpdir)
                )
            assert success
            assert len(content_hash or "") == 16
            assert (Path(tmpdir) / folder / "SKILL.md").exists()

    @pytest.mark.asyncio
    async def test_download_exception_returns_error(self):
        dl = GitHubDownloader()
        skill = _make_skill(github_url="https://github.com/acme/tool")
        with patch.object(
            RateLimitedClient,
            "download",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            success, msg, _ = await dl.download_skill(skill, Path("/tmp"))
        assert not success
        assert "boom" in (msg or "")

    @pytest.mark.asyncio
    async def test_timing_callback_called(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            timings: list[tuple[str, float]] = []
            dl = GitHubDownloader(timing_callback=lambda s, d: timings.append((s, d)))
            skill = _make_skill(github_url="https://github.com/acme/tool")

            zip_bytes = _make_zip({"SKILL.md": "# Skill"})
            with patch.object(
                RateLimitedClient,
                "download",
                new_callable=AsyncMock,
                return_value=zip_bytes,
            ):
                await dl.download_skill(skill, Path(tmpdir))

            step_names = [t[0] for t in timings]
            assert "github download" in step_names
            assert "extract + write" in step_names

    @pytest.mark.asyncio
    async def test_size_check_skipped_for_subdirectory(self):
        """Size check is skipped when downloading a subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = GitHubDownloader(max_repo_size_mb=1.0)
            skill = _make_skill(
                github_url="https://github.com/acme/mono/tree/main/.skills/s"
            )

            zip_bytes = _make_zip({".skills/s/SKILL.md": "# Skill"})
            with patch.object(
                RateLimitedClient,
                "download",
                new_callable=AsyncMock,
                return_value=zip_bytes,
            ):
                success, _, _ = await dl.download_skill(skill, Path(tmpdir))
            assert success


class TestGetRepoSizeMb:
    """Test repo size checking."""

    @pytest.mark.asyncio
    async def test_returns_size(self):
        dl = GitHubDownloader()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"size": 5120}  # 5MB in KB
        with patch.object(
            RateLimitedClient, "get", new_callable=AsyncMock, return_value=mock_resp
        ):
            size = await dl._get_repo_size_mb("acme", "tool")
        assert size == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        dl = GitHubDownloader()
        with patch.object(
            RateLimitedClient,
            "get",
            new_callable=AsyncMock,
            side_effect=RuntimeError("fail"),
        ):
            size = await dl._get_repo_size_mb("acme", "tool")
        assert size is None


class TestContextManager:
    """Test async context manager."""

    @pytest.mark.asyncio
    async def test_log_timing_noop_without_callback(self):
        dl = GitHubDownloader()
        dl._log_timing("test", 1.0)  # Should not raise

    @pytest.mark.asyncio
    async def test_log_timing_with_callback(self):
        calls: list[tuple[str, float]] = []
        dl = GitHubDownloader(timing_callback=lambda s, d: calls.append((s, d)))
        dl._log_timing("step", 0.5)
        assert calls == [("step", 0.5)]
