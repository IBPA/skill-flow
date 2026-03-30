"""GitHub repository downloader for skill content."""

import hashlib
import io
import re
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import aiofiles

from skill_crawler.models.skill import SkillMetadata
from skill_crawler.utils.http import RateLimitedClient

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager


class GitHubDownloader:
    """Download skill content from GitHub repositories."""

    def __init__(
        self,
        github_token: str | None = None,
        rate_limit: float = 1.0,
        timeout: float = 60.0,
        timing_callback: Callable[[str, float], None] | None = None,
        max_repo_size_mb: float = 0,
    ) -> None:
        """Initialize GitHub downloader.

        Args:
            github_token: GitHub personal access token for higher rate limits.
            rate_limit: Request rate limit in seconds.
            timeout: Download timeout in seconds.
            timing_callback: Optional callback for timing info (step_name, seconds).
            max_repo_size_mb: Maximum repo size in MB (0 = no limit).
        """
        headers = {"Accept": "application/vnd.github+json"}
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"

        self._client = RateLimitedClient(
            rate_limit=rate_limit,
            timeout=timeout,
            headers=headers,
        )
        self._session_cm: AbstractAsyncContextManager[RateLimitedClient] | None = None
        self._timing_callback = timing_callback
        self._max_repo_size_mb = max_repo_size_mb

    def _log_timing(self, step: str, duration: float) -> None:
        """Log timing for a step if callback is set."""
        if self._timing_callback:
            self._timing_callback(step, duration)

    async def _get_repo_size_mb(self, owner: str, repo: str) -> float | None:
        """Get repository size in MB from GitHub API.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            Size in MB, or None if unable to fetch.
        """
        try:
            response = await self._client.get(
                f"https://api.github.com/repos/{owner}/{repo}"
            )
            data = response.json()
            size_kb = int(data.get("size", 0))
            return size_kb / 1024
        except Exception:
            return None

    async def __aenter__(self) -> "GitHubDownloader":
        """Enter async context."""
        self._session_cm = self._client.session()
        await self._session_cm.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""
        if self._session_cm:
            await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
            self._session_cm = None

    async def download_skill(
        self,
        skill: SkillMetadata,
        output_dir: Path,
    ) -> tuple[bool, str | None, str]:
        """Download skill content from GitHub.

        Args:
            skill: Skill to download.
            output_dir: Directory to save skill.

        Returns:
            Tuple of (success, content_hash or error message, folder_name).
        """
        if not skill.github_url:
            return False, "No GitHub URL", ""

        try:
            owner, repo, branch, skill_path = self._parse_github_url(
                str(skill.github_url)
            )
            if not owner or not repo:
                return False, f"Invalid GitHub URL format: {skill.github_url}", ""

            # Check repo size before downloading (if limit is set)
            # Skip size check if downloading a subdirectory (skill_path set)
            # since we only extract that path, not the full repo
            if self._max_repo_size_mb > 0 and not skill_path:
                t_start = time.perf_counter()
                size_mb = await self._get_repo_size_mb(owner, repo)
                self._log_timing("size check", time.perf_counter() - t_start)

                if size_mb is not None:
                    self._log_timing("repo size (MB)", size_mb)
                    if size_mb > self._max_repo_size_mb:
                        limit = self._max_repo_size_mb
                        msg = (
                            f"Skipped: repo size {size_mb:.1f}MB > {limit:.1f}MB limit"
                        )
                        return (False, msg, "")

            # Download repository archive (use branch if available)
            if branch:
                archive_url = (
                    f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"
                )
            else:
                archive_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"

            t_start = time.perf_counter()
            content = await self._client.download(archive_url)
            self._log_timing("github download", time.perf_counter() - t_start)
            self._log_timing("archive size", len(content) / 1024 / 1024)  # MB

            # Use skill name as folder, with counter for conflicts
            folder_name = self._resolve_folder_name(output_dir, skill.name)
            skill_dir = output_dir / folder_name
            skill_dir.mkdir(parents=True, exist_ok=True)

            t_start = time.perf_counter()
            content_hash = await self._extract_archive(
                content, skill_dir, skill_path=skill_path
            )
            self._log_timing("extract + write", time.perf_counter() - t_start)

            return True, content_hash, folder_name

        except Exception as e:
            return False, str(e), ""

    def _resolve_folder_name(self, output_dir: Path, skill_name: str) -> str:
        """Resolve folder name using skill name, with counter for conflicts.

        Args:
            output_dir: Directory containing skill folders.
            skill_name: Base skill name to use.

        Returns:
            Folder name (e.g., "my-skill", "my-skill-1", "my-skill-2").
        """
        if not (output_dir / skill_name).exists():
            return skill_name

        # Find next available counter
        counter = 1
        while (output_dir / f"{skill_name}-{counter}").exists():
            counter += 1

        return f"{skill_name}-{counter}"

    def _parse_github_url(
        self, url: str
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """Parse owner, repo, branch, and path from GitHub URL.

        Supports URLs like:
        - https://github.com/owner/repo
        - https://github.com/owner/repo/tree/branch/path/to/skill

        Args:
            url: GitHub repository URL.

        Returns:
            Tuple of (owner, repo, branch, path) or (None, None, None, None) if invalid.
        """
        # Pattern for URLs with tree/branch/path
        tree_pattern = r"github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)"
        match = re.search(tree_pattern, url)
        if match:
            owner = match.group(1)
            repo = match.group(2).removesuffix(".git")
            branch = match.group(3)
            path = match.group(4).rstrip("/")
            return owner, repo, branch, path

        # Fallback for simple repo URLs
        simple_patterns = [
            r"github\.com/([^/]+)/([^/\?#]+)",
            r"github\.com:([^/]+)/([^/\?#]+)",
        ]

        for pattern in simple_patterns:
            match = re.search(pattern, url)
            if match:
                owner = match.group(1)
                repo = match.group(2).removesuffix(".git")
                return owner, repo, None, None

        return None, None, None, None

    async def _extract_archive(
        self,
        content: bytes,
        output_dir: Path,
        skill_path: str | None = None,
    ) -> str:
        """Extract zip archive and calculate content hash.

        Args:
            content: Zip archive bytes.
            output_dir: Directory to extract to.
            skill_path: Path within the archive to extract (e.g., ".skills/my-skill").
                        If provided, only files under this path are extracted.

        Returns:
            Content hash of SKILL.md if found.
        """
        content_hash = ""

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            # Find the root directory in the archive (GitHub adds a prefix)
            root_dir = ""
            for name in zf.namelist():
                if "/" in name:
                    root_dir = name.split("/")[0]
                    break

            # Build the full prefix to match (root + skill_path)
            if skill_path:
                full_prefix = f"{root_dir}/{skill_path}/"
            else:
                full_prefix = f"{root_dir}/" if root_dir else ""

            # Extract files
            for zip_info in zf.infolist():
                if zip_info.is_dir():
                    continue

                # Check if file is under the target path
                if full_prefix and not zip_info.filename.startswith(full_prefix):
                    continue

                # Remove prefix to get relative path
                rel_path = zip_info.filename[len(full_prefix) :]

                if not rel_path:
                    continue

                # Extract file
                target_path = output_dir / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)

                file_content = zf.read(zip_info.filename)

                # Calculate hash for SKILL.md
                if rel_path.upper() == "SKILL.MD":
                    content_hash = hashlib.sha256(file_content).hexdigest()[:16]

                async with aiofiles.open(target_path, "wb") as f:
                    await f.write(file_content)

        return content_hash
