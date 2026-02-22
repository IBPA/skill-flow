"""
Skill injection logic for uploading skills to evaluation containers.
"""

import logging
import shutil
import tempfile
from pathlib import Path

from harbor.environments.base import BaseEnvironment


class TarGzSkillInjector:
    """
    Injects skills into containers using tar.gz archive upload.

    Uses tar instead of zip because tar is universally available on Linux
    containers while unzip may not be installed.
    """

    CONTAINER_SKILLS_DIR = "/logs/agent/skills"
    CONTAINER_TMP_ARCHIVE = "/tmp/skills.tar.gz"  # nosec B108

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """
        Initialize skill injector.

        Args:
            logger: Optional logger instance.
        """
        self._logger = logger or logging.getLogger(__name__)

    async def inject(
        self,
        environment: BaseEnvironment,
        skill_folders: list[Path],
        logs_dir: Path,
    ) -> int:
        """
        Upload skills to container via tar.gz archive.

        Args:
            environment: The execution environment.
            skill_folders: List of skill folder paths to upload.
            logs_dir: Local logs directory for copying skills.

        Returns:
            Number of skills injected.
        """
        if not skill_folders:
            return 0

        self._logger.debug(f"Injecting {len(skill_folders)} skills")

        # Create skills directory in container
        await environment.exec(command=f"mkdir -p {self.CONTAINER_SKILLS_DIR}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            skills_staging = temp_path / "skills"
            skills_staging.mkdir()

            # Clean up existing local skills directory
            self._cleanup_local_skills(logs_dir)

            # Stage skills for archive
            for folder in skill_folders:
                self._stage_skill(folder, skills_staging, logs_dir)

            # Create and upload archive
            await self._upload_archive(environment, temp_path)

        # Verify upload
        result = await environment.exec(command=f"ls -la {self.CONTAINER_SKILLS_DIR}/")
        self._logger.debug(f"Skills directory contents:\n{result.stdout}")

        return len(skill_folders)

    def _cleanup_local_skills(self, logs_dir: Path) -> None:
        """Remove existing skills directory in logs."""
        local_skills_dir = logs_dir / "skills"
        if local_skills_dir.exists():
            try:
                shutil.rmtree(local_skills_dir)
            except PermissionError:
                self._logger.warning(
                    f"Could not remove existing skills dir: {local_skills_dir}"
                )

    def _stage_skill(
        self, skill_folder: Path, staging_dir: Path, logs_dir: Path
    ) -> None:
        """
        Stage a skill for archive creation.

        Args:
            skill_folder: Source skill folder.
            staging_dir: Temporary staging directory.
            logs_dir: Local logs directory for copying.
        """
        skill_name = skill_folder.name
        self._logger.debug(f"Staging skill: {skill_name}")

        # Copy to staging
        shutil.copytree(skill_folder, staging_dir / skill_name)

        # Copy to logs directory for visibility in outputs
        skills_log_dir = logs_dir / "skills" / skill_name
        try:
            skills_log_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(skill_folder, skills_log_dir, dirs_exist_ok=True)
        except PermissionError:
            self._logger.warning(f"Could not copy skill to logs dir: {skill_name}")

    async def _upload_archive(
        self, environment: BaseEnvironment, temp_path: Path
    ) -> None:
        """
        Create tar.gz archive and upload to container.

        Args:
            environment: The execution environment.
            temp_path: Temporary directory containing staged skills.
        """
        tar_path = temp_path / "skills.tar.gz"
        shutil.make_archive(
            str(tar_path.with_suffix("").with_suffix("")),
            "gztar",
            root_dir=temp_path,
            base_dir="skills",
        )

        self._logger.debug(
            f"Created skills archive: {tar_path} "
            f"({tar_path.stat().st_size / 1024:.1f} KB)"
        )

        # Upload archive
        await environment.upload_file(
            source_path=tar_path,
            target_path=self.CONTAINER_TMP_ARCHIVE,
        )

        # Extract and cleanup in container
        extract_cmd = (
            f"tar -xzf {self.CONTAINER_TMP_ARCHIVE} -C /logs/agent && "
            f"rm {self.CONTAINER_TMP_ARCHIVE}"
        )
        result = await environment.exec(command=extract_cmd)
        if result.return_code != 0:
            self._logger.error(f"Failed to extract skills: {result.stderr}")
