"""Tests for evaluation agents skills injector."""

import asyncio
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.agents.skills.injector import TarGzSkillInjector


class TestTarGzSkillInjectorInit:
    """Tests for TarGzSkillInjector initialization."""

    def test_init_with_default_logger(self) -> None:
        """Test initialization with default logger."""
        injector = TarGzSkillInjector()
        assert injector._logger is not None

    def test_init_with_custom_logger(self) -> None:
        """Test initialization with custom logger."""
        logger = MagicMock()
        injector = TarGzSkillInjector(logger=logger)
        assert injector._logger is logger


class TestTarGzSkillInjectorCleanupLocalSkills:
    """Tests for TarGzSkillInjector._cleanup_local_skills."""

    def test_cleanup_existing_directory(self, tmp_path: Path) -> None:
        """Test cleaning up existing skills directory."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        skills_dir = logs_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "test.txt").write_text("test")

        injector = TarGzSkillInjector()
        injector._cleanup_local_skills(logs_dir)

        assert not skills_dir.exists()

    def test_cleanup_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test cleanup when skills directory doesn't exist."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        injector = TarGzSkillInjector()
        injector._cleanup_local_skills(logs_dir)

    def test_cleanup_permission_error(self, tmp_path: Path) -> None:
        """Test cleanup logs warning on permission error."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        skills_dir = logs_dir / "skills"
        skills_dir.mkdir()

        logger = MagicMock()
        injector = TarGzSkillInjector(logger=logger)

        with patch.object(shutil, "rmtree", side_effect=PermissionError):
            injector._cleanup_local_skills(logs_dir)

        logger.warning.assert_called_once()


class TestTarGzSkillInjectorStageSkill:
    """Tests for TarGzSkillInjector._stage_skill."""

    def test_stage_skill_copies_to_staging(self, tmp_path: Path) -> None:
        """Test that skill is copied to staging directory."""
        skill_folder = tmp_path / "source" / "my-skill"
        skill_folder.mkdir(parents=True)
        (skill_folder / "SKILL.md").write_text("# My Skill")

        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        injector = TarGzSkillInjector()
        injector._stage_skill(skill_folder, staging_dir, logs_dir)

        staged = staging_dir / "my-skill"
        assert staged.exists()
        assert (staged / "SKILL.md").read_text() == "# My Skill"

    def test_stage_skill_copies_to_logs(self, tmp_path: Path) -> None:
        """Test that skill is copied to logs directory."""
        skill_folder = tmp_path / "source" / "my-skill"
        skill_folder.mkdir(parents=True)
        (skill_folder / "SKILL.md").write_text("# My Skill")

        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        injector = TarGzSkillInjector()
        injector._stage_skill(skill_folder, staging_dir, logs_dir)

        logged = logs_dir / "skills" / "my-skill"
        assert logged.exists()
        assert (logged / "SKILL.md").read_text() == "# My Skill"

    def test_stage_skill_permission_error_on_logs(self, tmp_path: Path) -> None:
        """Test that permission error on logs doesn't crash."""
        skill_folder = tmp_path / "source" / "my-skill"
        skill_folder.mkdir(parents=True)
        (skill_folder / "SKILL.md").write_text("# My Skill")

        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        logger = MagicMock()
        injector = TarGzSkillInjector(logger=logger)

        original_copytree = shutil.copytree
        call_count = 0

        def mock_copytree(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise PermissionError("No permission")
            return original_copytree(*args, **kwargs)

        with patch.object(shutil, "copytree", side_effect=mock_copytree):
            injector._stage_skill(skill_folder, staging_dir, logs_dir)

        logger.warning.assert_called_once()


class TestTarGzSkillInjectorInject:
    """Tests for TarGzSkillInjector.inject."""

    def test_inject_empty_skill_list(self, tmp_path: Path) -> None:
        """Test injection with empty skill list returns 0."""
        environment = MagicMock()
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        injector = TarGzSkillInjector()
        count = asyncio.run(injector.inject(environment, [], logs_dir))

        assert count == 0
        environment.exec.assert_not_called()

    def test_inject_single_skill(self, tmp_path: Path) -> None:
        """Test injecting a single skill."""
        skill_folder = tmp_path / "skills" / "my-skill"
        skill_folder.mkdir(parents=True)
        (skill_folder / "SKILL.md").write_text("# My Skill")

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        environment = AsyncMock()
        environment.exec.return_value = MagicMock(stdout="skill files", return_code=0)
        environment.upload_file = AsyncMock()

        injector = TarGzSkillInjector()
        count = asyncio.run(injector.inject(environment, [skill_folder], logs_dir))

        assert count == 1
        assert environment.exec.call_count >= 2
        environment.upload_file.assert_called_once()

    def test_inject_multiple_skills(self, tmp_path: Path) -> None:
        """Test injecting multiple skills."""
        skills_dir = tmp_path / "skills"

        skill1 = skills_dir / "skill1"
        skill1.mkdir(parents=True)
        (skill1 / "SKILL.md").write_text("# Skill 1")

        skill2 = skills_dir / "skill2"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("# Skill 2")

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        environment = AsyncMock()
        environment.exec.return_value = MagicMock(stdout="files", return_code=0)
        environment.upload_file = AsyncMock()

        injector = TarGzSkillInjector()
        count = asyncio.run(injector.inject(environment, [skill1, skill2], logs_dir))

        assert count == 2


class TestTarGzSkillInjectorUploadArchive:
    """Tests for TarGzSkillInjector._upload_archive."""

    def test_upload_archive_creates_tar(self, tmp_path: Path) -> None:
        """Test that archive is created and uploaded."""
        staging = tmp_path / "staging"
        skills_dir = staging / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# Test")

        environment = AsyncMock()
        environment.exec.return_value = MagicMock(return_code=0)
        environment.upload_file = AsyncMock()

        injector = TarGzSkillInjector()
        asyncio.run(injector._upload_archive(environment, staging))

        environment.upload_file.assert_called_once()
        call_args = environment.upload_file.call_args
        assert str(call_args.kwargs["target_path"]).endswith(".tar.gz")

    def test_upload_archive_logs_error_on_failure(self, tmp_path: Path) -> None:
        """Test that extraction failure is logged."""
        staging = tmp_path / "staging"
        skills_dir = staging / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# Test")

        environment = AsyncMock()
        environment.exec.return_value = MagicMock(return_code=1, stderr="error")
        environment.upload_file = AsyncMock()

        logger = MagicMock()
        injector = TarGzSkillInjector(logger=logger)
        asyncio.run(injector._upload_archive(environment, staging))

        logger.error.assert_called_once()
