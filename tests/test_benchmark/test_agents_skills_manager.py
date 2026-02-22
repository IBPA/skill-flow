"""Tests for evaluation agents skills manager."""

from pathlib import Path

from benchmark.agents.skills.manager import (
    SkillManager,
    extract_task_name_from_trial_dir,
)


class TestExtractTaskNameFromTrialDir:
    """Tests for extract_task_name_from_trial_dir."""

    def test_extract_simple_name(self) -> None:
        """Test extracting task name with separator."""
        result = extract_task_name_from_trial_dir("configure-git__abc123")
        assert result == "configure-git"

    def test_extract_name_with_multiple_separators(self) -> None:
        """Test extracting task name when task name contains double underscore."""
        result = extract_task_name_from_trial_dir("task__with__underscores__xyz789")
        assert result == "task__with__underscores"

    def test_no_separator_returns_none(self) -> None:
        """Test that missing separator returns None."""
        result = extract_task_name_from_trial_dir("no-separator-here")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Test empty string returns None."""
        result = extract_task_name_from_trial_dir("")
        assert result is None


class TestSkillManagerInit:
    """Tests for SkillManager initialization."""

    def test_init_with_defaults(self, tmp_path: Path) -> None:
        """Test initialization with default values."""
        manager = SkillManager(source_dir=tmp_path)
        assert manager._source_dir == tmp_path
        assert manager._skills_list_file is None
        assert manager._match_skill_to_task is False

    def test_init_with_skills_list_file(self, tmp_path: Path) -> None:
        """Test initialization with skills list file."""
        skills_file = tmp_path / "skills.txt"
        manager = SkillManager(source_dir=tmp_path, skills_list_file=skills_file)
        assert manager._skills_list_file == skills_file

    def test_init_with_match_skill_to_task(self, tmp_path: Path) -> None:
        """Test initialization with match_skill_to_task enabled."""
        manager = SkillManager(source_dir=tmp_path, match_skill_to_task=True)
        assert manager._match_skill_to_task is True


class TestSkillManagerFindSkillFolders:
    """Tests for SkillManager.find_skill_folders."""

    def test_find_flat_structure(self, tmp_path: Path) -> None:
        """Test finding skills in flat directory structure."""
        skill1 = tmp_path / "skill1"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("# Skill 1")

        skill2 = tmp_path / "skill2"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("# Skill 2")

        manager = SkillManager(source_dir=tmp_path)
        folders = manager.find_skill_folders()

        assert len(folders) == 2
        assert skill1 in folders
        assert skill2 in folders

    def test_find_nested_structure(self, tmp_path: Path) -> None:
        """Test finding skills in nested directory structure."""
        repo_dir = tmp_path / "repo1"
        repo_dir.mkdir()

        skill1 = repo_dir / "skill1"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("# Skill 1")

        manager = SkillManager(source_dir=tmp_path)
        folders = manager.find_skill_folders()

        assert len(folders) == 1
        assert skill1 in folders

    def test_skip_hidden_directories(self, tmp_path: Path) -> None:
        """Test that hidden directories are skipped."""
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "SKILL.md").write_text("# Hidden")

        visible = tmp_path / "visible"
        visible.mkdir()
        (visible / "SKILL.md").write_text("# Visible")

        manager = SkillManager(source_dir=tmp_path)
        folders = manager.find_skill_folders()

        assert len(folders) == 1
        assert visible in folders

    def test_skip_pycache(self, tmp_path: Path) -> None:
        """Test that __pycache__ directories are skipped."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()

        skill = tmp_path / "skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text("# Skill")

        manager = SkillManager(source_dir=tmp_path)
        folders = manager.find_skill_folders()

        assert len(folders) == 1
        assert skill in folders

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test with empty source directory."""
        manager = SkillManager(source_dir=tmp_path)
        folders = manager.find_skill_folders()
        assert folders == []

    def test_directory_without_skill_md(self, tmp_path: Path) -> None:
        """Test that directories without SKILL.md are not included."""
        no_skill = tmp_path / "no-skill"
        no_skill.mkdir()
        (no_skill / "README.md").write_text("# Not a skill")

        manager = SkillManager(source_dir=tmp_path)
        folders = manager.find_skill_folders()
        assert folders == []


class TestSkillManagerLoadSkillNamesFromFile:
    """Tests for SkillManager.load_skill_names_from_file."""

    def test_load_simple_file(self, tmp_path: Path) -> None:
        """Test loading skill names from simple file."""
        skills_file = tmp_path / "skills.txt"
        skills_file.write_text("skill1\nskill2\nskill3\n")

        manager = SkillManager(source_dir=tmp_path, skills_list_file=skills_file)
        names = manager.load_skill_names_from_file()

        assert names == {"skill1", "skill2", "skill3"}

    def test_skip_comments(self, tmp_path: Path) -> None:
        """Test that comments are skipped."""
        skills_file = tmp_path / "skills.txt"
        skills_file.write_text("skill1\n# comment\nskill2\n")

        manager = SkillManager(source_dir=tmp_path, skills_list_file=skills_file)
        names = manager.load_skill_names_from_file()

        assert names == {"skill1", "skill2"}

    def test_skip_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are skipped."""
        skills_file = tmp_path / "skills.txt"
        skills_file.write_text("skill1\n\n\nskill2\n")

        manager = SkillManager(source_dir=tmp_path, skills_list_file=skills_file)
        names = manager.load_skill_names_from_file()

        assert names == {"skill1", "skill2"}

    def test_no_skills_list_file(self, tmp_path: Path) -> None:
        """Test returns empty set when no skills list file."""
        manager = SkillManager(source_dir=tmp_path)
        names = manager.load_skill_names_from_file()
        assert names == set()

    def test_missing_skills_list_file(self, tmp_path: Path) -> None:
        """Test returns empty set when skills list file doesn't exist."""
        missing_file = tmp_path / "missing.txt"
        manager = SkillManager(source_dir=tmp_path, skills_list_file=missing_file)
        names = manager.load_skill_names_from_file()
        assert names == set()


class TestSkillManagerFilterSkills:
    """Tests for SkillManager.filter_skills."""

    def test_filter_by_skills_list(self, tmp_path: Path) -> None:
        """Test filtering skills by skills list file."""
        skills_file = tmp_path / "skills.txt"
        skills_file.write_text("skill1\nskill3\n")

        skill1 = tmp_path / "skill1"
        skill2 = tmp_path / "skill2"
        skill3 = tmp_path / "skill3"

        manager = SkillManager(source_dir=tmp_path, skills_list_file=skills_file)
        filtered = manager.filter_skills([skill1, skill2, skill3])

        assert len(filtered) == 2
        assert skill1 in filtered
        assert skill3 in filtered
        assert skill2 not in filtered

    def test_filter_match_skill_to_task(self, tmp_path: Path) -> None:
        """Test filtering to matching task skill."""
        skill1 = tmp_path / "task-one"
        skill2 = tmp_path / "task-two"

        manager = SkillManager(source_dir=tmp_path, match_skill_to_task=True)
        filtered = manager.filter_skills([skill1, skill2], task_name="task-one")

        assert len(filtered) == 1
        assert skill1 in filtered

    def test_filter_match_no_matching_skill(self, tmp_path: Path) -> None:
        """Test filtering when no skill matches task."""
        skill1 = tmp_path / "skill1"
        skill2 = tmp_path / "skill2"

        manager = SkillManager(source_dir=tmp_path, match_skill_to_task=True)
        filtered = manager.filter_skills([skill1, skill2], task_name="other-task")

        assert filtered == []

    def test_filter_match_no_task_name(self, tmp_path: Path) -> None:
        """Test filtering when task name is not provided but matching enabled."""
        skill1 = tmp_path / "skill1"

        manager = SkillManager(source_dir=tmp_path, match_skill_to_task=True)
        filtered = manager.filter_skills([skill1], task_name=None)

        assert filtered == []

    def test_filter_no_filtering(self, tmp_path: Path) -> None:
        """Test that all skills pass through when no filtering configured."""
        skill1 = tmp_path / "skill1"
        skill2 = tmp_path / "skill2"

        manager = SkillManager(source_dir=tmp_path)
        filtered = manager.filter_skills([skill1, skill2])

        assert len(filtered) == 2


class TestSkillManagerGetSkills:
    """Tests for SkillManager.get_skills."""

    def test_get_skills_from_directory(self, tmp_path: Path) -> None:
        """Test getting skills from directory."""
        skill1 = tmp_path / "skill1"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("# Skill 1")

        skill2 = tmp_path / "skill2"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("# Skill 2")

        manager = SkillManager(source_dir=tmp_path)
        skills = manager.get_skills()

        assert len(skills) == 2

    def test_get_skills_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test getting skills from nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"
        manager = SkillManager(source_dir=nonexistent)
        skills = manager.get_skills()

        assert skills == []

    def test_get_skills_empty_directory(self, tmp_path: Path) -> None:
        """Test getting skills from empty directory returns empty list."""
        manager = SkillManager(source_dir=tmp_path)
        skills = manager.get_skills()
        assert skills == []

    def test_get_skills_with_task_matching(self, tmp_path: Path) -> None:
        """Test getting skills with task matching enabled."""
        skill1 = tmp_path / "my-task"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text("# My Task Skill")

        skill2 = tmp_path / "other"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text("# Other Skill")

        manager = SkillManager(source_dir=tmp_path, match_skill_to_task=True)
        skills = manager.get_skills(task_name="my-task")

        assert len(skills) == 1
        assert skills[0].name == "my-task"
