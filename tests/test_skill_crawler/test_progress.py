"""Tests for Rich progress utilities."""

from io import StringIO
from typing import Any

from rich.console import Console
from rich.progress import Progress
from skill_crawler.utils.progress import ProgressTracker


def _make_tracker() -> tuple[ProgressTracker, StringIO]:
    """Create a tracker with captured output."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120)
    return ProgressTracker(console), buf


class TestProgressTracker:
    """Test ProgressTracker methods."""

    def test_default_console(self):
        """Test creating tracker without explicit console."""
        tracker = ProgressTracker()
        assert tracker.console is not None

    def test_create_progress(self):
        """Test create_progress returns a Progress instance."""
        tracker, _ = _make_tracker()
        progress = tracker.create_progress()
        assert isinstance(progress, Progress)

    def test_success_message(self):
        """Test success prints green message."""
        tracker, buf = _make_tracker()
        tracker.success("Done!")
        output = buf.getvalue()
        assert "Done!" in output

    def test_error_message(self):
        """Test error prints red message."""
        tracker, buf = _make_tracker()
        tracker.error("Failed!")
        output = buf.getvalue()
        assert "Failed!" in output

    def test_warning_message(self):
        """Test warning prints yellow message."""
        tracker, buf = _make_tracker()
        tracker.warning("Careful!")
        output = buf.getvalue()
        assert "Careful!" in output

    def test_info_message(self):
        """Test info prints blue message."""
        tracker, buf = _make_tracker()
        tracker.info("Note")
        output = buf.getvalue()
        assert "Note" in output

    def test_print_status(self):
        """Test print_status renders a table."""
        tracker, buf = _make_tracker()
        tracker.print_status("skillsmp", total=100, synced=80, failed=5)
        output = buf.getvalue()
        assert "skillsmp" in output
        assert "100" in output
        assert "80" in output
        assert "5" in output

    def test_print_skill_table(self):
        """Test print_skill_table renders skills."""
        tracker, buf = _make_tracker()
        skills: list[dict[str, Any]] = [
            {
                "name": "Test Skill",
                "source": "skillsmp",
                "author": "Alice",
                "tags": ["python", "ai"],
            },
            {"name": "Another"},
        ]
        tracker.print_skill_table(skills)
        output = buf.getvalue()
        assert "Test Skill" in output
        assert "Alice" in output
        assert "python" in output

    def test_print_skill_table_empty(self):
        """Test print_skill_table with no skills."""
        tracker, buf = _make_tracker()
        tracker.print_skill_table([])
        output = buf.getvalue()
        assert "Skills" in output
