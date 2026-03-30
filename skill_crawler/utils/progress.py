"""Rich progress output utilities."""

from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


class ProgressTracker:
    """Rich-based progress tracking for crawling operations."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize progress tracker.

        Args:
            console: Rich console instance.
        """
        self.console = console or Console()

    def create_progress(self) -> Progress:
        """Create a progress bar for downloads."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    def print_status(
        self,
        source: str,
        total: int,
        synced: int,
        failed: int,
    ) -> None:
        """Print sync status summary.

        Args:
            source: Source name.
            total: Total skills found.
            synced: Successfully synced count.
            failed: Failed count.
        """
        table = Table(title=f"Sync Status: {source}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Found", str(total))
        table.add_row("Synced", str(synced))
        table.add_row("Failed", str(failed))
        table.add_row("Remaining", str(total - synced - failed))

        self.console.print(table)

    def print_skill_table(self, skills: list[dict[str, Any]]) -> None:
        """Print skills in a table format.

        Args:
            skills: List of skill dictionaries.
        """
        table = Table(title="Skills")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Source", style="green")
        table.add_column("Author", style="yellow")
        table.add_column("Tags")

        for skill in skills:
            table.add_row(
                skill.get("name", "Unknown"),
                skill.get("source", "Unknown"),
                skill.get("author", "-"),
                ", ".join(skill.get("tags", [])),
            )

        self.console.print(table)

    def success(self, message: str) -> None:
        """Print success message."""
        self.console.print(f"[green]:heavy_check_mark:[/green] {message}")

    def error(self, message: str) -> None:
        """Print error message."""
        self.console.print(f"[red]:x:[/red] {message}")

    def warning(self, message: str) -> None:
        """Print warning message."""
        self.console.print(f"[yellow]:warning:[/yellow] {message}")

    def info(self, message: str) -> None:
        """Print info message."""
        self.console.print(f"[blue]:information:[/blue] {message}")
