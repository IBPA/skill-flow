"""CLI commands for skill crawler."""

import asyncio
from typing import Annotated

import typer
from rich.console import Console

from skill_crawler.commands import crawl_async, search_async
from skill_crawler.config import get_settings
from skill_crawler.storage.manager import StorageManager
from skill_crawler.utils.progress import ProgressTracker

app = typer.Typer(
    name="skill-crawler",
    help="Crawl and download agent skills from online marketplaces.",
)

console = Console()
tracker = ProgressTracker(console)


@app.command()
def crawl(
    source: Annotated[
        str | None,
        typer.Option("--source", "-s", help="Source to crawl (skillsmp)"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="List skills without downloading"),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume", "-r", help="Resume from last sync state"),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed timing for each step"),
    ] = False,
) -> None:
    """Crawl and download skills from marketplaces."""
    asyncio.run(crawl_async(source, dry_run, resume, verbose))


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum results"),
    ] = 20,
) -> None:
    """Search for skills on SkillsMP."""
    asyncio.run(search_async(query, limit))


@app.command()
def status() -> None:
    """Show sync status and statistics."""
    settings = get_settings()
    storage = StorageManager(settings.skills_dir)

    state = storage.load_sync_state()
    stats = storage.get_stats()

    console.print("\n[bold]Storage Statistics:[/bold]")
    for key, value in stats.items():
        console.print(f"  {key}: {value}")

    console.print("\n[bold]Sync State:[/bold]")
    for source, source_state in state.sources.items():
        console.print(f"\n  [cyan]{source}:[/cyan]")
        console.print(f"    Status: {source_state.status.value}")
        console.print(f"    Total found: {source_state.total_found}")
        console.print(f"    Synced: {source_state.synced_count}")
        console.print(f"    Skipped: {source_state.skipped_count}")
        console.print(f"    Failed: {source_state.failed_count}")
        if source_state.last_sync:
            console.print(f"    Last sync: {source_state.last_sync}")
        if source_state.error_message:
            console.print(f"    Error: {source_state.error_message}")


if __name__ == "__main__":
    app()
