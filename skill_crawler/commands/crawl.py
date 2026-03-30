"""Crawl command implementations."""

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from skill_crawler.config import Settings, get_settings
from skill_crawler.crawlers.skillsmp.api_client import SkillsMPAPIClient
from skill_crawler.crawlers.skillsmp.scraper import SkillsMPScraper
from skill_crawler.downloaders.github import GitHubDownloader
from skill_crawler.models.skill import SkillMetadata
from skill_crawler.models.state import SourceState, SyncStatus
from skill_crawler.storage.manager import StorageManager
from skill_crawler.utils.progress import ProgressTracker

if TYPE_CHECKING:
    from skill_crawler.crawlers.base import BaseCrawler

console = Console()
tracker = ProgressTracker(console)


def _should_skip_failed(skill_id: str, failed_skills: list[str]) -> bool:
    """Check if skill previously failed with client error."""
    prev_failure = next(
        (f for f in failed_skills if f.startswith(f"{skill_id}:")),
        None,
    )
    return prev_failure is not None and "Client error" in prev_failure


def _handle_download_result(
    skill: SkillMetadata,
    success: bool,
    result: str | None,
    folder_name: str,
    output_dir: Path,
    source_state: SourceState,
    storage: StorageManager,
) -> None:
    """Handle the result of a skill download."""
    if success:
        skill.content_hash = result
        skill.local_path = str(output_dir / folder_name)
        storage.save_skill_metadata(skill)
        source_state.synced_count += 1
        tracker.success(f"Downloaded: {skill.name}")
    elif result and result.startswith("Skipped:"):
        source_state.skipped_count += 1
        source_state.skipped_skills.append(f"{skill.id}: {result}")
        tracker.warning(f"{skill.name} - {result}")
    else:
        source_state.failed_count += 1
        source_state.failed_skills.append(f"{skill.id}: {result}")
        tracker.error(f"Failed: {skill.name} - {result}")


def _create_timing_logger(verbose: bool):
    """Create a timing logger function if verbose mode is enabled."""
    if not verbose:
        return None

    def log_timing(step: str, duration: float) -> None:
        console.print(f"  [dim]⏱ {step}: {duration:.2f}s[/dim]")

    return log_timing


async def crawl_async(
    source: str | None,
    dry_run: bool,
    resume: bool,
    verbose: bool = False,
) -> None:
    """Async implementation of crawl command."""
    timing_callback = _create_timing_logger(verbose)

    settings = get_settings()
    settings.ensure_directories()

    storage = StorageManager(settings.skills_dir)
    storage.ensure_directories()

    sources = [source] if source else ["skillsmp"]

    for src in sources:
        console.print(f"\n[bold blue]Crawling {src}...[/bold blue]")

        if src == "skillsmp":
            await _crawl_skillsmp(
                settings, storage, dry_run, resume, verbose, timing_callback
            )
        else:
            tracker.error(f"Unknown source: {src}")


async def _crawl_skillsmp(
    settings: Settings,
    storage: StorageManager,
    dry_run: bool,
    resume: bool,
    verbose: bool,
    timing_callback,
) -> None:
    """Crawl SkillsMP marketplace."""
    state = storage.load_sync_state()
    source_state = state.get_source_state("skillsmp")

    # Start cursor from state if resuming
    cursor = source_state.cursor if resume else None

    # Try API first, fall back to scraper
    crawler: BaseCrawler
    downloader: GitHubDownloader

    try:
        if settings.skillsmp_api_token:
            tracker.info("Using SkillsMP API client")
            crawler = SkillsMPAPIClient(
                api_token=settings.skillsmp_api_token,
                rate_limit=settings.rate_limit_delay,
            )
        else:
            tracker.info("Using SkillsMP web scraper (no API token)")
            crawler = SkillsMPScraper(rate_limit=settings.rate_limit_delay)

        downloader = GitHubDownloader(
            github_token=settings.github_token,
            rate_limit=settings.rate_limit_delay,
            timing_callback=timing_callback,
            max_repo_size_mb=settings.max_repo_size_mb,
        )

        async with crawler, downloader:
            source_state.status = SyncStatus.IN_PROGRESS
            storage.save_sync_state(state)

            with tracker.create_progress() as progress:
                task = progress.add_task("Crawling SkillsMP", total=None)

                async for skill in crawler.iter_all_skills(cursor=cursor):
                    source_state.total_found += 1
                    progress.update(task, advance=1)

                    if dry_run:
                        console.print(
                            f"  [cyan]{skill.name}[/cyan] - {skill.github_url}"
                        )
                        continue

                    # Skip if already downloaded or previously failed with client error
                    if storage.skill_exists("skillsmp", skill.id):
                        source_state.synced_count += 1
                        continue
                    if _should_skip_failed(skill.id, source_state.failed_skills):
                        continue

                    if verbose:
                        console.print(
                            f"\n[cyan]{skill.name}[/cyan] ({skill.github_url})"
                        )

                    success, result, folder_name = await downloader.download_skill(
                        skill, settings.skillsmp_dir
                    )
                    _handle_download_result(
                        skill,
                        success,
                        result,
                        folder_name,
                        settings.skillsmp_dir,
                        source_state,
                        storage,
                    )
                    storage.save_sync_state(state)

            source_state.status = SyncStatus.COMPLETED
            storage.save_sync_state(state)

    except Exception as e:
        source_state.status = SyncStatus.FAILED
        source_state.error_message = str(e)
        storage.save_sync_state(state)
        tracker.error(f"Crawl failed: {e}")


async def search_async(query: str, limit: int) -> None:
    """Async implementation of search command."""
    settings = get_settings()

    crawler: BaseCrawler
    if settings.skillsmp_api_token:
        crawler = SkillsMPAPIClient(api_token=settings.skillsmp_api_token)
    else:
        crawler = SkillsMPScraper()

    async with crawler:
        results = await crawler.search(query, limit=limit)

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        console.print(f"\n[bold]Found {len(results)} skills:[/bold]\n")

        for skill in results:
            console.print(f"[cyan]{skill.name}[/cyan]")
            if skill.description:
                console.print(f"  {skill.description[:100]}...")
            if skill.github_url:
                console.print(f"  [dim]{skill.github_url}[/dim]")
            console.print()
