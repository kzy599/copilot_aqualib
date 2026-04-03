"""AquaLib CLI – the primary user interface.

Usage:
    aqualib run "Align these protein sequences"
    aqualib skills
    aqualib tasks
    aqualib report <task_id>
    aqualib init
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="aqualib",
    help="AquaLib – Multi-agent framework with Clawbio skill priority and RAG retrieval.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_settings(base_dir: str | None, verbose: bool):
    """Load settings with optional overrides."""
    import os

    from aqualib.utils.logging import setup_logging

    if base_dir:
        os.environ["AQUALIB_BASE_DIR"] = base_dir
    from aqualib.config import get_settings, reset_settings

    reset_settings()
    settings = get_settings()
    settings.verbose = verbose
    setup_logging(verbose=verbose)
    return settings


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    query: str = typer.Argument(..., help="The user request / task description."),
    base_dir: str | None = typer.Option(None, "--base-dir", "-d", help="Workspace base directory."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    skip_rag: bool = typer.Option(False, "--skip-rag", help="Skip RAG index build (faster, no context)."),
) -> None:
    """Run the full agent pipeline (Searcher → Executor → Reviewer)."""
    settings = _get_settings(base_dir, verbose)

    async def _run():
        from aqualib.bootstrap import build_orchestrator

        orch = await build_orchestrator(settings, skip_rag_index=skip_rag)
        task = await orch.run(query)
        return task

    task = asyncio.run(_run())

    # Pretty output
    status_colour = "green" if task.review_passed else "yellow"
    rprint(Panel(
        f"[bold]Task:[/bold] {task.task_id}\n"
        f"[bold]Status:[/bold] [{status_colour}]{task.status.value}[/{status_colour}]\n"
        f"[bold]Clawbio Priority:[/bold] {'✅' if task.clawbio_priority_satisfied else '⚠️'}\n"
        f"[bold]Review:[/bold] {task.review_notes[:200] or 'N/A'}",
        title="🐙 AquaLib Result",
    ))

    # Show skill invocations
    if task.skill_invocations:
        table = Table(title="Skill Invocations")
        table.add_column("Skill", style="cyan")
        table.add_column("Source", style="magenta")
        table.add_column("OK?", justify="center")
        table.add_column("Output Dir")
        for inv in task.skill_invocations:
            table.add_row(
                inv.skill_name,
                inv.source.value,
                "✅" if inv.success else "❌",
                inv.output_dir or "N/A",
            )
        console.print(table)

    rprint(f"\n📁 Results: {settings.directories.results / task.task_id}")


@app.command()
def skills(
    base_dir: str | None = typer.Option(None, "--base-dir", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List all registered skills (Clawbio skills shown first)."""
    settings = _get_settings(base_dir, verbose)
    from aqualib.bootstrap import build_registry

    registry = build_registry(settings)

    table = Table(title="Registered Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Description")
    table.add_column("Tags", style="dim")

    for skill in registry.list_clawbio() + registry.list_generic():
        table.add_row(
            skill.meta.name,
            skill.meta.source.value,
            skill.meta.description[:80],
            ", ".join(skill.meta.tags),
        )
    console.print(table)


@app.command()
def tasks(
    base_dir: str | None = typer.Option(None, "--base-dir", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List all completed tasks."""
    settings = _get_settings(base_dir, verbose)
    from aqualib.workspace.manager import WorkspaceManager

    ws = WorkspaceManager(settings)
    task_ids = ws.list_tasks()

    if not task_ids:
        rprint("[dim]No tasks found.[/dim]")
        return

    table = Table(title="Tasks")
    table.add_column("Task ID", style="cyan")
    table.add_column("Status")
    table.add_column("Query")
    for tid in task_ids:
        t = ws.load_task(tid)
        if t:
            table.add_row(tid, t.status.value, t.user_query[:60])
    console.print(table)


@app.command()
def report(
    task_id: str = typer.Argument(..., help="Task ID to display."),
    base_dir: str | None = typer.Option(None, "--base-dir", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown | json"),
) -> None:
    """Display the audit report for a task."""
    settings = _get_settings(base_dir, verbose)
    from aqualib.workspace.manager import WorkspaceManager

    ws = WorkspaceManager(settings)
    audit = ws.load_audit_report(task_id)
    if audit is None:
        rprint(f"[red]No audit report found for task {task_id}.[/red]")
        raise typer.Exit(1)

    if format == "json":
        rprint(audit.model_dump_json(indent=2))
    else:
        rprint(audit.to_markdown())


@app.command()
def init(
    base_dir: str | None = typer.Option(None, "--base-dir", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Initialise the workspace directory structure and default config."""
    settings = _get_settings(base_dir, verbose)
    from aqualib.workspace.manager import WorkspaceManager

    WorkspaceManager(settings)

    # Write a starter config file if it doesn't exist
    cfg_path = Path("aqualib.yaml")
    if not cfg_path.exists():
        import yaml

        cfg_path.write_text(yaml.dump(settings.model_dump(mode="json"), default_flow_style=False, sort_keys=False))
        rprint(f"[green]✅ Config written → {cfg_path}[/green]")

    rprint(f"[green]✅ Workspace initialised at {settings.directories.base}[/green]")
    rprint(f"   work/              → {settings.directories.work}")
    rprint(f"   results/           → {settings.directories.results}")
    rprint(f"   data/              → {settings.directories.data}")
    rprint(f"   skills/clawbio/    → {settings.directories.skills_clawbio}")
    rprint(f"   clawbio_traces/    → {settings.directories.clawbio_traces}")
    rprint()
    rprint(
        "[dim]Drop your Clawbio skill library into [bold]skills/clawbio/[/bold] – "
        "it will be auto-discovered at runtime.[/dim]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
