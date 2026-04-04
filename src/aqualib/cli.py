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
from typing import Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="aqualib",
    help="AquaLib – Multi-agent framework with vendor skill priority and RAG retrieval.",
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
    skip_rag: bool = typer.Option(False, "--skip-rag", help="(Legacy) Skip RAG index build."),
) -> None:
    """Run a task using the Copilot SDK agent pipeline."""
    settings = _get_settings(base_dir, verbose)

    from aqualib.workspace.manager import WorkspaceManager

    ws = WorkspaceManager(settings)

    # Project awareness
    project = ws.load_project()
    if project is None:
        rprint("[yellow]⚠️ No project found. Run 'aqualib init' first to set up your workspace.[/yellow]")
    else:
        task_count = project.get("task_count", 0)
        session_id = project.get("session_id", "")
        session_label = f"session: {session_id[:20]}…" if session_id else "new session"
        rprint(
            f"[cyan]📂 Project: {project.get('name', 'unknown')} "
            f"({task_count} previous tasks, {session_label})[/cyan]"
        )

    async def _run() -> list[str]:
        from aqualib.sdk.client import AquaLibClient
        from aqualib.sdk.session_manager import SessionManager

        aqua_client = AquaLibClient(settings)
        client = await aqua_client.start()

        try:
            sm = SessionManager(client, settings, ws)
            session = await sm.get_or_create_session()

            done = asyncio.Event()
            result_messages: list[str] = []

            def on_event(event: Any) -> None:
                event_type = getattr(event, "type", None)
                type_val = event_type.value if hasattr(event_type, "value") else str(event_type)
                data = getattr(event, "data", {})

                if type_val == "assistant.message":
                    content = getattr(data, "content", "") or ""
                    result_messages.append(content)
                    if content:
                        rprint(f"[green]{content}[/green]")
                elif type_val == "subagent.started":
                    name = getattr(data, "agent_display_name", "agent")
                    rprint(f"  [dim]▶ {name} started[/dim]")
                elif type_val == "subagent.completed":
                    name = getattr(data, "agent_display_name", "agent")
                    rprint(f"  [dim]✅ {name} completed[/dim]")
                elif type_val == "session.idle":
                    done.set()

            session.on(on_event)
            await session.send(query)
            await done.wait()

            ws.update_project_after_task(query, result_messages)
            return result_messages

        finally:
            await aqua_client.stop()

    try:
        results = asyncio.run(_run())
        rprint(Panel(
            f"[bold]Query:[/bold] {query[:120]}\n"
            f"[bold]Status:[/bold] [green]completed[/green]\n"
            f"[bold]Messages:[/bold] {len(results)} response(s)",
            title="🐙 AquaLib Result",
        ))
        rprint(f"\n📁 Results: {settings.directories.results}")
    except ImportError as exc:
        rprint(f"[red]❌ {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def skills(
    base_dir: str | None = typer.Option(None, "--base-dir", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List all registered skills (vendor skills shown first)."""
    settings = _get_settings(base_dir, verbose)
    from aqualib.bootstrap import build_registry

    registry = build_registry(settings)

    table = Table(title="Registered Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Description")
    table.add_column("Tags", style="dim")

    for skill in registry.list_vendor() + registry.list_generic():
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
    name: str | None = typer.Option(None, "--name", "-n", help="Project name (defaults to directory name)."),
    description: str = typer.Option("", "--description", help="Project description."),
) -> None:
    """Initialise the workspace directory structure and default config."""
    settings = _get_settings(base_dir, verbose)
    from aqualib.workspace.manager import WorkspaceManager

    ws = WorkspaceManager(settings)

    # Detect existing vs. new project
    existing = ws.load_project()
    if existing:
        created = existing.get("created_at", "unknown")[:10]
        task_count = existing.get("task_count", 0)
        rprint(
            f"[cyan]📂 Existing project found: {existing['name']} "
            f"(created {created}, {task_count} tasks). Workspace is ready.[/cyan]"
        )
    else:
        meta = ws.create_project(name=name, description=description)
        rprint(f"[green]🆕 New project initialised: {meta['name']}[/green]")

    # Write a starter config file if it doesn't exist
    cfg_path = Path("aqualib.yaml")
    if not cfg_path.exists():
        cfg_template = (
            f"llm:\n"
            f"  api_key: \"\"              # defaults to OPENAI_API_KEY env var\n"
            f"  base_url: null           # set for Azure, DeepSeek, Ollama, etc."
            f" Also reads AQUALIB_LLM_BASE_URL / OPENAI_BASE_URL\n"
            f"  model: {settings.llm.model}\n"
            f"  temperature: {settings.llm.temperature}\n"
            f"  max_tokens: {settings.llm.max_tokens}\n"
            f"\n"
            f"rag:\n"
            f"  api_key: \"\"              # defaults to AQUALIB_RAG_API_KEY env var,"
            f" then falls back to llm.api_key\n"
            f"  base_url: null           # defaults to AQUALIB_RAG_BASE_URL env var,"
            f" then falls back to llm.base_url\n"
            f"  chunk_size: {settings.rag.chunk_size}\n"
            f"  chunk_overlap: {settings.rag.chunk_overlap}\n"
            f"  similarity_top_k: {settings.rag.similarity_top_k}\n"
            f"  embed_model: {settings.rag.embed_model}\n"
            f"\n"
            f"vendor_priority: {str(settings.vendor_priority).lower()}\n"
            f"\n"
            f"directories:\n"
            f"  base: ./aqualib_workspace\n"
        )
        cfg_path.write_text(cfg_template)
        rprint(f"[green]✅ Config written → {cfg_path}[/green]")

    rprint(f"[green]✅ Workspace initialised at {settings.directories.base}[/green]")
    rprint(f"   work/              → {settings.directories.work}")
    rprint(f"   results/           → {settings.directories.results}")
    rprint(f"   data/              → {settings.directories.data}")
    rprint(f"   skills/vendor/     → {settings.directories.skills_vendor}")
    rprint(f"   vendor_traces/     → {settings.directories.vendor_traces}")
    rprint()
    rprint(
        "[dim]Drop your vendor skill libraries into [bold]skills/vendor/[/bold] – "
        "they will be auto-discovered at runtime.[/dim]"
    )


@app.command()
def status(
    base_dir: str | None = typer.Option(None, "--base-dir", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of recent tasks to show."),
) -> None:
    """Show the project context at a glance."""
    settings = _get_settings(base_dir, verbose)
    from collections import Counter

    from aqualib.workspace.manager import WorkspaceManager

    ws = WorkspaceManager(settings)
    meta = ws.load_project()
    if meta is None:
        rprint("[yellow]⚠️ No project found. Run 'aqualib init' first to set up your workspace.[/yellow]")
        raise typer.Exit(1)

    entries = ws.load_context_log()
    status_counts: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    for entry in entries:
        status_counts[entry.get("status", "unknown")] += 1
        for skill in entry.get("skills_used", []):
            skill_counts[skill] += 1

    # Task status summary
    task_count = meta.get("task_count", 0)
    status_parts = [f"{count} {s}" for s, count in status_counts.most_common()]
    tasks_detail = f" ({', '.join(status_parts)})" if status_parts else ""

    # Data files
    data_dir = settings.directories.data
    data_files = [f.name for f in data_dir.iterdir() if f.is_file()] if data_dir.exists() else []
    data_summary = (
        f"{len(data_files)} files in data/ ({', '.join(data_files[:5])})"
        if data_files
        else "No files in data/"
    )

    # Skills summary
    skill_parts = [f"{name} ({count}×)" for name, count in skill_counts.most_common()]
    skills_summary = ", ".join(skill_parts) if skill_parts else "none"

    rprint()
    rprint(f"[bold cyan]📂 Project:[/bold cyan] {meta.get('name', 'unknown')}")
    rprint(f"   [bold]Created:[/bold]  {meta.get('created_at', 'unknown')[:10]}")
    rprint(f"   [bold]Updated:[/bold]  {meta.get('updated_at', 'unknown')[:10]}")
    rprint(f"   [bold]Tasks:[/bold]    {task_count}{tasks_detail}")
    rprint(f"   [bold]Data:[/bold]     {data_summary}")
    rprint(f"   [bold]Skills:[/bold]   {skills_summary}")

    if entries:
        rprint()
        rprint("[bold]Recent tasks:[/bold]")
        for entry in entries[-limit:]:
            tid = entry.get("task_id", "?")
            query = entry.get("query", "")[:50]
            entry_status = entry.get("status", "unknown")
            icon = "✅" if entry_status == "approved" else "⚠️"
            rprint(f'  • [{tid}] "{query}" {icon} {entry_status}')
    rprint()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
