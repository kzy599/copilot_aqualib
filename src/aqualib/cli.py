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
    session: str | None = typer.Option(
        None, "--session", "-s",
        help="Resume a specific session by slug or prefix. Default: most recent active session.",
    ),
    new_session: bool = typer.Option(
        False, "--new-session",
        help="Force creation of a new session instead of resuming.",
    ),
    session_name: str | None = typer.Option(
        None, "--session-name",
        help="Name for the new session (only used with --new-session).",
    ),
    skip_rag: bool = typer.Option(False, "--skip-rag", help="(Legacy) Skip RAG index build."),
) -> None:
    """Run a task using the Copilot SDK agent pipeline."""
    settings = _get_settings(base_dir, verbose)

    from aqualib.workspace.manager import WorkspaceManager

    ws = WorkspaceManager(settings)

    # Ensure project exists
    project = ws.load_project()
    if project is None:
        rprint("[yellow]⚠️ No project found. Run 'aqualib init' first to set up your workspace.[/yellow]")

    # Determine session slug
    target_slug: str | None = None
    if new_session:
        meta = ws.create_session(name=session_name)
        target_slug = meta["slug"]
        rprint(f"[green]🆕 New session: {meta['name']} ({target_slug})[/green]")
    elif session:
        found = ws.find_session_by_prefix(session)
        if found:
            target_slug = found["slug"]
            rprint(
                f"[cyan]📂 Resuming session: {found['name']} "
                f"({target_slug}, {found['task_count']} tasks)[/cyan]"
            )
        else:
            rprint(f"[red]Session '{session}' not found.[/red]")
            raise typer.Exit(1)
    else:
        active = ws.get_active_session()
        if active:
            target_slug = active["slug"]
            rprint(
                f"[cyan]📂 Session: {active['name']} "
                f"({target_slug}, {active['task_count']} tasks)[/cyan]"
            )

    async def _run() -> list[str]:
        from aqualib.sdk.client import AquaLibClient
        from aqualib.sdk.session_manager import SessionManager

        aqua_client = AquaLibClient(settings)
        client = await aqua_client.start()

        try:
            sm = SessionManager(client, settings, ws, session_slug=target_slug)
            sdk_session, actual_slug = await sm.get_or_create_session()

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
                    # Write reviewer memory when reviewer completes
                    agent_name = getattr(data, "agent_name", "")
                    if agent_name == "reviewer":
                        content = getattr(data, "content", "") or ""
                        ws.append_agent_memory_entry(actual_slug, "reviewer", {
                            "query": query,
                            "verdict": _extract_verdict(content),
                            "violations": _extract_violations(content),
                            "suggestions": _extract_suggestions(content),
                        })
                elif type_val == "session.idle":
                    done.set()

            sdk_session.on(on_event)
            await sdk_session.send(query)
            await done.wait()

            ws.update_session_after_task(actual_slug, query, result_messages)
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


def _extract_verdict(content: str) -> str:
    """Extract VERDICT from reviewer output."""
    for line in content.splitlines():
        if "VERDICT:" in line:
            if "approved" in line.lower():
                return "approved"
            if "needs_revision" in line.lower():
                return "needs_revision"
    return "unknown"


def _extract_violations(content: str) -> list[str]:
    """Extract VENDOR_PRIORITY violations from reviewer output."""
    for line in content.splitlines():
        if "VENDOR_PRIORITY:" in line and "violated" in line.lower():
            # Extract reason after the dash
            parts = line.split("-", 1)
            if len(parts) > 1:
                return [parts[1].strip()]
    return []


def _extract_suggestions(content: str) -> list[str]:
    """Extract SUGGESTIONS from reviewer output (up to 3)."""
    suggestions: list[str] = []
    in_suggestions = False
    for line in content.splitlines():
        if "SUGGESTIONS:" in line:
            in_suggestions = True
            continue
        if in_suggestions and line.strip().startswith("-"):
            suggestions.append(line.strip().lstrip("- "))
            if len(suggestions) >= 3:
                break
        elif in_suggestions and line.strip() and not line.strip().startswith("-"):
            break
    return suggestions


@app.command()
def skills(
    base_dir: str | None = typer.Option(None, "--base-dir", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List all registered skills (vendor skills shown first)."""
    settings = _get_settings(base_dir, verbose)
    from aqualib.skills.scanner import scan_all_skill_dirs
    from aqualib.workspace.manager import WorkspaceManager

    ws = WorkspaceManager(settings)
    skill_metas = scan_all_skill_dirs(settings, ws)

    table = Table(title="Registered Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Description")
    table.add_column("Tags", style="dim")

    for meta in skill_metas:
        table.add_row(
            f"vendor_{meta.name}",
            str(meta.skill_dir.parent.name),
            meta.description[:80],
            ", ".join(meta.tags),
        )
    console.print(table)


@app.command()
def sessions(
    base_dir: str | None = typer.Option(None, "--base-dir", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List all sessions in the current project."""
    settings = _get_settings(base_dir, verbose)
    from aqualib.workspace.manager import WorkspaceManager

    ws = WorkspaceManager(settings)
    all_sessions = ws.list_sessions()

    if not all_sessions:
        rprint("[dim]No sessions found. Run 'aqualib run' to create one.[/dim]")
        return

    active = ws.get_active_session()
    active_slug = active["slug"] if active else ""

    table = Table(title="Sessions")
    table.add_column("", width=2)  # active indicator
    table.add_column("Slug", style="cyan")
    table.add_column("Name")
    table.add_column("Tasks", justify="right")
    table.add_column("Last Updated", style="dim")
    table.add_column("Status")

    for s in all_sessions:
        indicator = "▶" if s["slug"] == active_slug else ""
        table.add_row(
            indicator,
            s["slug"],
            s.get("name", ""),
            str(s.get("task_count", 0)),
            s.get("updated_at", "")[:16],
            s.get("status", "active"),
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
        # Read the example config as the template (includes copilot: section)
        example_path = Path(__file__).parent.parent.parent / "aqualib.yaml.example"
        if example_path.is_file():
            cfg_template = example_path.read_text()
        else:
            cfg_template = (
                f"copilot:\n"
                f"  auth: \"github\"          # \"github\" | \"token\" | \"byok\"\n"
                f"  github_token: \"\"        # reads GH_TOKEN / GITHUB_TOKEN\n"
                f"  model: {settings.copilot.model}\n"
                f"  streaming: {str(settings.copilot.streaming).lower()}\n"
                f"  cli_path: null\n"
                f"  use_stdio: true\n"
                f"\n"
                f"directories:\n"
                f"  base: ./aqualib_workspace\n"
                f"\n"
                f"vendor_priority: {str(settings.vendor_priority).lower()}\n"
                f"\n"
                f"rag:\n"
                f"  enabled: false\n"
                f"  api_key: \"\"              # defaults to AQUALIB_RAG_API_KEY\n"
                f"  base_url: null\n"
                f"  chunk_size: {settings.rag.chunk_size}\n"
                f"  chunk_overlap: {settings.rag.chunk_overlap}\n"
                f"  similarity_top_k: {settings.rag.similarity_top_k}\n"
                f"  embed_model: {settings.rag.embed_model}\n"
                f"\n"
                f"llm:\n"
                f"  api_key: \"\"              # defaults to OPENAI_API_KEY env var\n"
                f"  base_url: null\n"
                f"  model: {settings.llm.model}\n"
                f"  temperature: {settings.llm.temperature}\n"
                f"  max_tokens: {settings.llm.max_tokens}\n"
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
