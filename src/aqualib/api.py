"""AquaLib REST API server (FastAPI).

Start with:
    aqualib-api              (via console_scripts, see pyproject.toml)
    uvicorn aqualib.api:app  (direct)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aqualib.bootstrap import build_orchestrator, build_registry
from aqualib.config import get_settings
from aqualib.core.orchestrator import Orchestrator
from aqualib.utils.logging import setup_logging
from aqualib.workspace.manager import WorkspaceManager

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_orchestrator: Orchestrator | None = None
_workspace: WorkspaceManager | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup: build the orchestrator; shutdown: nothing special."""
    global _orchestrator, _workspace  # noqa: PLW0603
    settings = get_settings()
    setup_logging(verbose=settings.verbose)
    _workspace = WorkspaceManager(settings)
    _orchestrator = await build_orchestrator(settings)
    yield


app = FastAPI(
    title="AquaLib API",
    version="0.1.0",
    description="Multi-agent framework with Clawbio skill priority and RAG retrieval.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    query: str = Field(..., description="User request / task description.")


class TaskSummary(BaseModel):
    task_id: str
    status: str
    review_passed: bool | None
    clawbio_priority_satisfied: bool | None
    review_notes: str
    skill_invocations: list[dict[str, Any]]


class SkillInfo(BaseModel):
    name: str
    source: str
    description: str
    tags: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/run", response_model=TaskSummary)
async def run_task(req: RunRequest):
    """Execute the full Searcher → Executor → Reviewer pipeline."""
    if _orchestrator is None:
        raise HTTPException(503, "Orchestrator not ready.")
    task = await _orchestrator.run(req.query)
    return TaskSummary(
        task_id=task.task_id,
        status=task.status.value,
        review_passed=task.review_passed,
        clawbio_priority_satisfied=task.clawbio_priority_satisfied,
        review_notes=task.review_notes,
        skill_invocations=[inv.model_dump(mode="json") for inv in task.skill_invocations],
    )


@app.get("/skills", response_model=list[SkillInfo])
async def list_skills():
    """List all registered skills."""
    settings = get_settings()
    registry = build_registry(settings)
    return [
        SkillInfo(
            name=s.meta.name,
            source=s.meta.source.value,
            description=s.meta.description,
            tags=s.meta.tags,
        )
        for s in registry.list_clawbio() + registry.list_generic()
    ]


@app.get("/tasks")
async def list_tasks():
    """List all completed tasks."""
    if _workspace is None:
        raise HTTPException(503, "Workspace not ready.")
    return _workspace.list_tasks()


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get full task state."""
    if _workspace is None:
        raise HTTPException(503, "Workspace not ready.")
    task = _workspace.load_task(task_id)
    if task is None:
        raise HTTPException(404, f"Task {task_id} not found.")
    return task.model_dump(mode="json")


@app.get("/tasks/{task_id}/report")
async def get_report(task_id: str, format: str = "json"):
    """Get audit report for a task."""
    if _workspace is None:
        raise HTTPException(503, "Workspace not ready.")
    report = _workspace.load_audit_report(task_id)
    if report is None:
        raise HTTPException(404, f"Audit report for task {task_id} not found.")
    if format == "markdown":
        return {"markdown": report.to_markdown()}
    return report.model_dump(mode="json")


@app.get("/tasks/{task_id}/skills")
async def get_task_skills(task_id: str):
    """List skill invocation outputs for a task."""
    if _workspace is None:
        raise HTTPException(503, "Workspace not ready.")
    return _workspace.list_skill_outputs(task_id)


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------

def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the API server (convenience wrapper)."""
    import uvicorn

    uvicorn.run("aqualib.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    serve()
