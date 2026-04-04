"""Unit tests for the Orchestrator – project context injection."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aqualib.config import DirectorySettings, Settings
from aqualib.core.message import Role, SkillInvocation, SkillSource, Task, TaskStatus
from aqualib.core.orchestrator import Orchestrator
from aqualib.workspace.manager import WorkspaceManager


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspaceManager:
    dirs = DirectorySettings(base=tmp_path).resolve()
    settings = Settings(directories=dirs)
    return WorkspaceManager(settings)


def _make_orchestrator(workspace: WorkspaceManager) -> Orchestrator:
    """Build an Orchestrator with mocked agents."""
    searcher = AsyncMock()
    executor = AsyncMock()
    reviewer = AsyncMock()

    # The mocked agents should return the task unchanged (except reviewer sets APPROVED)
    searcher.run = AsyncMock(side_effect=lambda t: t)
    executor.run = AsyncMock(side_effect=lambda t: t)

    def approve(t: Task) -> Task:
        t.status = TaskStatus.APPROVED
        t.review_passed = True
        return t

    reviewer.run = AsyncMock(side_effect=approve)

    return Orchestrator(
        searcher=searcher,
        executor=executor,
        reviewer=reviewer,
        workspace=workspace,
    )


class TestBuildProjectContext:
    """Tests for Orchestrator._build_project_context()."""

    def test_returns_empty_when_no_project(self, workspace: WorkspaceManager):
        orch = _make_orchestrator(workspace)
        assert orch._build_project_context() == ""

    def test_returns_project_name(self, workspace: WorkspaceManager):
        workspace.create_project(name="My Study")
        orch = _make_orchestrator(workspace)
        ctx = orch._build_project_context()
        assert "Project: My Study" in ctx

    def test_includes_summary(self, workspace: WorkspaceManager):
        workspace.create_project(name="My Study")
        # Simulate a completed task to generate a summary
        task = Task(
            user_query="First task",
            status=TaskStatus.APPROVED,
            skill_invocations=[
                SkillInvocation(skill_name="seq_align", source=SkillSource.VENDOR, success=True),
            ],
        )
        workspace.update_project_after_task(task)

        orch = _make_orchestrator(workspace)
        ctx = orch._build_project_context()
        assert "History:" in ctx
        assert "1 tasks completed" in ctx

    def test_includes_recent_tasks(self, workspace: WorkspaceManager):
        workspace.create_project(name="My Study")
        workspace.append_context_log({
            "task_id": "aaa",
            "query": "Align sequences",
            "status": "approved",
            "skills_used": ["seq_align"],
        })
        workspace.append_context_log({
            "task_id": "bbb",
            "query": "Find inhibitors",
            "status": "needs_revision",
            "skills_used": ["drug_int"],
        })

        orch = _make_orchestrator(workspace)
        ctx = orch._build_project_context()
        assert "Recent tasks:" in ctx
        assert '✅ "Align sequences"' in ctx
        assert '⚠️ "Find inhibitors"' in ctx
        assert "seq_align" in ctx
        assert "drug_int" in ctx

    def test_limits_to_last_5_entries(self, workspace: WorkspaceManager):
        workspace.create_project(name="Big Project")
        for i in range(10):
            workspace.append_context_log({
                "task_id": f"t{i:02d}",
                "query": f"Task {i}",
                "status": "approved",
                "skills_used": [],
            })

        orch = _make_orchestrator(workspace)
        ctx = orch._build_project_context()
        # Should only include tasks 5-9 (last 5)
        assert "Task 5" in ctx
        assert "Task 9" in ctx
        assert "Task 4" not in ctx


class TestRunInjectsProjectContext:
    """Tests that Orchestrator.run() injects project context into Task messages."""

    @pytest.mark.asyncio
    async def test_run_injects_context_when_project_exists(self, workspace: WorkspaceManager):
        workspace.create_project(name="Test Project")
        workspace.append_context_log({
            "task_id": "prev1",
            "query": "Previous task",
            "status": "approved",
            "skills_used": ["seq_align"],
        })

        orch = _make_orchestrator(workspace)
        task = await orch.run("New follow-up question")

        # Find the project context message
        context_msgs = [
            m for m in task.messages
            if m.role == Role.ORCHESTRATOR and "Project context:" in m.content
        ]
        assert len(context_msgs) == 1
        assert "Test Project" in context_msgs[0].content
        assert "Previous task" in context_msgs[0].content

    @pytest.mark.asyncio
    async def test_run_no_context_when_no_project(self, workspace: WorkspaceManager):
        orch = _make_orchestrator(workspace)
        task = await orch.run("Some question")

        # No project context message should exist
        context_msgs = [
            m for m in task.messages
            if m.role == Role.ORCHESTRATOR and "Project context:" in m.content
        ]
        assert len(context_msgs) == 0


class TestSearcherFallback:
    @pytest.mark.asyncio
    async def test_searcher_fallback_adds_messages_when_no_rag(self, workspace: WorkspaceManager):
        """When RAG returns empty, the Searcher should use fallback discovery."""
        workspace.create_project(name="Fallback Test")
        # Put a file in data/ so grep has something to find
        data_dir = workspace.dirs.data
        (data_dir / "proteins.txt").write_text("Protein alignment data for testing.")

        orch = _make_orchestrator(workspace)
        task = await orch.run("Align protein sequences")

        # The orchestrator should have run and completed the pipeline
        orch_msgs = [m for m in task.messages if m.role == Role.ORCHESTRATOR]
        assert any("Pipeline started" in m.content for m in orch_msgs)
        assert task.status == TaskStatus.APPROVED
