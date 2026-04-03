"""Unit tests for the workspace manager."""

import json
from pathlib import Path

import pytest

from aqualib.config import DirectorySettings, Settings
from aqualib.core.message import AuditReport, SkillInvocation, SkillSource, Task, TaskStatus
from aqualib.workspace.manager import WorkspaceManager


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspaceManager:
    dirs = DirectorySettings(base=tmp_path).resolve()
    settings = Settings(directories=dirs)
    return WorkspaceManager(settings)


def test_dirs_created(workspace: WorkspaceManager):
    assert workspace.dirs.work.exists()
    assert workspace.dirs.results.exists()
    assert workspace.dirs.data.exists()


def test_save_and_load_task(workspace: WorkspaceManager):
    task = Task(user_query="test query")
    workspace.save_task(task)

    loaded = workspace.load_task(task.task_id)
    assert loaded is not None
    assert loaded.user_query == "test query"
    assert loaded.task_id == task.task_id


def test_list_tasks(workspace: WorkspaceManager):
    for i in range(3):
        t = Task(user_query=f"query {i}")
        workspace.save_task(t)
    assert len(workspace.list_tasks()) == 3


def test_save_audit_report(workspace: WorkspaceManager):
    report = AuditReport(
        task_id="test123",
        user_query="test",
        status=TaskStatus.APPROVED,
        executor_summary="ok",
        reviewer_verdict="approved",
        skill_invocations=[
            SkillInvocation(skill_name="s1", source=SkillSource.CLAWBIO, success=True),
        ],
    )
    td = workspace.save_audit_report(report)
    assert (td / "audit_report.json").exists()
    assert (td / "audit_report.md").exists()

    loaded = workspace.load_audit_report("test123")
    assert loaded is not None
    assert loaded.status == TaskStatus.APPROVED


def test_skill_invocation_dir(workspace: WorkspaceManager):
    d = workspace.skill_invocation_dir("task1", "inv1")
    assert d.exists()
    assert "task1" in str(d)
    assert "inv1" in str(d)

    # Write some files and list them
    (d / "result.json").write_text(json.dumps({"ok": True}))
    (d / "invocation_meta.json").write_text(json.dumps({"skill": "test"}))

    outputs = workspace.list_skill_outputs("task1")
    assert len(outputs) == 1
    assert "result.json" in outputs[0]["files"]
