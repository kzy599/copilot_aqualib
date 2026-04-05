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
    assert workspace.dirs.skills_vendor.exists()
    assert workspace.dirs.vendor_traces.exists()


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
            SkillInvocation(skill_name="s1", source=SkillSource.VENDOR, success=True),
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


@pytest.mark.asyncio
async def test_next_invocation_dir_no_slug(workspace: WorkspaceManager):
    """Without a session slug the legacy work/inv_NNNN/ path is used."""
    d1 = await workspace.next_invocation_dir()
    d2 = await workspace.next_invocation_dir()
    assert d1.exists()
    assert d2.exists()
    assert d1.parent == workspace.dirs.work
    assert d2.parent == workspace.dirs.work
    assert d1 != d2


@pytest.mark.asyncio
async def test_next_invocation_dir_with_slug(workspace: WorkspaceManager):
    """With a session slug the path is scoped to work/<slug>/inv_NNNN/."""
    slug = "session-abc12345"
    d = await workspace.next_invocation_dir(session_slug=slug)
    assert d.exists()
    # Directory should be nested under work/<slug>/
    assert d.parent.parent == workspace.dirs.work
    assert d.parent.name == slug


@pytest.mark.asyncio
async def test_next_invocation_dir_different_slugs_do_not_collide(workspace: WorkspaceManager):
    """Two sessions with the same counter value get different directories."""
    # Both counters start at 0, so both would produce inv_0001 without scoping
    slug_a = "session-aaaaaaaa"
    slug_b = "session-bbbbbbbb"
    # We need separate WorkspaceManager instances to simulate separate processes
    from aqualib.config import DirectorySettings
    from aqualib.config import Settings as _Settings
    dirs = DirectorySettings(base=workspace.dirs.base).resolve()
    ws_b = WorkspaceManager(_Settings(directories=dirs))

    d_a = await workspace.next_invocation_dir(session_slug=slug_a)
    d_b = await ws_b.next_invocation_dir(session_slug=slug_b)

    # Both are inv_0001 in name but live under different session sub-directories
    assert d_a.name == d_b.name  # both "inv_0001"
    assert d_a != d_b            # but different absolute paths


def test_save_vendor_trace(workspace: WorkspaceManager):
    inv = SkillInvocation(
        skill_name="vendor_test_skill",
        source=SkillSource.VENDOR,
        parameters={"seq": "ATCG"},
        output={"score": 0.95},
        success=True,
    )
    trace_path = workspace.save_vendor_trace("task42", inv)
    assert trace_path.exists()
    assert trace_path.parent == workspace.dirs.vendor_traces

    data = json.loads(trace_path.read_text())
    assert data["task_id"] == "task42"
    assert data["skill_name"] == "vendor_test_skill"
    assert data["success"] is True

    # list_vendor_traces
    traces = workspace.list_vendor_traces("task42")
    assert len(traces) == 1
    assert traces[0]["skill_name"] == "vendor_test_skill"

    # listing without filter returns all
    all_traces = workspace.list_vendor_traces()
    assert len(all_traces) == 1


# ---------------------------------------------------------------------------
# Project metadata tests
# ---------------------------------------------------------------------------


def test_create_project_defaults(workspace: WorkspaceManager):
    """create_project uses the base dir name when no name given."""
    meta = workspace.create_project()
    assert meta["name"] == workspace.dirs.base.name
    assert meta["task_count"] == 0
    assert meta["description"] == ""
    assert meta["summary"] == ""
    assert len(meta["project_id"]) == 8
    assert workspace.dirs.project_file.exists()


def test_create_project_custom_name(workspace: WorkspaceManager):
    meta = workspace.create_project(name="my_study", description="protein research")
    assert meta["name"] == "my_study"
    assert meta["description"] == "protein research"


def test_load_project_returns_none_when_missing(workspace: WorkspaceManager):
    assert workspace.load_project() is None


def test_save_and_load_project(workspace: WorkspaceManager):
    meta = workspace.create_project(name="test_proj")
    loaded = workspace.load_project()
    assert loaded is not None
    assert loaded["name"] == "test_proj"
    assert loaded["project_id"] == meta["project_id"]


def test_append_and_load_context_log(workspace: WorkspaceManager):
    entry1 = {"task_id": "aaa", "query": "q1", "status": "approved", "skills_used": ["s1"]}
    entry2 = {"task_id": "bbb", "query": "q2", "status": "needs_revision", "skills_used": ["s2"]}
    workspace.append_context_log(entry1)
    workspace.append_context_log(entry2)

    entries = workspace.load_context_log()
    assert len(entries) == 2
    assert entries[0]["task_id"] == "aaa"
    assert entries[1]["task_id"] == "bbb"


def test_load_context_log_empty(workspace: WorkspaceManager):
    assert workspace.load_context_log() == []


def test_build_project_summary_empty(workspace: WorkspaceManager):
    assert workspace.build_project_summary() == ""


def test_build_project_summary(workspace: WorkspaceManager):
    workspace.append_context_log({
        "task_id": "a1", "query": "q1", "status": "approved",
        "skills_used": ["seq_align", "drug_int"], "timestamp": "2026-04-01T00:00:00",
    })
    workspace.append_context_log({
        "task_id": "a2", "query": "q2", "status": "approved",
        "skills_used": ["seq_align"], "timestamp": "2026-04-02T00:00:00",
    })
    workspace.append_context_log({
        "task_id": "a3", "query": "q3", "status": "needs_revision",
        "skills_used": ["drug_int"], "timestamp": "2026-04-03T00:00:00",
    })

    summary = workspace.build_project_summary()
    assert "3 tasks completed" in summary
    assert "approved" in summary
    assert "needs_revision" in summary
    assert "seq_align (2×)" in summary
    assert "drug_int (2×)" in summary
    assert "Last run: 2026-04-03" in summary


def test_update_project_after_task(workspace: WorkspaceManager):
    workspace.create_project(name="test")

    task = Task(
        user_query="Align MVKLF",
        status=TaskStatus.APPROVED,
        vendor_priority_satisfied=True,
        skill_invocations=[
            SkillInvocation(skill_name="seq_align", source=SkillSource.VENDOR, success=True),
        ],
    )
    workspace.update_project_after_task(task)

    meta = workspace.load_project()
    assert meta is not None
    assert meta["task_count"] == 1
    assert "1 tasks completed" in meta["summary"]

    entries = workspace.load_context_log()
    assert len(entries) == 1
    assert entries[0]["task_id"] == task.task_id
    assert entries[0]["status"] == "approved"
    assert entries[0]["skills_used"] == ["seq_align"]


def test_update_project_after_task_no_project(workspace: WorkspaceManager):
    """update_project_after_task is a no-op when no project.json exists."""
    task = Task(user_query="test", status=TaskStatus.APPROVED)
    workspace.update_project_after_task(task)  # should not raise
    assert workspace.load_project() is None
    assert workspace.load_context_log() == []


# ---------------------------------------------------------------------------
# scan_data_files tests
# ---------------------------------------------------------------------------


class TestScanDataFiles:
    def test_returns_empty_when_no_data(self, workspace: WorkspaceManager):
        results = workspace.scan_data_files("protein alignment")
        assert results == []

    def test_finds_matching_file(self, workspace: WorkspaceManager):
        data_dir = workspace.dirs.data
        (data_dir / "proteins.txt").write_text("MVKLF is a protein sequence for alignment testing.")
        results = workspace.scan_data_files("protein alignment")
        assert len(results) == 1
        assert results[0]["path"] == "proteins.txt"
        assert "protein" in results[0]["matched_keywords"]

    def test_ignores_short_keywords(self, workspace: WorkspaceManager):
        data_dir = workspace.dirs.data
        (data_dir / "test.txt").write_text("an is it")
        results = workspace.scan_data_files("an is it")
        assert results == []  # all keywords are ≤ 2 chars

    def test_respects_max_files(self, workspace: WorkspaceManager):
        data_dir = workspace.dirs.data
        for i in range(20):
            (data_dir / f"file_{i}.txt").write_text(f"protein data {i}")
        results = workspace.scan_data_files("protein", max_files=3)
        assert len(results) == 3

    def test_sorts_by_keyword_count(self, workspace: WorkspaceManager):
        data_dir = workspace.dirs.data
        (data_dir / "low.txt").write_text("protein data")
        (data_dir / "high.txt").write_text("protein alignment sequence data")
        results = workspace.scan_data_files("protein alignment sequence")
        assert results[0]["path"] == "high.txt"  # more keyword matches
