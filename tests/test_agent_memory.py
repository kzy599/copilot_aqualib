"""Tests for agent role memory: read, write, compact, and injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from aqualib.config import DirectorySettings, Settings
from aqualib.workspace.manager import WorkspaceManager


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspaceManager:
    dirs = DirectorySettings(base=tmp_path).resolve()
    settings = Settings(directories=dirs)
    ws = WorkspaceManager(settings)
    ws.create_project(name="Memory Test")
    return ws


@pytest.fixture()
def session_slug(workspace: WorkspaceManager) -> str:
    meta = workspace.create_session(name="test-session")
    return meta["slug"]


# ---------------------------------------------------------------------------
# load_agent_memory
# ---------------------------------------------------------------------------


class TestLoadAgentMemory:
    def test_returns_empty_structure_when_no_file(
        self, workspace: WorkspaceManager, session_slug: str
    ):
        mem = workspace.load_agent_memory(session_slug, "executor")
        assert mem["agent"] == "executor"
        assert mem["session_slug"] == session_slug
        assert mem["entries"] == []

    def test_returns_saved_memory(self, workspace: WorkspaceManager, session_slug: str):
        memory = {
            "agent": "executor",
            "session_slug": session_slug,
            "entries": [{"query": "test", "skills_used": ["seq_align"]}],
        }
        workspace.save_agent_memory(session_slug, "executor", memory)
        loaded = workspace.load_agent_memory(session_slug, "executor")
        assert len(loaded["entries"]) == 1
        assert loaded["entries"][0]["query"] == "test"

    def test_handles_corrupt_file_gracefully(
        self, workspace: WorkspaceManager, session_slug: str
    ):
        path = workspace.session_dir(session_slug) / "memory" / "executor.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json{{{")
        mem = workspace.load_agent_memory(session_slug, "executor")
        assert mem["entries"] == []


# ---------------------------------------------------------------------------
# save_agent_memory
# ---------------------------------------------------------------------------


class TestSaveAgentMemory:
    def test_saves_memory_to_correct_path(
        self, workspace: WorkspaceManager, session_slug: str
    ):
        memory = {"agent": "executor", "session_slug": session_slug, "entries": []}
        workspace.save_agent_memory(session_slug, "executor", memory)
        path = workspace.session_dir(session_slug) / "memory" / "executor.json"
        assert path.exists()

    def test_creates_parent_directories(self, workspace: WorkspaceManager, tmp_path: Path):
        """Memory dir is created even if session was created externally."""
        dirs = DirectorySettings(base=tmp_path).resolve()
        settings = Settings(directories=dirs)
        ws = WorkspaceManager(settings)
        ws.create_project(name="ext")
        # Manually create session dir without memory subdir
        slug = "manual-session-abc"
        (ws.dirs.base / "sessions" / slug).mkdir(parents=True, exist_ok=True)
        memory = {"agent": "reviewer", "session_slug": slug, "entries": []}
        ws.save_agent_memory(slug, "reviewer", memory)
        path = ws.session_dir(slug) / "memory" / "reviewer.json"
        assert path.exists()

    def test_compacts_to_20_entries(self, workspace: WorkspaceManager, session_slug: str):
        entries = [{"query": f"task-{i}", "skills_used": []} for i in range(25)]
        memory = {"agent": "executor", "session_slug": session_slug, "entries": entries}
        workspace.save_agent_memory(session_slug, "executor", memory)
        loaded = workspace.load_agent_memory(session_slug, "executor")
        assert len(loaded["entries"]) == 20
        # Should keep the most recent 20
        assert loaded["entries"][0]["query"] == "task-5"
        assert loaded["entries"][-1]["query"] == "task-24"

    def test_keeps_exactly_20_entries(self, workspace: WorkspaceManager, session_slug: str):
        entries = [{"query": f"task-{i}"} for i in range(20)]
        memory = {"agent": "executor", "session_slug": session_slug, "entries": entries}
        workspace.save_agent_memory(session_slug, "executor", memory)
        loaded = workspace.load_agent_memory(session_slug, "executor")
        assert len(loaded["entries"]) == 20

    def test_keeps_fewer_than_20_entries(self, workspace: WorkspaceManager, session_slug: str):
        entries = [{"query": f"task-{i}"} for i in range(5)]
        memory = {"agent": "executor", "session_slug": session_slug, "entries": entries}
        workspace.save_agent_memory(session_slug, "executor", memory)
        loaded = workspace.load_agent_memory(session_slug, "executor")
        assert len(loaded["entries"]) == 5


# ---------------------------------------------------------------------------
# append_agent_memory_entry
# ---------------------------------------------------------------------------


class TestAppendAgentMemoryEntry:
    def test_appends_entry(self, workspace: WorkspaceManager, session_slug: str):
        workspace.append_agent_memory_entry(
            session_slug, "executor", {"query": "align sequences", "skills_used": ["seq_align"]}
        )
        mem = workspace.load_agent_memory(session_slug, "executor")
        assert len(mem["entries"]) == 1
        assert mem["entries"][0]["query"] == "align sequences"

    def test_appends_multiple_entries(self, workspace: WorkspaceManager, session_slug: str):
        for i in range(3):
            workspace.append_agent_memory_entry(
                session_slug, "executor", {"query": f"task-{i}"}
            )
        mem = workspace.load_agent_memory(session_slug, "executor")
        assert len(mem["entries"]) == 3

    def test_adds_timestamp_if_missing(self, workspace: WorkspaceManager, session_slug: str):
        workspace.append_agent_memory_entry(
            session_slug, "executor", {"query": "no timestamp"}
        )
        mem = workspace.load_agent_memory(session_slug, "executor")
        assert "timestamp" in mem["entries"][0]

    def test_preserves_existing_timestamp(self, workspace: WorkspaceManager, session_slug: str):
        workspace.append_agent_memory_entry(
            session_slug,
            "executor",
            {"query": "fixed time", "timestamp": "2026-01-01T00:00:00Z"},
        )
        mem = workspace.load_agent_memory(session_slug, "executor")
        assert mem["entries"][0]["timestamp"] == "2026-01-01T00:00:00Z"

    def test_auto_compact_at_21_entries(self, workspace: WorkspaceManager, session_slug: str):
        for i in range(21):
            workspace.append_agent_memory_entry(
                session_slug, "executor", {"query": f"task-{i}"}
            )
        mem = workspace.load_agent_memory(session_slug, "executor")
        assert len(mem["entries"]) == 20
        # Most recent 20 kept
        assert mem["entries"][-1]["query"] == "task-20"

    def test_separate_memory_per_agent(self, workspace: WorkspaceManager, session_slug: str):
        workspace.append_agent_memory_entry(
            session_slug, "executor", {"query": "executor task"}
        )
        workspace.append_agent_memory_entry(
            session_slug, "reviewer", {"query": "reviewer audit", "verdict": "approved"}
        )
        exec_mem = workspace.load_agent_memory(session_slug, "executor")
        rev_mem = workspace.load_agent_memory(session_slug, "reviewer")
        assert len(exec_mem["entries"]) == 1
        assert len(rev_mem["entries"]) == 1
        assert exec_mem["entries"][0]["query"] == "executor task"
        assert rev_mem["entries"][0]["query"] == "reviewer audit"


# ---------------------------------------------------------------------------
# Memory injection into build_custom_agents
# ---------------------------------------------------------------------------


class TestBuildCustomAgentsMemoryInjection:
    def test_no_memory_injection_without_workspace(self):
        from aqualib.config import Settings
        from aqualib.sdk.agents import build_custom_agents

        settings = Settings()
        agents = build_custom_agents(settings)
        assert len(agents) == 2
        executor = next(a for a in agents if a["name"] == "executor")
        # No memory context injected
        assert "Your previous work" not in executor["prompt"]

    def test_injects_executor_memory(self, workspace: WorkspaceManager, session_slug: str):
        """Executor does NOT get memory injection; it shares conversation history with Planner."""
        from aqualib.config import Settings
        from aqualib.sdk.agents import build_custom_agents

        workspace.append_agent_memory_entry(
            session_slug,
            "executor",
            {
                "query": "align MVKLF and MVKLT",
                "skills_used": ["sequence_alignment"],
                "output_preview": "alignment score 0.85",
            },
        )

        settings = Settings()
        agents = build_custom_agents(settings, workspace=workspace, session_slug=session_slug)
        executor = next(a for a in agents if a["name"] == "executor")
        # Executor shares conversation history with Planner — no separate memory injection
        assert "Your previous work in this session" not in executor["prompt"]
        assert "align MVKLF and MVKLT" not in executor["prompt"]

    def test_injects_reviewer_memory(self, workspace: WorkspaceManager, session_slug: str):
        from aqualib.config import Settings
        from aqualib.sdk.agents import build_custom_agents

        workspace.append_agent_memory_entry(
            session_slug,
            "reviewer",
            {
                "query": "audit alignment task",
                "verdict": "approved",
                "violations": [],
            },
        )

        settings = Settings()
        agents = build_custom_agents(settings, workspace=workspace, session_slug=session_slug)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        assert "Your previous verdicts in this session" in reviewer["prompt"]
        assert "audit alignment task" in reviewer["prompt"]

    def test_no_injection_when_empty_memory(self, workspace: WorkspaceManager, session_slug: str):
        from aqualib.config import Settings
        from aqualib.sdk.agents import build_custom_agents

        settings = Settings()
        agents = build_custom_agents(settings, workspace=workspace, session_slug=session_slug)
        executor = next(a for a in agents if a["name"] == "executor")
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        assert "Your previous work" not in executor["prompt"]
        assert "Your previous audits" not in reviewer["prompt"]

    def test_only_injects_last_5_entries(self, workspace: WorkspaceManager, session_slug: str):
        from aqualib.config import Settings
        from aqualib.sdk.agents import build_custom_agents

        # Add 8 vendor_tool_use entries to executor memory
        for i in range(8):
            workspace.append_agent_memory_entry(
                session_slug,
                "executor",
                {
                    "event": "vendor_tool_use",
                    "tool": f"vendor_task_{i}",
                    "success": True,
                    "output_preview": f"result-{i}",
                },
            )

        settings = Settings()
        agents = build_custom_agents(settings, workspace=workspace, session_slug=session_slug)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        prompt = reviewer["prompt"]

        # Should contain last 5 (task-3 through task-7), not older ones
        assert "vendor_task_7" in prompt
        assert "vendor_task_3" in prompt
        # task-0, task-1, task-2 should NOT be in the injected context
        assert "vendor_task_0" not in prompt
        assert "vendor_task_1" not in prompt
        assert "vendor_task_2" not in prompt

    def test_injects_execution_report_into_reviewer(
        self, workspace: WorkspaceManager, session_slug: str
    ):
        """Reviewer prompt includes the executor's latest EXECUTION_REPORT fields."""
        from aqualib.config import Settings
        from aqualib.sdk.agents import build_custom_agents

        workspace.append_agent_memory_entry(
            session_slug,
            "executor",
            {
                "event": "execution_report",
                "pre_flight": "passed",
                "steps_completed": "2/2",
                "total_vendor_calls": "2",
                "errors_encountered": "0",
                "sanity_checks": "all_passed",
            },
        )

        settings = Settings()
        agents = build_custom_agents(settings, workspace=workspace, session_slug=session_slug)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        prompt = reviewer["prompt"]

        assert "Executor's latest execution report" in prompt
        assert "PRE_FLIGHT: passed" in prompt
        assert "STEPS_COMPLETED: 2/2" in prompt
        assert "TOTAL_VENDOR_CALLS: 2" in prompt
        assert "SANITY_CHECKS: all_passed" in prompt
        assert "ERRORS_ENCOUNTERED: 0" in prompt

    def test_no_execution_report_section_when_absent(
        self, workspace: WorkspaceManager, session_slug: str
    ):
        """When no execution_report in memory, reviewer prompt has no report section."""
        from aqualib.config import Settings
        from aqualib.sdk.agents import build_custom_agents

        settings = Settings()
        agents = build_custom_agents(settings, workspace=workspace, session_slug=session_slug)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        assert "Executor's latest execution report" not in reviewer["prompt"]

    def test_only_latest_execution_report_injected(
        self, workspace: WorkspaceManager, session_slug: str
    ):
        """Only the most recent EXECUTION_REPORT is injected into reviewer prompt."""
        from aqualib.config import Settings
        from aqualib.sdk.agents import build_custom_agents

        workspace.append_agent_memory_entry(
            session_slug,
            "executor",
            {
                "event": "execution_report",
                "pre_flight": "failed - old run",
                "steps_completed": "0/3",
                "total_vendor_calls": "0",
                "errors_encountered": "1",
                "sanity_checks": "unknown",
            },
        )
        workspace.append_agent_memory_entry(
            session_slug,
            "executor",
            {
                "event": "execution_report",
                "pre_flight": "passed",
                "steps_completed": "3/3",
                "total_vendor_calls": "3",
                "errors_encountered": "0",
                "sanity_checks": "all_passed",
            },
        )

        settings = Settings()
        agents = build_custom_agents(settings, workspace=workspace, session_slug=session_slug)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        prompt = reviewer["prompt"]

        # Latest report (passed) should be present; old one (failed) should not
        assert "PRE_FLIGHT: passed" in prompt
        assert "failed - old run" not in prompt


# ---------------------------------------------------------------------------
# on_session_end hook — resource cleanup only (executor memory moved to CLI)
# ---------------------------------------------------------------------------


class TestSessionEndHookMemory:
    @pytest.mark.asyncio
    async def test_session_end_does_not_write_executor_memory(
        self, workspace: WorkspaceManager, session_slug: str
    ):
        """on_session_end hook should only call finalize_task(), not write executor memory."""
        from aqualib.sdk.hooks import _make_session_end_hook

        hook = _make_session_end_hook(workspace, session_slug)
        await hook(
            {
                "query": "align sequences",
                "skills_used": ["sequence_alignment"],
                "summary": "aligned 2 sequences",
            },
            None,
        )

        mem = workspace.load_agent_memory(session_slug, "executor")
        assert len(mem["entries"]) == 0

    @pytest.mark.asyncio
    async def test_does_not_write_when_no_slug(self, workspace: WorkspaceManager):
        from aqualib.sdk.hooks import _make_session_end_hook

        hook = _make_session_end_hook(workspace, None)
        # Should not raise
        await hook({"query": "test", "skills_used": []}, None)


# ---------------------------------------------------------------------------
# CLI-layer executor memory write (Bug 1 fix)
# ---------------------------------------------------------------------------


class TestCLIExecutorMemoryWrite:
    def test_append_agent_memory_entry_writes_executor_memory(
        self, workspace: WorkspaceManager, session_slug: str
    ):
        """CLI layer should write executor memory with query, skills_used, output_preview."""
        query = "align sequences"
        task_skills = ["vendor_seq_align"]
        result_messages = ["Alignment completed with score 0.85"]

        workspace.append_agent_memory_entry(session_slug, "executor", {
            "query": query,
            "skills_used": task_skills,
            "output_preview": (result_messages[-1][:200] if result_messages else ""),
        })

        mem = workspace.load_agent_memory(session_slug, "executor")
        assert len(mem["entries"]) == 1
        assert mem["entries"][0]["query"] == "align sequences"
        assert mem["entries"][0]["skills_used"] == ["vendor_seq_align"]
        assert "score 0.85" in mem["entries"][0]["output_preview"]

    def test_empty_result_messages_gives_empty_preview(
        self, workspace: WorkspaceManager, session_slug: str
    ):
        """When no result messages, output_preview should be empty string."""
        result_messages: list[str] = []
        workspace.append_agent_memory_entry(session_slug, "executor", {
            "query": "test",
            "skills_used": [],
            "output_preview": (result_messages[-1][:200] if result_messages else ""),
        })
        mem = workspace.load_agent_memory(session_slug, "executor")
        assert mem["entries"][0]["output_preview"] == ""
