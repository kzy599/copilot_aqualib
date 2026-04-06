"""Tests for the Plan-First workflow: write_plan tool, prompt updates, and integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aqualib.config import DirectorySettings, Settings
from aqualib.workspace.manager import WorkspaceManager


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    dirs = DirectorySettings(base=tmp_path).resolve()
    return Settings(directories=dirs)


@pytest.fixture()
def workspace(tmp_path: Path, settings: Settings) -> WorkspaceManager:
    ws = WorkspaceManager(settings)
    ws.create_project(name="Plan Test")
    return ws


# ---------------------------------------------------------------------------
# _write_plan_to_session
# ---------------------------------------------------------------------------


class TestWritePlanToSession:
    def test_creates_file_in_session_dir(self, workspace: WorkspaceManager) -> None:
        """write_plan should create plan.md in the session directory."""
        from aqualib.skills.tool_adapter import _write_plan_to_session

        meta = workspace.create_session(name="plan-sess")
        slug = meta["slug"]

        result = _write_plan_to_session(workspace, slug, "# Plan\n\nGoal: test")
        plan_path = workspace.session_dir(slug) / "plan.md"

        assert plan_path.exists()
        assert plan_path.read_text(encoding="utf-8") == "# Plan\n\nGoal: test"
        assert "Plan saved" in result

    def test_fallback_no_session(self, workspace: WorkspaceManager) -> None:
        """When session_slug is None, plan.md should be written to workspace root."""
        from aqualib.skills.tool_adapter import _write_plan_to_session

        result = _write_plan_to_session(workspace, None, "# Fallback Plan")
        plan_path = workspace.dirs.base / "plan.md"

        assert plan_path.exists()
        assert plan_path.read_text(encoding="utf-8") == "# Fallback Plan"
        assert "Plan saved" in result

    def test_overwrites_previous_plan(self, workspace: WorkspaceManager) -> None:
        """Subsequent writes should overwrite the previous plan."""
        from aqualib.skills.tool_adapter import _write_plan_to_session

        meta = workspace.create_session(name="overwrite-sess")
        slug = meta["slug"]

        _write_plan_to_session(workspace, slug, "# Plan A")
        _write_plan_to_session(workspace, slug, "# Plan B")

        plan_path = workspace.session_dir(slug) / "plan.md"
        content = plan_path.read_text(encoding="utf-8")
        assert content == "# Plan B"
        assert "Plan A" not in content


# ---------------------------------------------------------------------------
# build_tools_from_skills includes write_plan
# ---------------------------------------------------------------------------


class TestBuildToolsIncludesWritePlan:
    def test_includes_write_plan(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """build_tools_from_skills should include the write_plan tool."""
        from aqualib.skills.tool_adapter import build_tools_from_skills

        with patch("aqualib.skills.scanner.scan_all_skill_dirs", return_value=[]):
            tools = build_tools_from_skills(settings, workspace)

        tool_names = []
        for t in tools:
            if isinstance(t, dict):
                tool_names.append(t.get("name", ""))
            else:
                tool_names.append(getattr(t, "name", getattr(t, "__name__", "")))

        assert any("write_plan" in n for n in tool_names)


# ---------------------------------------------------------------------------
# System prompt contains Plan-First
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_contains_plan_first(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """System prompt guidelines should include Plan-First Workflow."""
        from aqualib.sdk.system_prompt import build_system_message

        msg = build_system_message(settings, workspace)
        # In customize mode, guidelines are in sections
        assert msg["mode"] == "customize"
        guidelines = msg["sections"]["guidelines"]["content"]
        assert "Plan-First" in guidelines
        assert "write_plan" in guidelines

    def test_identity_mentions_planner(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Identity section should describe AquaLib as a task planner."""
        from aqualib.sdk.system_prompt import build_system_message

        msg = build_system_message(settings, workspace)
        # In customize mode, identity is in sections
        assert msg["mode"] == "customize"
        identity = msg["sections"]["identity"]["content"]
        assert "task planner" in identity


# ---------------------------------------------------------------------------
# Agent prompts reference plan.md
# ---------------------------------------------------------------------------


class TestAgentPrompts:
    def test_executor_prompt_reads_plan(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Executor prompt should reference plan.md but NOT instruct re-reading it."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        executor = next(a for a in agents if a["name"] == "executor")
        assert "plan.md" in executor["prompt"]
        # Executor trusts conversation history — it should NOT redundantly re-read plan.md
        assert "do NOT re-read plan.md" in executor["prompt"]

    def test_reviewer_prompt_reads_plan(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Reviewer prompt should instruct reading plan.md independently."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        assert "plan.md" in reviewer["prompt"]
        assert "Read the Plan First" in reviewer["prompt"]

    def test_reviewer_prompt_audits_plan_adherence(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Reviewer prompt must include a plan adherence audit step and verdict field."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        # Explicit plan adherence responsibility in the prompt
        assert "Plan Adherence Audit" in reviewer["prompt"]
        # PLAN_ADHERENCE appears in the verdict format the reviewer must emit
        assert "PLAN_ADHERENCE" in reviewer["prompt"]

    def test_reviewer_prompt_audits_plan_reasonableness(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Reviewer prompt must evaluate the plan's soundness and can request revision."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        # Explicit plan reasonableness audit
        assert "Plan Reasonableness Audit" in reviewer["prompt"]
        # revision_needed is a valid PLAN_QUALITY value
        assert "revision_needed" in reviewer["prompt"]
        # plan_revision_needed is a valid VERDICT value
        assert "plan_revision_needed" in reviewer["prompt"]

    def test_reviewer_has_read_skill_doc_tool(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Reviewer needs read_skill_doc to independently verify skill capabilities."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        assert "read_skill_doc" in reviewer["tools"]

    def test_executor_prompt_handles_plan_revision(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Executor prompt must describe escalation when Reviewer requests plan revision."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        executor = next(a for a in agents if a["name"] == "executor")
        assert "Plan Revision Escalation" in executor["prompt"]
        assert "plan_revision_needed" in executor["prompt"]
        # Executor should NOT retry execution on plan revision
        assert "do NOT retry execution" in executor["prompt"]

    def test_planner_guidelines_include_plan_revision_loop(
        self, settings: Settings, workspace: WorkspaceManager,
    ) -> None:
        """Planner system prompt should describe the plan revision feedback loop."""
        from aqualib.sdk.system_prompt import build_system_message

        msg = build_system_message(settings, workspace)
        guidelines = msg["sections"]["guidelines"]["content"]
        assert "plan_revision_needed" in guidelines
        assert "Revise the plan" in guidelines


# ---------------------------------------------------------------------------
# Reviewer memory injection
# ---------------------------------------------------------------------------


class TestReviewerMemoryInjection:
    def test_reviewer_memory_injection_no_vendor_fragments(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Reviewer prompt must NOT contain vendor tool call fragments from executor memory."""
        from aqualib.sdk.agents import build_custom_agents

        meta = workspace.create_session(name="inj-test")
        slug = meta["slug"]

        # Seed executor memory with a vendor_tool_use entry (legacy / stray entry)
        workspace.append_agent_memory_entry(
            slug,
            "executor",
            {
                "event": "vendor_tool_use",
                "tool": "vendor_hiblup_ebv",
                "success": True,
                "output_preview": "EBV done",
            },
        )

        agents = build_custom_agents(settings, workspace, session_slug=slug)
        reviewer = next(a for a in agents if a["name"] == "reviewer")

        assert "vendor tool calls" not in reviewer["prompt"]
        assert "vendor_hiblup_ebv" not in reviewer["prompt"]

    def test_reviewer_prompt_describes_memory_structure(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Reviewer prompt must reference Execution Report and Previous Verdicts memory sources."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")

        assert "Execution Report" in reviewer["prompt"]
        assert "Previous Verdicts" in reviewer["prompt"]
        assert "did not produce execution report" in reviewer["prompt"]

