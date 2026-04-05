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
        # In override mode, content is a flat string
        content = msg["content"]
        assert "Plan-First" in content
        assert "write_plan" in content

    def test_identity_mentions_planner(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Identity section should describe AquaLib as a task planner."""
        from aqualib.sdk.system_prompt import build_system_message

        msg = build_system_message(settings, workspace)
        # In override mode, identity is embedded in the flat content string
        content = msg["content"]
        assert "task planner" in content


# ---------------------------------------------------------------------------
# Agent prompts reference plan.md
# ---------------------------------------------------------------------------


class TestAgentPrompts:
    def test_executor_prompt_reads_plan(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Executor prompt should instruct reading plan.md."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        executor = next(a for a in agents if a["name"] == "executor")
        assert "plan.md" in executor["prompt"]
        assert "Read the Plan" in executor["prompt"]

    def test_reviewer_prompt_reads_plan(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Reviewer prompt should instruct reading plan.md."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        assert "plan.md" in reviewer["prompt"]
        assert "Read the Plan" in reviewer["prompt"]
