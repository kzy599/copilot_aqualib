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
        """Executor prompt should reference reading docs and vendor skills."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        executor = next(a for a in agents if a["name"] == "executor")
        assert "read_library_doc" in executor["prompt"]
        assert "read_skill_doc" in executor["prompt"]

    def test_reviewer_prompt_reads_plan(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Reviewer prompt should instruct reading plan.md independently."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        assert "plan.md" in reviewer["prompt"]
        assert "mandatory" in reviewer["prompt"]

    def test_reviewer_prompt_audits_plan_adherence(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Reviewer prompt must include a plan adherence verdict field."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        # PLAN_ADHERENCE appears in the verdict format the reviewer must emit
        assert "PLAN_ADHERENCE" in reviewer["prompt"]

    def test_reviewer_prompt_audits_plan_reasonableness(self, settings: Settings, workspace: WorkspaceManager) -> None:
        """Reviewer prompt must evaluate the plan's soundness and can request revision."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")
        # Audit plan quality step
        assert "plan quality" in reviewer["prompt"].lower()
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
        """Plan revision loop must be present in the system prompt (Planner's responsibility)."""
        from aqualib.sdk.system_prompt import build_system_message

        msg = build_system_message(settings, workspace)
        guidelines = msg["sections"]["guidelines"]["content"]
        # Plan revision loop belongs in system_prompt, not in the executor's condensed prompt
        assert "plan_revision_needed" in guidelines

    def test_planner_guidelines_include_plan_revision_loop(
        self, settings: Settings, workspace: WorkspaceManager,
    ) -> None:
        """Planner system prompt should describe the plan revision feedback loop."""
        from aqualib.sdk.system_prompt import build_system_message

        msg = build_system_message(settings, workspace)
        guidelines = msg["sections"]["guidelines"]["content"]
        assert "plan_revision_needed" in guidelines
        assert "revise" in guidelines.lower()


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

        assert "EXECUTION_REPORT" in reviewer["prompt"]
        assert "previous verdicts" in reviewer["prompt"].lower()


# ---------------------------------------------------------------------------
# .plan_pending flag
# ---------------------------------------------------------------------------


class TestPlanPendingFlag:
    def test_creates_plan_pending_flag(self, workspace: WorkspaceManager) -> None:
        """write_plan should create .plan_pending in the session directory."""
        from aqualib.skills.tool_adapter import _write_plan_to_session

        meta = workspace.create_session(name="pending-sess")
        slug = meta["slug"]

        _write_plan_to_session(workspace, slug, "# Plan\n\nGoal: test")
        pending_path = workspace.session_dir(slug) / ".plan_pending"

        assert pending_path.exists()

    def test_creates_plan_pending_fallback(self, workspace: WorkspaceManager) -> None:
        """write_plan with no session_slug should create .plan_pending in workspace root."""
        from aqualib.skills.tool_adapter import _write_plan_to_session

        _write_plan_to_session(workspace, None, "# Fallback")
        pending_path = workspace.dirs.base / ".plan_pending"

        assert pending_path.exists()

    def test_stop_instruction_in_result(self, workspace: WorkspaceManager) -> None:
        """write_plan should return a hard-stop instruction."""
        from aqualib.skills.tool_adapter import _write_plan_to_session

        meta = workspace.create_session(name="stop-sess")
        slug = meta["slug"]

        result = _write_plan_to_session(workspace, slug, "# Plan")

        assert "STOP HERE" in result
        assert "confirmation" in result.lower()

    def test_overwrites_plan_still_pending(self, workspace: WorkspaceManager) -> None:
        """Re-writing the plan should keep .plan_pending set."""
        from aqualib.skills.tool_adapter import _write_plan_to_session

        meta = workspace.create_session(name="repend-sess")
        slug = meta["slug"]

        _write_plan_to_session(workspace, slug, "# Plan A")
        _write_plan_to_session(workspace, slug, "# Plan B")

        pending_path = workspace.session_dir(slug) / ".plan_pending"
        assert pending_path.exists()


# ---------------------------------------------------------------------------
# Executor tools list
# ---------------------------------------------------------------------------


class TestExecutorToolsList:
    def test_executor_tools_includes_utility_tools(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Executor tools list should include utility tools even with no vendor skills."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace, skill_metas=[])
        executor = next(a for a in agents if a["name"] == "executor")

        assert "workspace_search" in executor["tools"]
        assert "read_library_doc" in executor["tools"]
        assert "read_skill_doc" in executor["tools"]
        assert "write_plan" in executor["tools"]

    def test_executor_tools_includes_vendor_tools(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Executor tools list should include all vendor tool names."""
        from pathlib import Path

        from aqualib.sdk.agents import build_custom_agents
        from aqualib.skills.scanner import SkillMeta

        meta1 = SkillMeta(
            name="seq_align", description="Sequence alignment", tags=[],
            version="1.0", parameters_schema={},
            skill_dir=Path("/fake"), vendor_root=Path("/fake"),
        )
        meta2 = SkillMeta(
            name="drug_check", description="Drug interaction", tags=[],
            version="1.0", parameters_schema={},
            skill_dir=Path("/fake"), vendor_root=Path("/fake"),
        )

        agents = build_custom_agents(settings, workspace, skill_metas=[meta1, meta2])
        executor = next(a for a in agents if a["name"] == "executor")

        assert "vendor_seq_align" in executor["tools"]
        assert "vendor_drug_check" in executor["tools"]

    def test_executor_infer_is_false(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Executor must have infer=False so delegation is always explicit."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        executor = next(a for a in agents if a["name"] == "executor")

        assert executor.get("infer") is False

    def test_reviewer_tools_no_nonexistent_names(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Reviewer tools must not include non-existent SDK tool names."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")

        for invalid in ("grep", "glob", "view", "read_file"):
            assert invalid not in reviewer["tools"]

    def test_reviewer_tools_contains_read_library_doc(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Reviewer tools must include read_library_doc."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")

        assert "read_library_doc" in reviewer["tools"]


# ---------------------------------------------------------------------------
# Planner delegation enforcement (system prompt)
# ---------------------------------------------------------------------------


class TestPlannerDelegation:
    def test_guidelines_forbid_direct_execution(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """System prompt guidelines must explicitly forbid Planner from executing."""
        from aqualib.sdk.system_prompt import build_system_message

        msg = build_system_message(settings, workspace)
        guidelines = msg["sections"]["guidelines"]["content"]

        assert "DO NOT" in guidelines
        assert "Executor" in guidelines

    def test_guidelines_require_user_confirmation(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """System prompt must tell Planner to wait for user confirmation."""
        from aqualib.sdk.system_prompt import build_system_message

        msg = build_system_message(settings, workspace)
        guidelines = msg["sections"]["guidelines"]["content"]

        assert "confirmation" in guidelines.lower() or "confirm" in guidelines.lower()


# ---------------------------------------------------------------------------
# all_tool_names auto-propagation
# ---------------------------------------------------------------------------


class TestAllToolNamesAutoPropagate:
    def test_executor_gets_all_tool_names(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Executor tool list should be exactly all_tool_names when provided."""
        from aqualib.sdk.agents import build_custom_agents

        all_tool_names = ["workspace_search", "read_skill_doc", "rag_search", "vendor_seq_align"]
        agents = build_custom_agents(settings, workspace, all_tool_names=all_tool_names)
        executor = next(a for a in agents if a["name"] == "executor")

        assert executor["tools"] == all_tool_names

    def test_reviewer_excludes_vendor_tools(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Reviewer tool list should exclude vendor_* tools from all_tool_names."""
        from aqualib.sdk.agents import build_custom_agents

        all_tool_names = [
            "workspace_search", "read_skill_doc", "rag_search",
            "vendor_seq_align", "vendor_drug_check",
        ]
        agents = build_custom_agents(settings, workspace, all_tool_names=all_tool_names)
        reviewer = next(a for a in agents if a["name"] == "reviewer")

        assert "workspace_search" in reviewer["tools"]
        assert "read_skill_doc" in reviewer["tools"]
        assert "rag_search" in reviewer["tools"]
        assert "vendor_seq_align" not in reviewer["tools"]
        assert "vendor_drug_check" not in reviewer["tools"]

    def test_rag_search_propagated_to_reviewer(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """rag_search (non-vendor) should be visible to reviewer when provided."""
        from aqualib.sdk.agents import build_custom_agents

        all_tool_names = ["workspace_search", "rag_search", "vendor_align"]
        agents = build_custom_agents(settings, workspace, all_tool_names=all_tool_names)
        reviewer = next(a for a in agents if a["name"] == "reviewer")

        assert "rag_search" in reviewer["tools"]

    def test_none_all_tool_names_falls_back_to_manual_list(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """When all_tool_names is None, executor falls back to manual list construction."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace, skill_metas=[], all_tool_names=None)
        executor = next(a for a in agents if a["name"] == "executor")

        assert "workspace_search" in executor["tools"]
        assert "read_library_doc" in executor["tools"]
        assert "read_skill_doc" in executor["tools"]
        assert "write_plan" in executor["tools"]

    def test_executor_prompt_includes_plan_deviations(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Executor prompt must reference PLAN_DEVIATIONS field in EXECUTION_REPORT."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        executor = next(a for a in agents if a["name"] == "executor")

        assert "PLAN_DEVIATIONS" in executor["prompt"]

    def test_reviewer_prompt_includes_adapted_justified(
        self, settings: Settings, workspace: WorkspaceManager
    ) -> None:
        """Reviewer prompt must include adapted_justified as a PLAN_ADHERENCE value."""
        from aqualib.sdk.agents import build_custom_agents

        agents = build_custom_agents(settings, workspace)
        reviewer = next(a for a in agents if a["name"] == "reviewer")

        assert "adapted_justified" in reviewer["prompt"]
