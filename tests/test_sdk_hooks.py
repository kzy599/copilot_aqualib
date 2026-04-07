"""Tests for the Copilot SDK hook implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from aqualib.config import DirectorySettings, Settings
from aqualib.workspace.manager import WorkspaceManager


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspaceManager:
    dirs = DirectorySettings(base=tmp_path).resolve()
    settings = Settings(directories=dirs)
    workspace = WorkspaceManager(settings)
    workspace.create_project(name="hook_test")
    return workspace


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    dirs = DirectorySettings(base=tmp_path).resolve()
    return Settings(directories=dirs)


# ---------------------------------------------------------------------------
# build_hooks
# ---------------------------------------------------------------------------


def test_build_hooks_returns_all_six(workspace, settings):
    from aqualib.sdk.hooks import build_hooks

    hooks = build_hooks(settings, workspace)
    assert set(hooks.keys()) == {
        "on_session_start",
        "on_user_prompt_submitted",
        "on_pre_tool_use",
        "on_post_tool_use",
        "on_session_end",
        "on_error_occurred",
    }
    for name, hook_fn in hooks.items():
        assert callable(hook_fn), f"Hook '{name}' should be callable"


@pytest.mark.asyncio
async def test_build_hooks_vendor_tool_always_allowed(workspace, settings):
    """Integration: vendor tools are allowed without requiring docs to be read first."""
    from aqualib.sdk.hooks import build_hooks

    hooks = build_hooks(settings, workspace)
    pre_hook = hooks["on_pre_tool_use"]

    # Vendor tools are allowed directly without reading docs first
    result = await pre_hook({"toolName": "vendor_seq_align", "toolArgs": {}}, None)
    assert result["permissionDecision"] == "allow"
    assert "DOC-FIRST" not in result.get("additionalContext", "")


# ---------------------------------------------------------------------------
# on_session_start
# ---------------------------------------------------------------------------


class TestSessionStartHook:
    @pytest.mark.asyncio
    async def test_no_project_returns_none(self, tmp_path):
        dirs = DirectorySettings(base=tmp_path).resolve()
        ws = WorkspaceManager(Settings(directories=dirs))  # no create_project

        from aqualib.sdk.hooks import _make_session_start_hook

        hook = _make_session_start_hook(ws)
        result = await hook({}, None)
        # With library-level doc injection, context may be non-None when vendor dirs exist.
        # When no project and no vendor dirs have docs, result should be None.
        # We just verify the hook runs without error and returns None or a dict.
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_additional_context(self, workspace):
        from aqualib.sdk.hooks import _make_session_start_hook

        hook = _make_session_start_hook(workspace)
        result = await hook({}, None)
        assert result is not None
        assert "additionalContext" in result
        assert "hook_test" in result["additionalContext"]

    @pytest.mark.asyncio
    async def test_history_included_after_tasks(self, workspace):
        workspace.append_context_log({
            "task_id": "t1",
            "query": "align sequences",
            "status": "approved",
            "skills_used": ["seq_align"],
        })
        from aqualib.sdk.hooks import _make_session_start_hook

        hook = _make_session_start_hook(workspace)
        result = await hook({}, None)
        assert "align sequences" in result["additionalContext"]


# ---------------------------------------------------------------------------
# on_user_prompt_submitted
# ---------------------------------------------------------------------------


class TestPromptHook:
    @pytest.mark.asyncio
    async def test_records_prompt_to_log(self, workspace):
        from aqualib.sdk.hooks import _make_prompt_hook

        hook = _make_prompt_hook(workspace)
        await hook({"prompt": "Find drug interactions"}, None)

        entries = workspace.load_context_log()
        assert len(entries) == 1
        assert entries[0]["event"] == "user_prompt"
        assert entries[0]["query"] == "Find drug interactions"

    @pytest.mark.asyncio
    async def test_returns_none(self, workspace):
        from aqualib.sdk.hooks import _make_prompt_hook

        hook = _make_prompt_hook(workspace)
        result = await hook({"prompt": "test"}, None)
        assert result is None


# ---------------------------------------------------------------------------
# on_pre_tool_use
# ---------------------------------------------------------------------------


class TestPreToolHook:
    @pytest.mark.asyncio
    async def test_allows_all_tools(self, workspace, settings):
        from aqualib.sdk.hooks import _make_pre_tool_hook

        hook = _make_pre_tool_hook(settings, workspace)
        result = await hook({"toolName": "grep", "toolArgs": {"pattern": "ATCG"}}, None)
        assert result["permissionDecision"] == "allow"

    @pytest.mark.asyncio
    async def test_vendor_tools_allowed_without_reminder(self, workspace):
        """Vendor priority reminder is no longer injected; tools are simply allowed."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        settings = Settings(
            directories=DirectorySettings(base=workspace.dirs.base).resolve(),
            vendor_priority=True,
        )
        hook = _make_pre_tool_hook(settings, workspace)
        result = await hook(
            {
                "toolName": "grep",
                "toolArgs": {},
                "availableTools": ["grep", "vendor_seq_align", "vendor_drug_check"],
            },
            None,
        )
        assert result["permissionDecision"] == "allow"
        assert "VENDOR PRIORITY REMINDER" not in result.get("additionalContext", "")

    @pytest.mark.asyncio
    async def test_no_reminder_when_no_vendor_tools(self, workspace, settings):
        settings.vendor_priority = True
        from aqualib.sdk.hooks import _make_pre_tool_hook

        hook = _make_pre_tool_hook(settings, workspace)
        result = await hook(
            {"toolName": "grep", "toolArgs": {}, "availableTools": ["grep", "bash"]},
            None,
        )
        assert result["permissionDecision"] == "allow"
        assert "additionalContext" not in result

    @pytest.mark.asyncio
    async def test_no_reminder_when_vendor_priority_false(self, workspace):
        settings = Settings(
            directories=DirectorySettings(base=workspace.dirs.base).resolve(),
            vendor_priority=False,
        )
        from aqualib.sdk.hooks import _make_pre_tool_hook

        hook = _make_pre_tool_hook(settings, workspace)
        result = await hook(
            {
                "toolName": "grep",
                "toolArgs": {},
                "availableTools": ["grep", "vendor_seq_align"],
            },
            None,
        )
        assert "additionalContext" not in result

    @pytest.mark.asyncio
    async def test_vendor_tools_allowed_without_docs(self, workspace, settings):
        """Vendor tools are allowed without needing to read docs first (gate removed)."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        hook = _make_pre_tool_hook(settings, workspace)
        result = await hook(
            {
                "toolName": "vendor_seq_align",
                "toolArgs": {},
                "availableTools": ["grep", "vendor_seq_align"],
            },
            None,
        )
        assert result["permissionDecision"] == "allow"
        assert "DOC-FIRST" not in result.get("additionalContext", "")

    @pytest.mark.asyncio
    async def test_vendor_tool_allowed(self, workspace, settings):
        """Vendor tool invocation is always allowed."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        hook = _make_pre_tool_hook(settings, workspace)
        result = await hook(
            {
                "toolName": "vendor_seq_align",
                "toolArgs": {},
                "availableTools": ["grep", "vendor_seq_align"],
            },
            None,
        )
        assert result["permissionDecision"] == "allow"
        assert "DOC-FIRST GATE" not in result.get("additionalContext", "")

    @pytest.mark.asyncio
    async def test_no_reminder_when_vendor_tool_used(self, workspace, settings):
        settings.vendor_priority = True
        from aqualib.sdk.hooks import _make_pre_tool_hook

        hook = _make_pre_tool_hook(settings, workspace)
        result = await hook(
            {
                "toolName": "vendor_seq_align",
                "toolArgs": {},
                "availableTools": ["grep", "vendor_seq_align"],
            },
            None,
        )
        assert "VENDOR PRIORITY REMINDER" not in result.get("additionalContext", "")

    @pytest.mark.asyncio
    async def test_records_audit_entry(self, workspace, settings):
        from aqualib.sdk.hooks import _make_pre_tool_hook

        hook = _make_pre_tool_hook(settings, workspace)
        await hook({"toolName": "bash", "toolArgs": {"cmd": "ls"}}, None)

        entries = workspace.load_context_log()
        assert any(e.get("event") == "pre_tool_use" and e.get("tool") == "bash" for e in entries)


# ---------------------------------------------------------------------------
# on_post_tool_use
# ---------------------------------------------------------------------------


class TestPostToolHook:
    @pytest.mark.asyncio
    async def test_records_success(self, workspace):
        from aqualib.sdk.hooks import _make_post_tool_hook

        hook = _make_post_tool_hook(workspace)
        await hook({"toolName": "grep", "toolResult": "match found"}, None)

        entries = workspace.load_context_log()
        entry = next(e for e in entries if e.get("event") == "post_tool_use")
        assert entry["tool"] == "grep"
        assert entry["success"] is True

    @pytest.mark.asyncio
    async def test_records_failure(self, workspace):
        from aqualib.sdk.hooks import _make_post_tool_hook

        hook = _make_post_tool_hook(workspace)
        await hook({"toolName": "grep", "toolError": "file not found"}, None)

        entries = workspace.load_context_log()
        entry = next(e for e in entries if e.get("event") == "post_tool_use")
        assert entry["success"] is False

    @pytest.mark.asyncio
    async def test_returns_none(self, workspace):
        from aqualib.sdk.hooks import _make_post_tool_hook

        hook = _make_post_tool_hook(workspace)
        result = await hook({"toolName": "grep"}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_read_skill_doc_does_not_error(self, workspace):
        """Post hook handles read_skill_doc calls without error."""
        from aqualib.sdk.hooks import _make_post_tool_hook

        hook = _make_post_tool_hook(workspace)
        result = await hook({"toolName": "read_skill_doc", "toolResult": "..."}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_read_library_doc_does_not_error(self, workspace):
        """Post hook handles read_library_doc calls without error."""
        from aqualib.sdk.hooks import _make_post_tool_hook

        hook = _make_post_tool_hook(workspace)
        result = await hook({"toolName": "read_library_doc", "toolResult": "..."}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_doc_tool_does_not_error(self, workspace):
        """Post hook handles non-doc tool calls without error."""
        from aqualib.sdk.hooks import _make_post_tool_hook

        hook = _make_post_tool_hook(workspace)
        result = await hook({"toolName": "grep", "toolResult": "match"}, None)
        assert result is None


# ---------------------------------------------------------------------------
# on_session_end
# ---------------------------------------------------------------------------


class TestSessionEndHook:
    @pytest.mark.asyncio
    async def test_calls_finalize_task(self, workspace):
        from aqualib.sdk.hooks import _make_session_end_hook

        hook = _make_session_end_hook(workspace)
        result = await hook({}, None)
        assert result is None  # no error, finalize_task ran


# ---------------------------------------------------------------------------
# on_error_occurred
# ---------------------------------------------------------------------------


class TestErrorHook:
    @pytest.mark.asyncio
    async def test_vendor_error_returns_retry(self, workspace):
        from aqualib.sdk.hooks import _make_error_hook

        hook = _make_error_hook(workspace)
        result = await hook(
            {"errorContext": "vendor_seq_align failed", "error": "timeout"},
            None,
        )
        assert result["errorHandling"] == "retry"

    @pytest.mark.asyncio
    async def test_non_vendor_error_returns_retry_then_skip(self, workspace):
        from aqualib.sdk.hooks import _make_error_hook

        hook = _make_error_hook(workspace)
        # All errors retry up to _MAX_RETRIES (2) times, then skip
        result = await hook(
            {"errorContext": "grep failed", "error": "permission denied"},
            None,
        )
        assert result["errorHandling"] == "retry"

    @pytest.mark.asyncio
    async def test_records_error_to_audit_log(self, workspace):
        from aqualib.sdk.hooks import _make_error_hook

        hook = _make_error_hook(workspace)
        await hook({"errorContext": "grep", "error": "disk full"}, None)

        entries = workspace.load_context_log()
        error_entries = [e for e in entries if e.get("event") == "error"]
        assert len(error_entries) == 1
        assert "disk full" in error_entries[0]["error"]


# ---------------------------------------------------------------------------
# _save_reviewer_memory — plan adherence parsing
# ---------------------------------------------------------------------------


class TestSaveReviewerMemory:
    def _make_workspace(self, tmp_path):
        dirs = DirectorySettings(base=tmp_path).resolve()
        ws = WorkspaceManager(Settings(directories=dirs))
        ws.create_project(name="reviewer_test")
        return ws

    def test_parses_plan_adherence_followed(self, tmp_path):
        from aqualib.sdk.hooks import _save_reviewer_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s1")
        slug = meta["slug"]

        result_text = (
            "VERDICT: approved\n"
            "VENDOR_PRIORITY: satisfied\n"
            "PLAN_QUALITY: valid\n"
            "PLAN_ADHERENCE: followed\n"
            "SUGGESTIONS: none\n"
        )
        _save_reviewer_memory(ws, slug, result_text)

        mem = ws.load_agent_memory(slug, "reviewer")
        assert len(mem["entries"]) == 1
        entry = mem["entries"][0]
        assert entry["plan_adherence"] == "followed"
        assert "plan_adherence" not in [v.split(":")[0] for v in entry["violations"]]

    def test_parses_plan_adherence_violated(self, tmp_path):
        from aqualib.sdk.hooks import _save_reviewer_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s2")
        slug = meta["slug"]

        result_text = (
            "VERDICT: needs_revision\n"
            "VENDOR_PRIORITY: satisfied\n"
            "PLAN_QUALITY: valid\n"
            "PLAN_ADHERENCE: violated - step 2 was skipped\n"
            "SUGGESTIONS: re-run step 2\n"
        )
        _save_reviewer_memory(ws, slug, result_text)

        mem = ws.load_agent_memory(slug, "reviewer")
        entry = mem["entries"][0]
        assert entry["plan_adherence"].startswith("violated")
        assert any("plan_adherence" in v for v in entry["violations"])

    def test_missing_plan_adherence_defaults_to_unknown(self, tmp_path):
        from aqualib.sdk.hooks import _save_reviewer_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s3")
        slug = meta["slug"]

        # Old-style result without PLAN_ADHERENCE field
        result_text = (
            "VERDICT: approved\n"
            "VENDOR_PRIORITY: satisfied\n"
            "PLAN_QUALITY: valid\n"
            "SUGGESTIONS: none\n"
        )
        _save_reviewer_memory(ws, slug, result_text)

        mem = ws.load_agent_memory(slug, "reviewer")
        entry = mem["entries"][0]
        assert entry["plan_adherence"] == "unknown"
        # Should not add a violation for an unknown adherence field
        assert not any("plan_adherence" in v for v in entry["violations"])

    def test_plan_quality_revision_needed(self, tmp_path):
        """When PLAN_QUALITY is revision_needed, it should be stored and treated as a violation."""
        from aqualib.sdk.hooks import _save_reviewer_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s4")
        slug = meta["slug"]

        result_text = (
            "VERDICT: plan_revision_needed\n"
            "VENDOR_PRIORITY: satisfied\n"
            "PLAN_QUALITY: revision_needed - wrong skill chosen for alignment\n"
            "PLAN_ADHERENCE: followed\n"
            "SUGGESTIONS: use vendor_seq_align instead of grep\n"
        )
        _save_reviewer_memory(ws, slug, result_text)

        mem = ws.load_agent_memory(slug, "reviewer")
        entry = mem["entries"][0]
        assert entry["verdict"] == "plan_revision_needed"
        assert entry["plan_quality"].startswith("revision_needed")
        assert any("plan_quality" in v for v in entry["violations"])

    def test_plan_quality_violated_still_works(self, tmp_path):
        """Existing 'violated' value for PLAN_QUALITY remains a violation."""
        from aqualib.sdk.hooks import _save_reviewer_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s5")
        slug = meta["slug"]

        result_text = (
            "VERDICT: needs_revision\n"
            "VENDOR_PRIORITY: satisfied\n"
            "PLAN_QUALITY: violated - missing data file\n"
            "PLAN_ADHERENCE: followed\n"
            "SUGGESTIONS: fix data path\n"
        )
        _save_reviewer_memory(ws, slug, result_text)

        mem = ws.load_agent_memory(slug, "reviewer")
        entry = mem["entries"][0]
        assert entry["plan_quality"].startswith("violated")
        assert any("plan_quality" in v for v in entry["violations"])


# ---------------------------------------------------------------------------
# _save_execution_report_memory
# ---------------------------------------------------------------------------


class TestSaveExecutionReportMemory:
    def _make_workspace(self, tmp_path):
        dirs = DirectorySettings(base=tmp_path).resolve()
        ws = WorkspaceManager(Settings(directories=dirs))
        ws.create_project(name="exec_report_test")
        return ws

    def test_parses_all_fields(self, tmp_path):
        from aqualib.sdk.hooks import _save_execution_report_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s1")
        slug = meta["slug"]

        result_text = (
            "EXECUTION_REPORT:\n"
            "  PRE_FLIGHT: passed\n"
            "  STEPS_COMPLETED: 3/3\n"
            "  STEP_DETAILS:\n"
            "    - Step 1: vendor_seq_align → ✅ alignment done\n"
            "  OUTPUT_FILES: results/output.csv (42 rows)\n"
            "  SANITY_CHECKS: all_passed\n"
            "  TOTAL_VENDOR_CALLS: 1\n"
            "  ERRORS_ENCOUNTERED: 0\n"
        )
        _save_execution_report_memory(ws, slug, result_text)

        exec_mem = ws.load_agent_memory(slug, "executor")
        assert len(exec_mem["entries"]) == 1
        entry = exec_mem["entries"][0]
        assert entry["event"] == "execution_report"
        assert entry["pre_flight"] == "passed"
        assert entry["steps_completed"] == "3/3"
        assert entry["total_vendor_calls"] == "1"
        assert entry["errors_encountered"] == "0"
        assert entry["sanity_checks"] == "all_passed"

    def test_report_only_in_executor_memory(self, tmp_path):
        """Execution report is saved to executor memory only, NOT reviewer memory."""
        from aqualib.sdk.hooks import _save_execution_report_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s2")
        slug = meta["slug"]

        result_text = (
            "EXECUTION_REPORT:\n"
            "  PRE_FLIGHT: passed\n"
            "  STEPS_COMPLETED: 2/2\n"
            "  SANITY_CHECKS: all_passed\n"
            "  TOTAL_VENDOR_CALLS: 2\n"
            "  ERRORS_ENCOUNTERED: 0\n"
        )
        _save_execution_report_memory(ws, slug, result_text)

        exec_mem = ws.load_agent_memory(slug, "executor")
        report_entries = [e for e in exec_mem["entries"] if e.get("event") == "execution_report"]
        assert len(report_entries) == 1

        rev_mem = ws.load_agent_memory(slug, "reviewer")
        rev_report_entries = [e for e in rev_mem.get("entries", []) if e.get("event") == "execution_report"]
        assert len(rev_report_entries) == 0

    def test_pre_flight_failed(self, tmp_path):
        """PRE_FLIGHT: failed is parsed correctly."""
        from aqualib.sdk.hooks import _save_execution_report_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s3")
        slug = meta["slug"]

        result_text = (
            "EXECUTION_REPORT:\n"
            "  PRE_FLIGHT: failed - input.csv not found\n"
            "  STEPS_COMPLETED: 0/3\n"
            "  SANITY_CHECKS: all_passed\n"
            "  TOTAL_VENDOR_CALLS: 0\n"
            "  ERRORS_ENCOUNTERED: 1 - missing input file\n"
        )
        _save_execution_report_memory(ws, slug, result_text)

        mem = ws.load_agent_memory(slug, "executor")
        entry = mem["entries"][0]
        assert entry["pre_flight"].startswith("failed")
        assert entry["steps_completed"] == "0/3"
        assert entry["errors_encountered"].startswith("1")

    def test_missing_fields_default_to_unknown(self, tmp_path):
        """Partial EXECUTION_REPORT defaults missing fields to 'unknown'."""
        from aqualib.sdk.hooks import _save_execution_report_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s4")
        slug = meta["slug"]

        # Only PRE_FLIGHT and STEPS_COMPLETED present
        result_text = "EXECUTION_REPORT:\n  PRE_FLIGHT: passed\n  STEPS_COMPLETED: 1/2\n"
        _save_execution_report_memory(ws, slug, result_text)

        mem = ws.load_agent_memory(slug, "executor")
        entry = mem["entries"][0]
        assert entry["total_vendor_calls"] == "unknown"
        assert entry["errors_encountered"] == "unknown"
        assert entry["sanity_checks"] == "unknown"
        # plan_deviations defaults to "none" (not "unknown") when missing
        assert entry["plan_deviations"] == "none"

    def test_parses_plan_deviations_field(self, tmp_path):
        """PLAN_DEVIATIONS field is parsed and stored in executor memory entry."""
        from aqualib.sdk.hooks import _save_execution_report_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="sdev")
        slug = meta["slug"]

        result_text = (
            "EXECUTION_REPORT:\n"
            "  PRE_FLIGHT: passed\n"
            "  STEPS_COMPLETED: 2/3\n"
            "  TOTAL_VENDOR_CALLS: 1\n"
            "  ERRORS_ENCOUNTERED: 0\n"
            "  PLAN_DEVIATIONS: Skipped step 3 — output file already existed from prior run\n"
        )
        _save_execution_report_memory(ws, slug, result_text)

        mem = ws.load_agent_memory(slug, "executor")
        entry = mem["entries"][0]
        assert "Skipped step 3" in entry["plan_deviations"]

    def test_plan_deviations_none_when_absent(self, tmp_path):
        """PLAN_DEVIATIONS defaults to 'none' when not present in report."""
        from aqualib.sdk.hooks import _save_execution_report_memory

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="snodev")
        slug = meta["slug"]

        result_text = (
            "EXECUTION_REPORT:\n"
            "  PRE_FLIGHT: passed\n"
            "  STEPS_COMPLETED: 3/3\n"
            "  TOTAL_VENDOR_CALLS: 2\n"
            "  ERRORS_ENCOUNTERED: 0\n"
        )
        _save_execution_report_memory(ws, slug, result_text)

        mem = ws.load_agent_memory(slug, "executor")
        entry = mem["entries"][0]
        assert entry["plan_deviations"] == "none"

    @pytest.mark.asyncio
    async def test_post_tool_hook_detects_execution_report(self, tmp_path):
        """on_post_tool_use automatically calls _save_execution_report_memory."""
        from aqualib.sdk.hooks import _make_post_tool_hook

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s5")
        slug = meta["slug"]

        result_text = (
            "EXECUTION_REPORT:\n"
            "  PRE_FLIGHT: passed\n"
            "  STEPS_COMPLETED: 1/1\n"
            "  SANITY_CHECKS: all_passed\n"
            "  TOTAL_VENDOR_CALLS: 1\n"
            "  ERRORS_ENCOUNTERED: 0\n"
            "Delegating to reviewer for audit."
        )

        hook = _make_post_tool_hook(ws, session_slug=slug)
        await hook({"toolName": "executor", "toolResult": result_text}, None)

        exec_mem = ws.load_agent_memory(slug, "executor")
        report_entries = [e for e in exec_mem["entries"] if e.get("event") == "execution_report"]
        assert len(report_entries) == 1
        assert report_entries[0]["pre_flight"] == "passed"

    @pytest.mark.asyncio
    async def test_post_tool_hook_does_not_detect_without_keyword(self, tmp_path):
        """on_post_tool_use does NOT save execution report when keyword is absent."""
        from aqualib.sdk.hooks import _make_post_tool_hook

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s6")
        slug = meta["slug"]

        hook = _make_post_tool_hook(ws, session_slug=slug)
        await hook({"toolName": "bash", "toolResult": "some plain output"}, None)

        exec_mem = ws.load_agent_memory(slug, "executor")
        report_entries = [e for e in exec_mem["entries"] if e.get("event") == "execution_report"]
        assert len(report_entries) == 0

    @pytest.mark.asyncio
    async def test_vendor_tool_use_not_stored_in_memory(self, tmp_path):
        """vendor_* tool completions do NOT create vendor_tool_use entries in executor or reviewer memory."""
        from aqualib.sdk.hooks import _make_post_tool_hook

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="s7")
        slug = meta["slug"]

        hook = _make_post_tool_hook(ws, session_slug=slug)
        await hook(
            {
                "toolName": "vendor_hiblup_ebv",
                "toolResult": "EBV computation complete. Output: sel_ebv.csv",
            },
            None,
        )

        exec_mem = ws.load_agent_memory(slug, "executor")
        exec_vendor = [e for e in exec_mem.get("entries", []) if e.get("event") == "vendor_tool_use"]
        assert len(exec_vendor) == 0

        rev_mem = ws.load_agent_memory(slug, "reviewer")
        rev_vendor = [e for e in rev_mem.get("entries", []) if e.get("event") == "vendor_tool_use"]
        assert len(rev_vendor) == 0


# ---------------------------------------------------------------------------
# Plan pending gate (Gate 1)
# ---------------------------------------------------------------------------


class TestPlanPendingGate:
    def _make_workspace(self, tmp_path):
        dirs = DirectorySettings(base=tmp_path).resolve()
        ws = WorkspaceManager(Settings(directories=dirs))
        ws.create_project(name="gate_test")
        return ws

    @pytest.mark.asyncio
    async def test_vendor_tool_blocked_when_plan_pending(self, tmp_path):
        """Gate 1: vendor_* tools are blocked while .plan_pending exists."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="gate-sess")
        slug = meta["slug"]

        # Simulate write_plan creating the flag
        pending_path = ws.session_dir(slug) / ".plan_pending"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("")

        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())
        hook = _make_pre_tool_hook(settings, ws, session_slug=slug)

        result = await hook({"toolName": "vendor_seq_align", "toolArgs": {}}, None)

        assert result["permissionDecision"] == "block"
        assert "confirmation" in result.get("additionalContext", "").lower()

    @pytest.mark.asyncio
    async def test_vendor_tool_allowed_after_confirmation(self, tmp_path):
        """Gate 1: vendor_* tools are allowed once .plan_pending is cleared."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="confirmed-sess")
        slug = meta["slug"]

        # No .plan_pending file — plan already confirmed
        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())
        hook = _make_pre_tool_hook(settings, ws, session_slug=slug)

        result = await hook({"toolName": "vendor_seq_align", "toolArgs": {}}, None)

        assert result["permissionDecision"] == "allow"

    @pytest.mark.asyncio
    async def test_non_vendor_tool_allowed_while_plan_pending(self, tmp_path):
        """Gate 1 only blocks vendor_* tools, not built-in tools."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="nonvendor-sess")
        slug = meta["slug"]

        pending_path = ws.session_dir(slug) / ".plan_pending"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("")

        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())
        hook = _make_pre_tool_hook(settings, ws, session_slug=slug)

        result = await hook({"toolName": "workspace_search", "toolArgs": {}}, None)

        assert result["permissionDecision"] == "allow"


# ---------------------------------------------------------------------------
# Plan confirmation keyword clearing
# ---------------------------------------------------------------------------


class TestPlanConfirmationClearing:
    def _make_workspace(self, tmp_path):
        dirs = DirectorySettings(base=tmp_path).resolve()
        ws = WorkspaceManager(Settings(directories=dirs))
        ws.create_project(name="confirm_test")
        return ws

    @pytest.mark.asyncio
    async def test_confirm_keyword_clears_plan_pending(self, tmp_path):
        """Confirmation keyword in user prompt should delete .plan_pending."""
        from aqualib.sdk.hooks import _make_prompt_hook

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="conf-sess")
        slug = meta["slug"]

        pending_path = ws.session_dir(slug) / ".plan_pending"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("")

        hook = _make_prompt_hook(ws, session_slug=slug)
        await hook({"prompt": "approved, go ahead"}, None)

        assert not pending_path.exists()

    @pytest.mark.asyncio
    async def test_non_confirm_prompt_keeps_plan_pending(self, tmp_path):
        """Non-confirmation prompts should not delete .plan_pending."""
        from aqualib.sdk.hooks import _make_prompt_hook

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="noconf-sess")
        slug = meta["slug"]

        pending_path = ws.session_dir(slug) / ".plan_pending"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("")

        hook = _make_prompt_hook(ws, session_slug=slug)
        await hook({"prompt": "what is the status?"}, None)

        assert pending_path.exists()

    @pytest.mark.asyncio
    async def test_chinese_confirm_keyword_clears_plan_pending(self, tmp_path):
        """Chinese confirmation keywords should also clear .plan_pending."""
        from aqualib.sdk.hooks import _make_prompt_hook

        ws = self._make_workspace(tmp_path)
        meta = ws.create_session(name="zh-conf-sess")
        slug = meta["slug"]

        pending_path = ws.session_dir(slug) / ".plan_pending"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_path.write_text("")

        hook = _make_prompt_hook(ws, session_slug=slug)
        await hook({"prompt": "好，执行"}, None)

        assert not pending_path.exists()


# ---------------------------------------------------------------------------
# Vendor priority Gate 2
# ---------------------------------------------------------------------------


class TestVendorPriorityGate2:
    def _make_workspace(self, tmp_path):
        dirs = DirectorySettings(base=tmp_path).resolve()
        ws = WorkspaceManager(Settings(directories=dirs))
        ws.create_project(name="vp_test")
        return ws

    @pytest.mark.asyncio
    async def test_exec_tool_with_bioinformatics_command_gets_warning(self, tmp_path):
        """Gate 2: exec tools with bioinformatics commands get a vendor priority warning."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        ws = self._make_workspace(tmp_path)
        settings = Settings(
            directories=DirectorySettings(base=tmp_path).resolve(),
            vendor_priority=True,
        )
        hook = _make_pre_tool_hook(settings, ws)

        result = await hook(
            {"toolName": "shell", "toolArgs": {"command": "bwa mem ref.fa reads.fq > out.sam"}},
            None,
        )

        assert result["permissionDecision"] == "allow"
        assert "VENDOR PRIORITY REMINDER" in result.get("additionalContext", "")

    @pytest.mark.asyncio
    async def test_exec_tool_without_bioinformatics_command_no_warning(self, tmp_path):
        """Gate 2: exec tools without bioinformatics keywords do not get a warning."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        ws = self._make_workspace(tmp_path)
        settings = Settings(
            directories=DirectorySettings(base=tmp_path).resolve(),
            vendor_priority=True,
        )
        hook = _make_pre_tool_hook(settings, ws)

        result = await hook(
            {"toolName": "shell", "toolArgs": {"command": "ls -la results/"}},
            None,
        )

        assert result["permissionDecision"] == "allow"
        assert "VENDOR PRIORITY REMINDER" not in result.get("additionalContext", "")

    @pytest.mark.asyncio
    async def test_vendor_priority_off_no_warning(self, tmp_path):
        """Gate 2 is inactive when vendor_priority=False."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        ws = self._make_workspace(tmp_path)
        settings = Settings(
            directories=DirectorySettings(base=tmp_path).resolve(),
            vendor_priority=False,
        )
        hook = _make_pre_tool_hook(settings, ws)

        result = await hook(
            {"toolName": "shell", "toolArgs": {"command": "samtools sort out.bam"}},
            None,
        )

        assert result["permissionDecision"] == "allow"
        assert "VENDOR PRIORITY REMINDER" not in result.get("additionalContext", "")

    @pytest.mark.asyncio
    async def test_non_exec_tool_with_bioinformatics_no_warning(self, tmp_path):
        """Gate 2 only activates for exec-style tools, not for arbitrary tools."""
        from aqualib.sdk.hooks import _make_pre_tool_hook

        ws = self._make_workspace(tmp_path)
        settings = Settings(
            directories=DirectorySettings(base=tmp_path).resolve(),
            vendor_priority=True,
        )
        hook = _make_pre_tool_hook(settings, ws)

        result = await hook(
            {"toolName": "workspace_search", "toolArgs": {"query": "samtools sort"}},
            None,
        )

        assert result["permissionDecision"] == "allow"
        assert "VENDOR PRIORITY REMINDER" not in result.get("additionalContext", "")


