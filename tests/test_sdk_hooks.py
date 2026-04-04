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
        assert result is None

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
    async def test_vendor_priority_reminder_when_vendor_available(self, workspace):
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
        assert "VENDOR PRIORITY REMINDER" in result.get("additionalContext", "")
        assert "vendor_seq_align" in result["additionalContext"]

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
        assert "additionalContext" not in result

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
    async def test_non_vendor_error_returns_skip(self, workspace):
        from aqualib.sdk.hooks import _make_error_hook

        hook = _make_error_hook(workspace)
        result = await hook(
            {"errorContext": "grep failed", "error": "permission denied"},
            None,
        )
        assert result["errorHandling"] == "skip"

    @pytest.mark.asyncio
    async def test_records_error_to_audit_log(self, workspace):
        from aqualib.sdk.hooks import _make_error_hook

        hook = _make_error_hook(workspace)
        await hook({"errorContext": "grep", "error": "disk full"}, None)

        entries = workspace.load_context_log()
        error_entries = [e for e in entries if e.get("event") == "error"]
        assert len(error_entries) == 1
        assert "disk full" in error_entries[0]["error"]
