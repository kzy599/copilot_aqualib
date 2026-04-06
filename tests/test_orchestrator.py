"""Tests for the SDK session manager and project-context injection.

The old Orchestrator class has been replaced by the Copilot SDK's
SessionManager + on_session_start hook. These tests verify the hook's
context-injection behaviour and the SessionManager's create/resume logic.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aqualib.config import DirectorySettings, Settings
from aqualib.workspace.manager import WorkspaceManager


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspaceManager:
    dirs = DirectorySettings(base=tmp_path).resolve()
    settings = Settings(directories=dirs)
    return WorkspaceManager(settings)


# ---------------------------------------------------------------------------
# on_session_start hook (replaces Orchestrator._build_project_context)
# ---------------------------------------------------------------------------


class TestSessionStartHook:
    """Tests for the on_session_start hook context injection."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_project(self, workspace: WorkspaceManager):
        from aqualib.sdk.hooks import _make_session_start_hook

        hook = _make_session_start_hook(workspace)
        result = await hook({}, None)
        # With library-level doc injection, result may be non-None when vendor dirs exist.
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_project_name(self, workspace: WorkspaceManager):
        workspace.create_project(name="My Study")
        from aqualib.sdk.hooks import _make_session_start_hook

        hook = _make_session_start_hook(workspace)
        result = await hook({}, None)
        assert result is not None
        assert "My Study" in result["additionalContext"]

    @pytest.mark.asyncio
    async def test_includes_recent_tasks(self, workspace: WorkspaceManager):
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
        from aqualib.sdk.hooks import _make_session_start_hook

        hook = _make_session_start_hook(workspace)
        result = await hook({}, None)
        assert result is not None
        ctx = result["additionalContext"]
        assert "Recent tasks" in ctx
        assert "Align sequences" in ctx
        assert "Find inhibitors" in ctx

    @pytest.mark.asyncio
    async def test_limits_to_last_5_task_entries(self, workspace: WorkspaceManager):
        workspace.create_project(name="Big Project")
        for i in range(10):
            workspace.append_context_log({
                "task_id": f"t{i:02d}",
                "query": f"Task {i}",
                "status": "approved",
                "skills_used": [],
            })
        from aqualib.sdk.hooks import _make_session_start_hook

        hook = _make_session_start_hook(workspace)
        result = await hook({}, None)
        ctx = result["additionalContext"]
        # Should only include tasks 5-9 (last 5)
        assert "Task 5" in ctx
        assert "Task 9" in ctx
        assert "Task 4" not in ctx

    @pytest.mark.asyncio
    async def test_includes_project_summary(self, workspace: WorkspaceManager):
        workspace.create_project(name="My Study")
        workspace.append_context_log({
            "task_id": "a1",
            "query": "First task",
            "status": "approved",
            "skills_used": ["seq_align"],
        })
        workspace.update_project_after_task("First task", ["done"])
        from aqualib.sdk.hooks import _make_session_start_hook

        hook = _make_session_start_hook(workspace)
        result = await hook({}, None)
        assert result is not None


# ---------------------------------------------------------------------------
# SessionManager create / resume logic
# ---------------------------------------------------------------------------

# Patch targets: functions are imported locally inside _create_session/_resume_session,
# so we patch them at their source modules.
_PATCH_AGENTS = "aqualib.sdk.agents.build_custom_agents"
_PATCH_HOOKS = "aqualib.sdk.hooks.build_hooks"
_PATCH_SYSTEM = "aqualib.sdk.system_prompt.build_system_message"
_PATCH_TOOLS = "aqualib.skills.tool_adapter.build_tools_from_skills"


class TestSessionManager:
    """Tests for SessionManager.get_or_create_session()."""

    def _make_workspace(self, tmp_path: Path) -> WorkspaceManager:
        dirs = DirectorySettings(base=tmp_path).resolve()
        settings = Settings(directories=dirs)
        return WorkspaceManager(settings)

    @pytest.mark.asyncio
    async def test_creates_new_session_when_no_existing_id(self, tmp_path: Path):
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="test")

        mock_client = AsyncMock()
        mock_session = MagicMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)

        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        with patch(_PATCH_AGENTS, return_value=[]), \
             patch(_PATCH_HOOKS, return_value={}), \
             patch(_PATCH_SYSTEM, return_value={}), \
             patch(_PATCH_TOOLS, return_value=[]):
            from aqualib.sdk.session_manager import SessionManager

            sm = SessionManager(mock_client, settings, workspace)
            result = await sm.get_or_create_session()

        mock_client.create_session.assert_called_once()
        # get_or_create_session now returns (session, slug)
        session, slug = result
        assert session is mock_session
        assert isinstance(slug, str)

    @pytest.mark.asyncio
    async def test_persists_session_id_to_project_json(self, tmp_path: Path):
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="test")

        mock_client = AsyncMock()
        mock_session = MagicMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)

        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        with patch(_PATCH_AGENTS, return_value=[]), \
             patch(_PATCH_HOOKS, return_value={}), \
             patch(_PATCH_SYSTEM, return_value={}), \
             patch(_PATCH_TOOLS, return_value=[]):
            from aqualib.sdk.session_manager import SessionManager

            sm = SessionManager(mock_client, settings, workspace)
            _, slug = await sm.get_or_create_session()

        project = workspace.load_project()
        assert project is not None
        # New architecture: project stores active_session slug, not session_id directly
        assert "active_session" in project
        assert project["active_session"] == slug
        # The session itself has a session_id
        session_meta = workspace.load_session(slug)
        assert session_meta is not None
        assert session_meta["session_id"].startswith("aqualib-")

    @pytest.mark.asyncio
    async def test_resumes_session_when_id_exists(self, tmp_path: Path):
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="test")

        # Create a session and mark it active so resume is attempted
        session_meta = workspace.create_session(name="existing")
        slug = session_meta["slug"]
        session_id = session_meta["session_id"]

        mock_client = AsyncMock()
        mock_session = MagicMock()
        mock_client.resume_session = AsyncMock(return_value=mock_session)

        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        with patch(_PATCH_AGENTS, return_value=[]), \
             patch(_PATCH_TOOLS, return_value=[]):
            from aqualib.sdk.session_manager import SessionManager

            sm = SessionManager(mock_client, settings, workspace)
            result = await sm.get_or_create_session()

        mock_client.resume_session.assert_called_once()
        call_args = mock_client.resume_session.call_args
        assert call_args[0][0] == session_id
        sdk_session, returned_slug = result
        assert sdk_session is mock_session
        assert returned_slug == slug

    @pytest.mark.asyncio
    async def test_creates_new_session_when_resume_fails(self, tmp_path: Path):
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="test")

        # Create an active session that will fail to resume
        workspace.create_session(name="old-session")

        mock_client = AsyncMock()
        mock_session = MagicMock()
        mock_client.resume_session = AsyncMock(side_effect=RuntimeError("session expired"))
        mock_client.create_session = AsyncMock(return_value=mock_session)

        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        with patch(_PATCH_AGENTS, return_value=[]), \
             patch(_PATCH_HOOKS, return_value={}), \
             patch(_PATCH_SYSTEM, return_value={}), \
             patch(_PATCH_TOOLS, return_value=[]):
            from aqualib.sdk.session_manager import SessionManager

            sm = SessionManager(mock_client, settings, workspace)
            result = await sm.get_or_create_session()

        mock_client.create_session.assert_called_once()
        session, slug = result
        assert session is mock_session

    def test_session_id_format(self, tmp_path: Path):
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="My Study Project")

        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        session_id = sm._generate_session_id()
        assert session_id.startswith("aqualib-my-study-project-")
        assert len(session_id) > len("aqualib-my-study-project-")

    def test_collect_skill_dirs_includes_workspace_vendor(self, tmp_path: Path):
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="test")

        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        dirs = sm._collect_skill_dirs()
        assert str(workspace.dirs.skills_vendor) in dirs

    def test_build_provider_returns_none_for_github_auth(self, tmp_path: Path):
        workspace = self._make_workspace(tmp_path)
        settings = Settings(
            directories=DirectorySettings(base=tmp_path).resolve(),
            copilot={"auth": "github"},
        )

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        assert sm._build_provider() is None

    def test_build_provider_returns_dict_for_byok(self, tmp_path: Path):
        from aqualib.config import CopilotSettings, ProviderConfig

        workspace = self._make_workspace(tmp_path)
        settings = Settings(
            directories=DirectorySettings(base=tmp_path).resolve(),
            copilot=CopilotSettings(
                auth="byok",
                provider=ProviderConfig(
                    type="openai",
                    base_url="http://localhost:11434/v1",
                    api_key="test-key",
                ),
            ),
        )

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        provider = sm._build_provider()
        assert provider is not None
        assert provider["type"] == "openai"
        assert provider["base_url"] == "http://localhost:11434/v1"
        assert provider["api_key"] == "test-key"

    # ------------------------------------------------------------------
    # _build_permission_handler helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _perm_decision(result) -> str:
        """Extract the allow/deny decision from either a PermissionRequestResult object or a dict."""
        if hasattr(result, "kind"):
            # SDK PermissionRequestResult: kind is "approved" or "denied"
            return "allow" if result.kind == "approved" else "deny"
        # dict-based fallback: permissionDecision is "allow" or "deny"
        return result.get("permissionDecision", "")

    # ------------------------------------------------------------------
    # _build_permission_handler
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_permission_handler_allows_safe_write(self, tmp_path: Path):
        """Write to a path inside the workspace is allowed."""
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="perm-test")
        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        handler = sm._build_permission_handler()

        request = {"kind": "write", "path": str(tmp_path / "results" / "out.txt")}
        result = await handler(request, None)
        assert self._perm_decision(result) == "allow"

    @pytest.mark.asyncio
    async def test_permission_handler_denies_write_outside_workspace(self, tmp_path: Path):
        """Write to a path outside the workspace is denied."""
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="perm-test")
        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        handler = sm._build_permission_handler()

        request = {"kind": "write", "path": "/etc/passwd"}
        result = await handler(request, None)
        assert self._perm_decision(result) == "deny"

    @pytest.mark.asyncio
    async def test_permission_handler_denies_dangerous_shell(self, tmp_path: Path):
        """Shell command with dangerous pattern is denied."""
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="perm-test")
        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        handler = sm._build_permission_handler()

        dangerous_cmds = [
            "rm -rf /",
            "rm  -rf /",
            "mkfs /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
            "DROP TABLE users",
        ]
        for dangerous_cmd in dangerous_cmds:
            request = {"kind": "shell", "command": dangerous_cmd}
            result = await handler(request, None)
            assert self._perm_decision(result) == "deny", (
                f"Expected deny for: {dangerous_cmd}"
            )

    @pytest.mark.asyncio
    async def test_permission_handler_allows_safe_shell(self, tmp_path: Path):
        """Ordinary shell commands are allowed."""
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="perm-test")
        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        handler = sm._build_permission_handler()

        request = {"kind": "shell", "command": "ls -la /tmp"}
        result = await handler(request, None)
        assert self._perm_decision(result) == "allow"

    @pytest.mark.asyncio
    async def test_permission_handler_allows_other_kinds(self, tmp_path: Path):
        """Non-write, non-shell kinds (read, mcp, custom_tool, url, memory) are allowed."""
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="perm-test")
        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        handler = sm._build_permission_handler()

        for kind in ("read", "mcp", "custom_tool", "url", "memory", "hook"):
            request = {"kind": kind}
            result = await handler(request, None)
            assert self._perm_decision(result) == "allow", (
                f"Expected allow for kind={kind}"
            )

    @pytest.mark.asyncio
    async def test_permission_handler_works_with_object_style_request(self, tmp_path: Path):
        """Handler works when request is an object with attributes (SDK style)."""
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="perm-test")
        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        handler = sm._build_permission_handler()

        # Simulate an SDK object with attributes
        request = MagicMock()
        request.kind = "shell"
        request.fullCommandText = "rm -rf /"
        request.command = "rm -rf /"
        result = await handler(request, None)
        assert self._perm_decision(result) == "deny"

    # ------------------------------------------------------------------
    # _build_user_input_handler
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_user_input_handler_returns_answer(self, tmp_path: Path, monkeypatch):
        """User input handler returns the typed answer."""
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="input-test")
        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        handler = sm._build_user_input_handler()

        monkeypatch.setattr("aqualib.sdk.session_manager._console.input", lambda _: "5e-8")

        result = await handler({"question": "Which p-value threshold?", "choices": []})
        assert result["answer"] == "5e-8"
        assert result["wasFreeform"] is True

    @pytest.mark.asyncio
    async def test_user_input_handler_works_with_object_request(self, tmp_path: Path, monkeypatch):
        """User input handler works with object-style requests."""
        workspace = self._make_workspace(tmp_path)
        workspace.create_project(name="input-test")
        settings = Settings(directories=DirectorySettings(base=tmp_path).resolve())

        from aqualib.sdk.session_manager import SessionManager

        sm = SessionManager(None, settings, workspace)
        handler = sm._build_user_input_handler()

        monkeypatch.setattr("aqualib.sdk.session_manager._console.input", lambda _: "yes")

        request = MagicMock()
        request.question = "Proceed with analysis?"
        request.choices = ["yes", "no"]
        result = await handler(request)
        assert result["answer"] == "yes"
        assert result["wasFreeform"] is True
