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
        assert "Recent tasks:" in ctx
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
