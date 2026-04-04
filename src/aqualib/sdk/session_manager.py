"""SessionManager – Copilot SDK session creation and resumption.

Each AquaLib project maps to a single Copilot ``session_id`` stored in
``project.json``.  On each ``aqualib run`` the manager first tries to resume
the existing session; if that fails (e.g. the CLI was restarted) it creates a
fresh one.

Session ID format: ``aqualib-{project_name_slug}-{8-char-uuid}``
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aqualib.config import Settings

if TYPE_CHECKING:
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages Copilot SDK session lifecycle on behalf of an AquaLib workspace.

    Responsibilities:
    - Create a new session with vendor tools, custom agents, and hooks wired up
    - Resume an existing session from ``project.json[session_id]``
    - Collect all vendor skill directories for ``skill_directories``
    - Build the BYOK provider config when ``auth == 'byok'``
    """

    def __init__(
        self,
        client: Any,
        settings: Settings,
        workspace: "WorkspaceManager",
    ) -> None:
        self.client = client
        self.settings = settings
        self.workspace = workspace

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_create_session(self) -> Any:
        """Return an existing session (resumed) or create a brand new one.

        Session ID is persisted in ``project.json`` so that consecutive
        ``aqualib run`` invocations can share context.
        """
        project = self.workspace.load_project()
        existing_id: str | None = project.get("session_id") if project else None

        if existing_id:
            try:
                session = await self._resume_session(existing_id)
                logger.info("Resumed session %s", existing_id)
                return session
            except Exception:
                logger.info("Could not resume session %s – creating a new one.", existing_id)

        return await self._create_session()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _create_session(self) -> Any:
        """Create a new Copilot SDK session and persist its ID."""
        from aqualib.sdk.agents import build_custom_agents
        from aqualib.sdk.hooks import build_hooks
        from aqualib.sdk.system_prompt import build_system_message
        from aqualib.skills.tool_adapter import build_tools_from_skills

        session_id = self._generate_session_id()
        s = self.settings.copilot

        session = await self.client.create_session(
            session_id=session_id,
            model=s.model,
            reasoning_effort=s.reasoning_effort,
            streaming=s.streaming,
            provider=self._build_provider(),
            skill_directories=self._collect_skill_dirs(),
            custom_agents=build_custom_agents(self.settings),
            tools=build_tools_from_skills(self.settings, self.workspace),
            system_message=build_system_message(self.settings, self.workspace),
            hooks=build_hooks(self.settings, self.workspace),
            on_permission_request=self._build_permission_handler(),
            infinite_sessions={
                "enabled": True,
                "background_compaction_threshold": 0.80,
                "buffer_exhaustion_threshold": 0.95,
            },
        )

        self.workspace.update_project({"session_id": session_id})
        logger.info("Created new session %s", session_id)
        return session

    async def _resume_session(self, session_id: str) -> Any:
        """Attempt to resume an existing session by ID."""
        from aqualib.sdk.agents import build_custom_agents
        from aqualib.skills.tool_adapter import build_tools_from_skills

        return await self.client.resume_session(
            session_id,
            on_permission_request=self._build_permission_handler(),
            provider=self._build_provider(),
            tools=build_tools_from_skills(self.settings, self.workspace),
            skill_directories=self._collect_skill_dirs(),
            custom_agents=build_custom_agents(self.settings),
        )

    def _collect_skill_dirs(self) -> list[str]:
        """Collect all vendor skill directories for SDK ``skill_directories``.

        Three-tier priority (highest → lowest):
        1. ``workspace/skills/vendor/``  (per-project customisation)
        2. ``<repo>/vendor/*/``          (repo-shipped submodule libraries)
        """
        dirs: list[str] = []

        ws_vendor = self.workspace.dirs.skills_vendor
        if ws_vendor.exists():
            dirs.append(str(ws_vendor))

        repo_vendor = Path(__file__).resolve().parent.parent.parent.parent / "vendor"
        if repo_vendor.is_dir():
            for lib_dir in sorted(repo_vendor.iterdir()):
                if lib_dir.is_dir() and any(lib_dir.rglob("SKILL.md")):
                    dirs.append(str(lib_dir))

        return dirs

    def _build_provider(self) -> dict | None:
        """Construct the BYOK provider dict, or *None* for GitHub auth."""
        if self.settings.copilot.auth != "byok":
            return None
        p = self.settings.copilot.provider
        if p is None:
            return None

        config: dict[str, Any] = {"type": p.type}
        if p.base_url:
            config["base_url"] = p.base_url
        if p.api_key:
            config["api_key"] = p.api_key
        if p.azure:
            config["azure"] = {"api_version": p.azure.api_version}
        return config

    def _build_permission_handler(self):
        """Return a permission request handler that allows all tool calls."""
        async def on_permission_request(input_data: dict, invocation: Any) -> dict:
            return {"permissionDecision": "allow"}

        return on_permission_request

    def _generate_session_id(self) -> str:
        """Generate a stable, human-readable session ID for this project."""
        project = self.workspace.load_project()
        project_name = (project or {}).get("name", "project")
        slug = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")[:30]
        return f"aqualib-{slug}-{uuid.uuid4().hex[:8]}"
