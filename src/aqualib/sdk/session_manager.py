"""SessionManager – Copilot SDK session creation and resumption.

Each AquaLib ``aqualib run`` maps to a session tracked in the workspace's
``sessions/`` directory.  The manager tries to resume the active session; if
that fails it creates a fresh one.

Session slug format: ``<name-slug>-<8-char-uuid>``
Session ID format:   ``aqualib-<name-slug>-<8-char-uuid>``
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
    - Resume an existing session from the active session in the workspace
    - Collect all vendor skill directories for ``skill_directories``
    - Build the BYOK provider config when ``auth == 'byok'``
    """

    def __init__(
        self,
        client: Any,
        settings: Settings,
        workspace: "WorkspaceManager",
        session_slug: str | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.workspace = workspace
        self._session_slug = session_slug  # None = use active session or create new

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_create_session(self) -> tuple[Any, str]:
        """Return (sdk_session, session_slug) — resuming or creating as needed."""
        ws = self.workspace

        if self._session_slug:
            # Explicit session specified (e.g. --session flag)
            session_meta = ws.find_session_by_prefix(self._session_slug)
            if session_meta:
                slug = session_meta["slug"]
                try:
                    session = await self._resume_sdk_session(session_meta["session_id"], slug)
                    logger.info("Resumed session %s", slug)
                    return session, slug
                except Exception:
                    logger.info("Could not resume session %s – creating new.", slug)
            # If not found or resume failed, create new with the given name
            new_meta = ws.create_session(name=self._session_slug)
            session = await self._create_sdk_session(new_meta["slug"], new_meta["session_id"])
            return session, new_meta["slug"]

        # Try to resume the active session
        active = ws.get_active_session()
        if active:
            try:
                session = await self._resume_sdk_session(active["session_id"], active["slug"])
                logger.info("Resumed active session %s", active["slug"])
                return session, active["slug"]
            except Exception:
                logger.info("Could not resume active session %s – creating new.", active["slug"])

        # Create a brand new session
        new_meta = ws.create_session()
        session = await self._create_sdk_session(new_meta["slug"], new_meta["session_id"])
        return session, new_meta["slug"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _create_sdk_session(self, slug: str, session_id: str) -> Any:
        """Create a new Copilot SDK session with all hooks and tools wired up."""
        from aqualib.sdk.agents import build_custom_agents
        from aqualib.sdk.hooks import build_hooks
        from aqualib.sdk.system_prompt import build_system_message
        from aqualib.skills.tool_adapter import build_tools_from_skills

        s = self.settings.copilot

        session = await self.client.create_session(
            session_id=session_id,
            model=s.model,
            reasoning_effort=s.reasoning_effort,
            streaming=s.streaming,
            provider=self._build_provider(),
            skill_directories=self._collect_skill_dirs(),
            custom_agents=build_custom_agents(self.settings, self.workspace, slug),
            tools=build_tools_from_skills(self.settings, self.workspace),
            system_message=build_system_message(self.settings, self.workspace),
            hooks=build_hooks(self.settings, self.workspace, slug),
            on_permission_request=self._build_permission_handler(),
            infinite_sessions={
                "enabled": True,
                "background_compaction_threshold": 0.80,
                "buffer_exhaustion_threshold": 0.95,
            },
        )

        logger.info("Created new SDK session %s (slug=%s)", session_id, slug)
        return session

    async def _resume_sdk_session(self, session_id: str, slug: str) -> Any:
        """Attempt to resume an existing SDK session by ID."""
        from aqualib.sdk.agents import build_custom_agents
        from aqualib.skills.tool_adapter import build_tools_from_skills

        return await self.client.resume_session(
            session_id,
            on_permission_request=self._build_permission_handler(),
            provider=self._build_provider(),
            tools=build_tools_from_skills(self.settings, self.workspace),
            skill_directories=self._collect_skill_dirs(),
            custom_agents=build_custom_agents(self.settings, self.workspace, slug),
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

    # ------------------------------------------------------------------
    # Legacy compatibility shim
    # ------------------------------------------------------------------

    async def _create_session(self) -> Any:
        """Legacy method — creates a session using the old single-session approach.

        .. deprecated::
            Use ``get_or_create_session()`` which returns ``(session, slug)``.
        """
        ws = self.workspace
        project = ws.load_project()

        # Migrate old session_id if present
        old_session_id: str | None = (project or {}).get("session_id")
        if old_session_id:
            # Extract a slug from the old session_id
            slug_part = old_session_id.removeprefix("aqualib-")
            new_meta = ws.create_session(name=slug_part)
            return await self._create_sdk_session(new_meta["slug"], new_meta["session_id"])

        new_meta = ws.create_session()
        return await self._create_sdk_session(new_meta["slug"], new_meta["session_id"])

    def _generate_session_id(self) -> str:
        """Generate a stable, human-readable session ID for this project."""
        project = self.workspace.load_project()
        project_name = (project or {}).get("name", "project")
        slug = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")[:30]
        return f"aqualib-{slug}-{uuid.uuid4().hex[:8]}"
