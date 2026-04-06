"""SessionManager – Copilot SDK session creation and resumption.

Each AquaLib ``aqualib run`` maps to a session tracked in the workspace's
``sessions/`` directory.  The manager tries to resume the active session; if
that fails it creates a fresh one.

Session slug format: ``<name-slug>-<8-char-uuid>``
Session ID format:   ``aqualib-<name-slug>-<8-char-uuid>``
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import print as rprint
from rich.console import Console

from aqualib.config import Settings

if TYPE_CHECKING:
    from aqualib.workspace.manager import WorkspaceManager

_console = Console()

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
        from aqualib.skills.scanner import scan_all_skill_dirs
        from aqualib.skills.tool_adapter import build_tools_from_skills

        s = self.settings.copilot

        # Pre-scan skill metas once; pass to both build_tools_from_skills and
        # build_hooks to avoid triple scanning (once per call site).
        skill_metas = scan_all_skill_dirs(self.settings, self.workspace)

        session = await self.client.create_session(
            session_id=session_id,
            model=s.model,
            reasoning_effort=s.reasoning_effort,
            streaming=s.streaming,
            provider=self._build_provider(),
            skill_directories=self._collect_skill_dirs(),
            custom_agents=build_custom_agents(self.settings, self.workspace, slug),
            tools=build_tools_from_skills(
                self.settings, self.workspace, session_slug=slug, skill_metas=skill_metas
            ),
            system_message=build_system_message(self.settings, self.workspace),
            hooks=build_hooks(self.settings, self.workspace, slug, skill_metas=skill_metas),
            on_permission_request=self._build_permission_handler(),
            on_user_input_request=self._build_user_input_handler(),
            mcp_servers=self._build_mcp_servers(),
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
        from aqualib.sdk.hooks import build_hooks
        from aqualib.sdk.system_prompt import build_system_message
        from aqualib.skills.scanner import scan_all_skill_dirs
        from aqualib.skills.tool_adapter import build_tools_from_skills

        s = self.settings.copilot

        # Pre-scan skill metas once; pass to both build_tools_from_skills and
        # build_hooks to avoid duplicate scanning on resume.
        skill_metas = scan_all_skill_dirs(self.settings, self.workspace)

        return await self.client.resume_session(
            session_id,
            on_permission_request=self._build_permission_handler(),
            on_user_input_request=self._build_user_input_handler(),
            provider=self._build_provider(),
            tools=build_tools_from_skills(
                self.settings, self.workspace, session_slug=slug, skill_metas=skill_metas
            ),
            skill_directories=self._collect_skill_dirs(),
            custom_agents=build_custom_agents(self.settings, self.workspace, slug),
            system_message=build_system_message(self.settings, self.workspace),
            hooks=build_hooks(self.settings, self.workspace, slug, skill_metas=skill_metas),
            model=s.model,
            reasoning_effort=s.reasoning_effort,
            streaming=s.streaming,
            mcp_servers=self._build_mcp_servers(),
            infinite_sessions={
                "enabled": True,
                "background_compaction_threshold": 0.80,
                "buffer_exhaustion_threshold": 0.95,
            },
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
        if p.wire_api:
            config["wire_api"] = p.wire_api
        if p.azure:
            config["azure"] = {"api_version": p.azure.api_version}
        return config

    def _build_mcp_servers(self) -> list[dict] | None:
        """Build the mcp_servers list from settings, or None if MCP is disabled."""
        mcp = self.settings.mcp
        if not mcp.enabled or not mcp.servers:
            return None
        result = []
        for srv in mcp.servers:
            if srv.transport == "stdio" and srv.command:
                entry: dict[str, Any] = {
                    "name": srv.name,
                    "transport": "stdio",
                    "command": srv.command,
                    "args": srv.args,
                }
                if srv.env:
                    entry["env"] = srv.env
                result.append(entry)
            elif srv.transport == "sse" and srv.url:
                result.append({
                    "name": srv.name,
                    "transport": "sse",
                    "url": srv.url,
                })
            else:
                logger.warning(
                    "Skipping MCP server '%s': transport='%s' requires %s",
                    srv.name,
                    srv.transport,
                    "'command'" if srv.transport == "stdio" else "'url'",
                )
        return result or None

    def _build_permission_handler(self):
        """Return a permission request handler with workspace-scoped safety rules.

        - ``write``: deny if the target path is outside the workspace base dir.
        - ``shell``: deny known dangerous command patterns.
        - All other kinds (read, mcp, custom_tool, url, memory, hook): allow.
        """
        base_dir = os.path.normcase(str(self.workspace.dirs.base))

        # Regex patterns for dangerous shell commands; covers common variations
        # (extra spaces, flags, etc.) of each dangerous operation.
        _DANGEROUS_RE = re.compile(
            r"rm\s+-[^\s]*r[^\s]*\s+/"          # rm -rf / and variants
            r"|DROP\s+TABLE"                      # SQL DROP TABLE
            r"|mkfs(\.[a-z0-9]+)?\s"             # mkfs / mkfs.ext4 etc.
            r"|dd\s+if=",                         # dd if= (disk wipe)
            re.IGNORECASE,
        )

        def _get_field(request: Any, field: str, default: str = "") -> str:
            if isinstance(request, dict):
                return request.get(field, default) or default
            return getattr(request, field, default) or default

        def _is_safe_write(request: Any) -> bool:
            path = _get_field(request, "fileName") or _get_field(request, "path") or _get_field(request, "file")
            if not path:
                return True
            try:
                resolved = os.path.normcase(str(Path(path).resolve()))
                return resolved.startswith(base_dir)
            except Exception:
                return False

        def _is_safe_shell(request: Any) -> bool:
            cmd = _get_field(request, "fullCommandText") or _get_field(request, "command") or _get_field(request, "cmd")
            if not cmd:
                return True
            return not bool(_DANGEROUS_RE.search(cmd))

        def _should_allow(request: Any) -> bool:
            kind = _get_field(request, "kind")
            if kind == "write":
                return _is_safe_write(request)
            if kind == "shell":
                return _is_safe_shell(request)
            return True  # read, mcp, custom_tool, url, memory, hook are safe

        try:
            from copilot.session import PermissionRequestResult  # type: ignore[import]

            async def on_permission_request(request: Any, invocation: Any) -> Any:
                if _should_allow(request):
                    return PermissionRequestResult(kind="approved")
                logger.warning(
                    "Permission denied for %s request: %r",
                    _get_field(request, "kind"),
                    request,
                )
                return PermissionRequestResult(kind="deniedByRules")

            return on_permission_request
        except ImportError:
            pass

        async def on_permission_request_dict(input_data: Any, invocation: Any) -> dict:
            if _should_allow(input_data):
                return {"permissionDecision": "allow"}
            logger.warning(
                "Permission denied for %s request: %r",
                _get_field(input_data, "kind"),
                input_data,
            )
            return {"permissionDecision": "deny"}

        return on_permission_request_dict

    def _build_user_input_handler(self):
        """Return an async callback for agent-initiated user input requests."""

        async def on_user_input_request(request: Any, invocation: Any = None) -> dict:
            if isinstance(request, dict):
                question = request.get("question", "")
                choices = request.get("choices", []) or []
            else:
                question = getattr(request, "question", "") or ""
                choices = getattr(request, "choices", []) or []

            rprint(f"\n[bold cyan]🤔 Agent question:[/bold cyan] {question}")
            if choices:
                for i, choice in enumerate(choices, 1):
                    rprint(f"  [dim]{i}. {choice}[/dim]")

            user_input = _console.input("[bold]> [/bold]")
            return {"answer": user_input, "wasFreeform": True}

        return on_user_input_request

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
