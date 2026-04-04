"""Copilot SDK hook implementations for AquaLib.

Provides six hooks that integrate workspace management, audit logging, and
vendor-priority enforcement into the Copilot SDK session lifecycle.

Hook                  | Purpose
--------------------- | -----------------------------------------------------------
on_session_start      | Inject project context + vendor skill overview into session
on_user_prompt_submitted | Record user prompt to context_log
on_pre_tool_use       | Vendor priority check + audit record
on_post_tool_use      | Audit log for tool results
on_session_end        | Save final state to physical files
on_error_occurred     | retry / skip strategy for vendor skill errors
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_hooks(settings: "Settings", workspace: "WorkspaceManager") -> dict:
    """Build and return the complete hook dict for a Copilot SDK session."""
    return {
        "on_session_start": _make_session_start_hook(workspace),
        "on_user_prompt_submitted": _make_prompt_hook(workspace),
        "on_pre_tool_use": _make_pre_tool_hook(settings, workspace),
        "on_post_tool_use": _make_post_tool_hook(workspace),
        "on_session_end": _make_session_end_hook(workspace),
        "on_error_occurred": _make_error_hook(workspace),
    }


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def _make_session_start_hook(workspace: "WorkspaceManager"):
    async def on_session_start(input_data: dict[str, Any], invocation: Any) -> dict | None:
        """Inject project context + vendor skill overview into the session."""
        project = workspace.load_project()
        context_parts: list[str] = []

        if project:
            context_parts.append(f"Project: {project.get('name', 'unknown')}")
            if project.get("summary"):
                context_parts.append(f"History: {project['summary']}")

        # Last 5 context log entries
        entries = workspace.load_context_log()
        task_entries = [e for e in entries if e.get("query")]  # skip hook-audit entries
        if task_entries:
            context_parts.append("Recent tasks:")
            for e in task_entries[-5:]:
                icon = "✅" if e.get("status") in ("approved", "completed") else "⚠️"
                context_parts.append(
                    f"  {icon} \"{e.get('query', '')}\" → {e.get('status', 'unknown')} "
                    f"(skills: {', '.join(e.get('skills_used', []))})"
                )

        # Vendor skill overview (progressive disclosure Level 1)
        try:
            from aqualib.skills.scanner import scan_all_skill_dirs

            skills = scan_all_skill_dirs(workspace.settings, workspace)
            if skills:
                context_parts.append(f"\nAvailable vendor skills ({len(skills)}):")
                for s in skills:
                    context_parts.append(f"  - vendor_{s.name}: {s.description[:80]}")
                context_parts.append(
                    "\nUse 'read_skill_doc' tool for full documentation before invoking."
                )
        except Exception:
            logger.debug("Could not load vendor skills for session context.", exc_info=True)

        if not context_parts:
            return None

        return {"additionalContext": "\n".join(context_parts)}

    return on_session_start


def _make_prompt_hook(workspace: "WorkspaceManager"):
    async def on_user_prompt_submitted(input_data: dict[str, Any], invocation: Any) -> None:
        """Record the user prompt to context_log for project memory."""
        workspace.append_audit_entry({
            "event": "user_prompt",
            "query": input_data.get("prompt", ""),
            "timestamp": input_data.get("timestamp"),
        })
        return None  # do not modify the prompt

    return on_user_prompt_submitted


def _make_pre_tool_hook(settings: "Settings", workspace: "WorkspaceManager"):
    async def on_pre_tool_use(
        input_data: dict[str, Any], invocation: Any
    ) -> dict[str, Any]:
        """Vendor priority check + pre-execution audit record.

        If vendor skills are available but the agent chose a built-in tool,
        return an ``additionalContext`` reminder to steer the agent back.
        """
        tool_name = input_data.get("toolName", "")

        workspace.append_audit_entry({
            "event": "pre_tool_use",
            "tool": tool_name,
            "args_preview": str(input_data.get("toolArgs", {}))[:200],
        })

        # Vendor priority reminder
        if settings.vendor_priority and not tool_name.startswith("vendor_"):
            vendor_skills = [
                t for t in input_data.get("availableTools", [])
                if str(t).startswith("vendor_")
            ]
            if vendor_skills:
                return {
                    "permissionDecision": "allow",
                    "additionalContext": (
                        f"⚠️ VENDOR PRIORITY REMINDER: You are about to use '{tool_name}' "
                        f"but vendor skills are available: {', '.join(str(v) for v in vendor_skills)}. "
                        f"Prefer vendor skills when applicable."
                    ),
                }

        return {"permissionDecision": "allow"}

    return on_pre_tool_use


def _make_post_tool_hook(workspace: "WorkspaceManager"):
    async def on_post_tool_use(input_data: dict[str, Any], invocation: Any) -> None:
        """Record tool execution result to the audit trail."""
        workspace.append_audit_entry({
            "event": "post_tool_use",
            "tool": input_data.get("toolName", ""),
            "success": not input_data.get("toolError"),
            "result_preview": str(input_data.get("toolResult", ""))[:300],
        })
        return None

    return on_post_tool_use


def _make_session_end_hook(workspace: "WorkspaceManager"):
    async def on_session_end(input_data: dict[str, Any], invocation: Any) -> None:
        """Flush and finalise the workspace state after the session ends."""
        workspace.finalize_task()
        return None

    return on_session_end


def _make_error_hook(workspace: "WorkspaceManager"):
    async def on_error_occurred(
        input_data: dict[str, Any], invocation: Any
    ) -> dict[str, str]:
        """Error handling strategy.

        - Vendor skill failure → retry once
        - All other errors    → skip and let the agent try another approach
        """
        error_context = input_data.get("errorContext", "")
        error_msg = input_data.get("error", "")

        workspace.append_audit_entry({
            "event": "error",
            "context": error_context,
            "error": str(error_msg)[:500],
        })

        if "vendor_" in str(error_context):
            return {"errorHandling": "retry"}

        return {"errorHandling": "skip"}

    return on_error_occurred
