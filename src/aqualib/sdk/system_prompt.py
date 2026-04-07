"""System prompt builder for AquaLib Copilot SDK sessions.

Uses the SDK's ``customize`` mode to surgically inject AquaLib identity and
guidelines sections without fully overriding the default Copilot system prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.workspace.manager import WorkspaceManager

# ---------------------------------------------------------------------------
# Content templates
# ---------------------------------------------------------------------------

_AQUALIB_GUIDELINES = """\
## AquaLib Rules

1. **Plan-First**: For tasks needing tools, verify data exists via workspace_search, \
present a plan (Goal/Data/Steps/Output), call write_plan, then **stop and wait for \
user confirmation**. Pure knowledge questions: answer directly.

2. **Delegation — DO NOT execute the plan yourself**: You do NOT have vendor_* \
execution tools. After the user confirms, explicitly delegate to the Executor agent. \
Executor runs plan → Reviewer audits → if plan_revision_needed, revise plan and \
re-confirm with user before re-delegating.

3. **Workspace**: Outputs go to sessions/<slug>/results/. Never modify data/. \
Never fabricate results.

4. **Vendor Skills**: Prefer vendor_* tools. Read docs (read_library_doc → \
read_skill_doc) before invoking.
"""


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_system_message(settings: "Settings", workspace: "WorkspaceManager") -> dict[str, Any]:
    identity_section = (
        "You are AquaLib, a multi-agent scientific research assistant and task planner. "
        "Your primary role is to understand the user's request, formulate an execution plan, "
        "and coordinate between an executor agent (task execution) and a reviewer agent "
        "(quality audit). You have access to specialised vendor skills for scientific "
        "workflows that should be preferred over built-in tools whenever applicable."
    )

    guidelines_section = _AQUALIB_GUIDELINES
    project_context = _build_additional_context(workspace)

    return {
        "mode": "customize",
        "sections": {
            "identity": {"action": "replace", "content": identity_section},
            "guidelines": {"action": "append", "content": guidelines_section},
        },
        "content": project_context if project_context else "",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_additional_context(workspace: "WorkspaceManager") -> str:
    """Build per-project context to include at the end of the system message."""
    parts: list[str] = []

    project = workspace.load_project()
    if project:
        parts.append(f"## Current Project\n\nName: {project.get('name', 'unknown')}")
        if project.get("description"):
            parts.append(f"Description: {project['description']}")
        if project.get("summary"):
            parts.append(f"History: {project['summary']}")

    if not parts:
        return ""

    return "\n\n".join(parts)
