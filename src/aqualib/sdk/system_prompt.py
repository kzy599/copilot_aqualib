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
## AquaLib Framework Rules

1. **Plan-First Workflow**:
   - When a user requests a task that involves tool execution or vendor skills, \
FIRST present a concise plan to the user. The plan should include:
     (a) Goal: what the task aims to achieve
     (b) Data: which input files or data will be used
     (c) Steps: ordered list of skills/tools to invoke
     (d) Output: expected deliverables and where they will be saved
   - Wait for the user to confirm (e.g. "go ahead", "execute", "ok", "yes", \
"确认", "执行", "好的") before delegating to the executor agent.
   - If the user modifies the plan, update accordingly and re-present.
   - If the user's request is a simple, unambiguous single-step task (e.g. \
"align MVKLF and MVKLT"), you may present the plan and execute in the \
same turn without waiting — but still write the plan first.
   - BEFORE delegating to any sub-agent, ALWAYS call the `write_plan` tool to \
persist the plan to disk. The executor and reviewer will read this plan.
   - For pure knowledge questions (no tool invocation needed), skip the plan \
entirely and answer directly.

2. **Vendor Priority**: {vendor_priority} prefer vendor tools (prefixed `vendor_`) over \
built-in tools when there is any possibility of using them.

3. **Progressive Disclosure**:
   - Check the available vendor skill list at the start of every task
   - Use `read_skill_doc` to read the full SKILL.md before invoking a vendor skill
   - Use `workspace_search` to locate relevant data files before starting

4. **Executor → Reviewer Pipeline**:
   - After completing a task, delegate to the reviewer agent for quality audit
   - If the reviewer says "needs_revision", address the feedback and re-run

5. **Workspace Discipline**:
   - All outputs go to the workspace results directory
   - Never modify files in data/ (treat as read-only source data)
   - Vendor skill invocations are automatically traced in vendor_traces/
"""


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_system_message(settings: "Settings", workspace: "WorkspaceManager") -> dict[str, Any]:
    """Build the SDK ``system_message`` dict using ``customize`` mode.

    The ``customize`` mode lets us:
    - Replace the ``identity`` section with an AquaLib-specific description
    - Append AquaLib guidelines to the ``guidelines`` section
    - Inject per-project context in the ``content`` field
    """
    vendor_priority_str = "ALWAYS" if settings.vendor_priority else "When appropriate,"

    return {
        "mode": "customize",
        "sections": {
            "identity": {
                "action": "replace",
                "content": (
                    "You are AquaLib, a multi-agent scientific research assistant and task planner. "
                    "Your primary role is to understand the user's request, formulate an execution plan, "
                    "and coordinate between an executor agent (task execution) and a reviewer agent "
                    "(quality audit). You have access to specialised vendor skills for scientific "
                    "workflows that should be preferred over built-in tools whenever applicable."
                ),
            },
            "guidelines": {
                "action": "append",
                "content": _AQUALIB_GUIDELINES.format(vendor_priority=vendor_priority_str),
            },
        },
        "content": _build_additional_context(workspace),
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
