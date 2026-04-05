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

1. **Plan-First Workflow** (MANDATORY):
   - For ANY task that involves tool execution or vendor skills, you MUST:
     (a) Present a plan with: Goal, Data, Steps, Output
     (b) Call `write_plan` to persist the plan
     (c) WAIT for user confirmation before proceeding
   - You are FORBIDDEN from delegating to executor or calling any vendor_* / \
workspace tool without user confirmation of the plan.
   - Confirmation keywords: "go ahead", "execute", "ok", "yes", "确认", "执行", "好的"
   - If the user modifies the plan, update accordingly and re-present.
   - For pure knowledge questions (no tool invocation needed), skip the plan \
entirely and answer directly.

2. **Vendor Priority**: {vendor_priority} prefer vendor tools (prefixed `vendor_`) over \
built-in tools when there is any possibility of using them.

3. **Progressive Disclosure**:
   - FIRST use `read_library_doc` to read the skill library's top-level docs \
(AGENTS.md, catalog.json) to understand the full architecture and CLI commands
   - THEN use `read_skill_doc` to read specific SKILL.md before invoking a vendor skill
   - Use `workspace_search` to locate relevant data files before starting

4. **Executor → Reviewer Pipeline**:
   - After completing a task, delegate to the reviewer agent for quality audit
   - If the reviewer says "needs_revision", address the feedback and re-run

5. **Workspace Discipline**:
   - All outputs go to the workspace results directory
   - Never modify files in data/ (treat as read-only source data)
   - Vendor skill invocations are automatically traced in vendor_traces/

6. **Skill Failure Handling**:
   - When a skill call fails, the framework will automatically retry up to 4 times.
   - Each retry attempt MUST use different parameters or approach based on error analysis.
   - DO NOT retry blindly with the same parameters — that wastes all retry budget.
   - After all 4 retries are exhausted, report the failure to the user with:
     (a) What was attempted
     (b) What errors occurred
     (c) Suggested manual actions the user can take
   - NEVER fabricate, simulate, or hallucinate results when a skill fails.
   - NEVER say "I'll use simulated data as a backup" — ask the user for help instead.
"""


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_system_message(settings: "Settings", workspace: "WorkspaceManager") -> dict[str, Any]:
    vendor_priority_str = "ALWAYS" if settings.vendor_priority else "When appropriate,"

    identity_section = (
        "You are AquaLib, a multi-agent scientific research assistant and task planner. "
        "Your primary role is to understand the user's request, formulate an execution plan, "
        "and coordinate between an executor agent (task execution) and a reviewer agent "
        "(quality audit). You have access to specialised vendor skills for scientific "
        "workflows that should be preferred over built-in tools whenever applicable."
    )

    guidelines_section = _AQUALIB_GUIDELINES.format(vendor_priority=vendor_priority_str)
    project_context = _build_additional_context(workspace)

    return {
        "mode": "override",
        "content": (
            identity_section + "\n\n" + guidelines_section
            + ("\n\n" + project_context if project_context else "")
        ),
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
