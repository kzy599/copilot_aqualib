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
     (a) Use `workspace_search` to discover what data files actually exist in \
the workspace and cross-reference against the user's request. If expected files \
do not exist, ask the user for clarification BEFORE creating a plan.
     (b) Present a plan with: Goal, Data (verified to exist), Steps, Output
     (c) Call `write_plan` to persist the plan
     (d) WAIT for user confirmation before proceeding
   - You are FORBIDDEN from delegating to executor or calling any vendor_* / \
workspace tool without user confirmation of the plan.
   - Confirmation keywords: "go ahead", "execute", "ok", "yes", "确认", "执行", "好的"
   - If the user modifies the plan, update accordingly and re-present.
   - For pure knowledge questions (no tool invocation needed), skip the plan \
entirely and answer directly.

2. **Executor → Reviewer → Plan Revision Pipeline**:
   - After completing a task, delegate to the reviewer agent for quality audit
   - If the reviewer says "needs_revision", address the feedback and re-run
   - If the reviewer says "plan_revision_needed", the plan itself is flawed:
     (a) Read the reviewer's PLAN_QUALITY reason and SUGGESTIONS
     (b) Revise the plan to address the reviewer's concerns
     (c) Call `write_plan` to persist the revised plan
     (d) Present the revised plan to the user for re-confirmation
     (e) After confirmation, re-delegate to the executor with the new plan

3. **Workspace Discipline**:
   - All task outputs go to the session results directory: \
`sessions/<slug>/results/`
   - Standard output structure: `report.md`, `result.json`, `tables/`, \
`reproducibility/`
   - NEVER create arbitrary output directories (e.g. `ebv_output/`, `output/`) \
in the workspace root — always use the session results path provided by the Executor
   - Never modify files in data/ (treat as read-only source data)
   - NEVER fabricate or simulate results — if a skill fails, report the failure \
honestly

4. **Vendor Skill Libraries**:
   - Vendor libraries are independently maintained projects; their CLI, parameters, \
and output formats may change between versions
   - ALWAYS call `read_library_doc` then `read_skill_doc` to learn the current \
usage — do NOT assume a fixed command format or hardcoded examples
   - Vendor skills are preferred over hand-written code: they are tested, \
reproducible, and produce standardised outputs
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
