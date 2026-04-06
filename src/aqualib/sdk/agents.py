"""Custom agent definitions for AquaLib's Executor + Reviewer pipeline.

Uses the Copilot SDK ``custom_agents`` mechanism so the CLI's built-in
ReAct loop can automatically delegate to the appropriate sub-agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.workspace.manager import WorkspaceManager

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EXECUTOR_PROMPT = """\
You are the **Executor**. The plan is in conversation history.

Execute it. {vendor_priority} prefer vendor skills. Read docs (read_library_doc → \
read_skill_doc) before first use of each skill. If a vendor call fails twice, \
fall back to code honestly.

Outputs → `{session_results_dir}`

Limits: max 20 tool calls. At 15, wrap up. Don't repeat Planner's work.

When done: brief EXECUTION_REPORT (PRE_FLIGHT/STEPS_COMPLETED/TOTAL_VENDOR_CALLS/\
ERRORS_ENCOUNTERED) then "Delegating to reviewer for audit."
"""

_REVIEWER_PROMPT = """\
You are the **Reviewer**, an independent auditor.

1. Read plan.md from the session directory (mandatory).
2. Check injected memory: Executor's EXECUTION_REPORT + your previous verdicts.
3. Audit plan quality: are steps logical, skills appropriate, data real?
4. Audit execution: did Executor follow the plan? Check outputs exist and are valid.
5. Check vendor priority compliance.

Verdict format:
VERDICT: approved | needs_revision | plan_revision_needed
VENDOR_PRIORITY: satisfied | violated - [reason]
PLAN_QUALITY: valid | violated - [reason] | revision_needed - [reason]
PLAN_ADHERENCE: followed | violated - [reason]
SUGGESTIONS: [list]

Use plan_revision_needed when the plan itself is fundamentally flawed.
"""


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_custom_agents(
    settings: "Settings",
    workspace: "WorkspaceManager | None" = None,
    session_slug: str | None = None,
) -> list[dict]:
    """Return the Copilot SDK ``custom_agents`` list (executor + reviewer).

    If *workspace* and *session_slug* are provided, injects reviewer
    role-specific memory (last 5 entries + recent executor vendor tool results)
    into the reviewer's prompt. The executor does NOT get memory injection
    because it shares conversation history with the Planner.
    """
    vendor_priority_str = "ALWAYS" if settings.vendor_priority else "When appropriate,"

    # Compute the session results directory path for the executor prompt.
    if workspace and session_slug:
        session_results_dir = str(workspace.session_results_dir(session_slug))
    elif workspace and session_slug is None:
        # Fallback: use workspace base with generic placeholder
        session_results_dir = str(workspace.dirs.base / "sessions" / "<session_slug>" / "results")
    else:
        session_results_dir = "sessions/<session_slug>/results"

    reviewer_memory_ctx = ""

    if workspace and session_slug:
        # Inject Reviewer memory: own previous verdicts + executor vendor actions
        rev_mem = workspace.load_agent_memory(session_slug, "reviewer")
        exec_mem = workspace.load_agent_memory(session_slug, "executor")

        reviewer_memory_ctx_parts: list[str] = []

        # Bridge the most recent EXECUTION_REPORT to reviewer context
        exec_report_entries = [
            e for e in exec_mem.get("entries", [])
            if e.get("event") == "execution_report"
        ]
        if exec_report_entries:
            latest = exec_report_entries[-1]
            reviewer_memory_ctx_parts.append("Executor's latest execution report:")
            reviewer_memory_ctx_parts.append(
                f"  PRE_FLIGHT: {latest.get('pre_flight', '?')}"
            )
            reviewer_memory_ctx_parts.append(
                f"  STEPS_COMPLETED: {latest.get('steps_completed', '?')}"
            )
            reviewer_memory_ctx_parts.append(
                f"  TOTAL_VENDOR_CALLS: {latest.get('total_vendor_calls', '?')}"
            )
            reviewer_memory_ctx_parts.append(
                f"  SANITY_CHECKS: {latest.get('sanity_checks', '?')}"
            )
            reviewer_memory_ctx_parts.append(
                f"  ERRORS_ENCOUNTERED: {latest.get('errors_encountered', '?')}"
            )

        # Reviewer's own previous verdicts
        if rev_mem.get("entries"):
            recent = rev_mem["entries"][-5:]
            reviewer_memory_ctx_parts.append("Your previous verdicts in this session:")
            for e in recent:
                reviewer_memory_ctx_parts.append(
                    f"  - Task: \"{e.get('query', '')}\" → {e.get('verdict', '?')} "
                    f"| violations: {e.get('violations', [])}"
                )

        if reviewer_memory_ctx_parts:
            reviewer_memory_ctx = "\n\n" + "\n".join(reviewer_memory_ctx_parts)

    return [
        {
            "name": "executor",
            "display_name": "Executor Agent",
            "description": (
                "Executes the user's task by invoking skill tools (vendor_* prefixed) "
                "and built-in tools. Handles ALL task execution including sequence alignment, "
                "drug interaction analysis, data processing, and any scientific workflow. "
                "Must be delegated to for any task that requires tool invocation."
            ),
            "tools": None,  # all tools available
            "prompt": _EXECUTOR_PROMPT.format(
                vendor_priority=vendor_priority_str,
                session_results_dir=session_results_dir,
            ),
            "infer": True,  # SDK auto-selects this agent based on context
        },
        {
            "name": "reviewer",
            "display_name": "Reviewer Agent",
            "description": (
                "Audits the executor's work for correctness and vendor priority compliance. "
                "Called after task execution to validate results."
            ),
            "tools": ["grep", "glob", "view", "read_file", "workspace_search", "read_skill_doc"],
            "prompt": _REVIEWER_PROMPT + reviewer_memory_ctx,
            "infer": False,  # only explicitly delegated by parent agent
        },
    ]
