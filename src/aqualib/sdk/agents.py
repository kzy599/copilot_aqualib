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

**Primary guidance**: Trust the plan and follow it step by step. \
{vendor_priority} prefer vendor skills ({vendor_tools}). \
Read docs (read_library_doc → read_skill_doc) before first use of each skill. \
If a vendor call fails twice, fall back to code honestly.

**Pre-flight sanity check** (lightweight, NOT a full re-plan): \
Before executing, quickly verify: do input files exist? Are obvious parameters \
clearly wrong (e.g. wrong genome, missing required args)? If something is clearly \
wrong, adapt and log why — don't re-plan from scratch.

**Key principle**: trust the plan, fix only obvious issues, log any deviations. \
This avoids token waste from excessive second-guessing while still catching real problems.

Outputs → `{session_results_dir}`

Limits: max 25 tool calls. At 20, wrap up.

When done, emit an EXECUTION_REPORT and then "Delegating to reviewer for audit.":
PRE_FLIGHT: [pass | issues found - description]
STEPS_COMPLETED: [n/total]
TOTAL_VENDOR_CALLS: [n]
ERRORS_ENCOUNTERED: [none | description]
PLAN_DEVIATIONS: [none | description of what was changed from the plan and why]
"""

_REVIEWER_PROMPT = """\
You are the **Reviewer**, an independent auditor.

1. Read plan.md from the session directory (mandatory).
2. Check injected memory: Executor's EXECUTION_REPORT + your previous verdicts.
3. Audit plan quality: are steps logical, skills appropriate, data real?
4. Audit execution: did Executor follow the plan? Check outputs exist and are valid.
5. Check vendor priority compliance.
6. If PLAN_DEVIATIONS were reported, judge whether each deviation was justified \
(e.g. fixing a real error) or unjustified (e.g. ignoring a valid step).

Verdict format:
VERDICT: approved | needs_revision | plan_revision_needed
VENDOR_PRIORITY: satisfied | violated - [reason]
PLAN_QUALITY: valid | violated - [reason] | revision_needed - [reason]
PLAN_ADHERENCE: followed | violated - [reason] | adapted_justified - [reason]
SUGGESTIONS: [list]

Use plan_revision_needed when the plan itself is fundamentally flawed.
Use adapted_justified when Executor deviated from the plan but the deviation was \
warranted (e.g. fixing a clear input error, missing file, or wrong parameter).
"""


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_custom_agents(
    settings: "Settings",
    workspace: "WorkspaceManager | None" = None,
    session_slug: str | None = None,
    skill_metas: "list | None" = None,
    all_tool_names: "list[str] | None" = None,
) -> list[dict]:
    """Return the Copilot SDK ``custom_agents`` list (executor + reviewer).

    If *workspace* and *session_slug* are provided, injects reviewer
    role-specific memory (last 5 entries + recent executor vendor tool results)
    into the reviewer's prompt. The executor does NOT get memory injection
    because it shares conversation history with the Planner.

    If *skill_metas* is provided (pre-scanned), it is used to build the
    executor's explicit tool list so the sub-agent can see vendor_* tools.

    If *all_tool_names* is provided, it takes precedence: Executor gets all
    custom tool names (auto-propagated from session tools), and Reviewer gets
    the subset that excludes ``vendor_*`` prefixed tools.  This prevents manual
    list maintenance when new tools (e.g. ``rag_search``) are added.
    """
    vendor_priority_str = "ALWAYS" if settings.vendor_priority else "When appropriate,"

    # Build vendor tool name list for executor prompt and tools list.
    vendor_tool_names = [f"vendor_{m.name}" for m in (skill_metas or [])]
    vendor_tools_display = ", ".join(vendor_tool_names) if vendor_tool_names else "none discovered"

    # Determine executor tool list:
    # - If all_tool_names was passed (auto-propagated from session), use it directly.
    # - Otherwise fall back to constructing the list manually from skill_metas + hardcoded utils.
    if all_tool_names is not None:
        executor_tool_names = list(all_tool_names)
    else:
        # Fallback: manual list (vendor tools + utility tools).
        # tools: None would mean NO tools in the Copilot SDK; must be explicit.
        executor_tool_names = [
            *vendor_tool_names,
            "workspace_search",
            "read_library_doc",
            "read_skill_doc",
            "write_plan",
        ]

    # Reviewer gets all non-execution tools: filter out vendor_* (execution) tools.
    # SDK built-in tools (shell, grep, glob, etc.) are always available regardless.
    if all_tool_names is not None:
        reviewer_tool_names = [t for t in all_tool_names if not t.startswith("vendor_")]
    else:
        reviewer_tool_names = ["workspace_search", "read_skill_doc", "read_library_doc"]

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
            reviewer_memory_ctx_parts.append(
                f"  PLAN_DEVIATIONS: {latest.get('plan_deviations', '?')}"
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
            "tools": executor_tool_names,
            "prompt": _EXECUTOR_PROMPT.format(
                vendor_priority=vendor_priority_str,
                vendor_tools=vendor_tools_display,
                session_results_dir=session_results_dir,
            ),
            "infer": False,  # delegation must be explicit from Planner
        },
        {
            "name": "reviewer",
            "display_name": "Reviewer Agent",
            "description": (
                "Audits the executor's work for correctness and vendor priority compliance. "
                "Called after task execution to validate results."
            ),
            "tools": reviewer_tool_names,
            "prompt": _REVIEWER_PROMPT + reviewer_memory_ctx,
            "infer": False,  # only explicitly delegated by parent agent
        },
    ]
