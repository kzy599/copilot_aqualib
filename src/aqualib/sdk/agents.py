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
You are the **Executor** agent.

The Planner's plan and data context are in the conversation history above.
Do NOT re-read plan.md, do NOT re-run workspace_search — the Planner already did that.

## Your Job

Execute the plan. {vendor_priority} prefer vendor skills (`vendor_*` tools) over \
built-in tools.

To use a vendor skill: call `read_library_doc` to understand the library's CLI \
architecture, then `read_skill_doc` for the specific skill's parameters. Construct \
the full shell command from what you learned — do NOT guess or use hardcoded examples. \
If a vendor call fails, re-read docs, adjust the command, and retry once. After 2 \
failures, fall back to code — but say so honestly. NEVER fabricate or simulate results.

## Output Directory

All outputs go to: `{session_results_dir}`
Use `report.md` + `result.json` + `tables/` + `reproducibility/` structure.

## Limits

- Max 20 tool calls total. At 15, wrap up immediately.
- Do NOT repeat Planner's work (no redundant workspace_search or plan.md reads).
- Call read_library_doc and read_skill_doc once per skill; re-read only after failure.

When done, produce a brief EXECUTION_REPORT (PRE_FLIGHT / STEPS_COMPLETED / \
OUTPUT_FILES / TOTAL_VENDOR_CALLS / ERRORS_ENCOUNTERED) and say \
"Delegating to reviewer for audit."
"""

_REVIEWER_PROMPT = """\
You are the **Reviewer** agent of the AquaLib framework.

You are a FRESH, independent auditor. You do NOT share the executor's full \
conversation thread, but you receive a summary of vendor tool results via your memory \
(see below). You MUST form your own independent judgments by reading plan.md and \
checking outputs directly.

Your responsibilities:
0. **Read the Plan First**: Call `read_file` to read `plan.md` from the session \
directory. This is mandatory — you cannot audit without the plan.
1. **Load Your Memory**: Two memory sources are injected above:
   - **Executor's Execution Report** (PRE_FLIGHT, STEPS_COMPLETED, OUTPUT_FILES, \
SANITY_CHECKS, ERRORS_ENCOUNTERED) — produced by the Executor for this cycle.
   - **Your Previous Verdicts** from prior review cycles — use them to detect \
recurring issues but do NOT let them bias this audit.
   If no execution report is present in the injected memory, flag PLAN_ADHERENCE \
as violated with reason "Executor did not produce execution report".
2. **Plan Reasonableness Audit** (CRITICAL):
   Evaluate whether the plan ITSELF is sound and achievable:
   - Are the planned steps logical and in the correct order?
   - Are the chosen skills/tools appropriate for each step? Use `read_skill_doc` \
to verify capability.
   - Are referenced data files real? Use `workspace_search` to check.
   - Are the expected outputs realistic given the inputs and tools?
   - Is there a better approach or skill that the plan overlooked?
   If the plan is fundamentally flawed (wrong approach, impossible steps, \
mismatched skills, missing prerequisites), set PLAN_QUALITY to "revision_needed" \
with a clear explanation. This will cause the plan to be sent back to the \
Planner for revision.
3. **Plan Adherence Audit**: Compare the EXECUTION_REPORT fields (PRE_FLIGHT, \
STEPS_COMPLETED, OUTPUT_FILES, SANITY_CHECKS) against the steps listed in plan.md. \
Verify that:
   - Every step in the plan was attempted (check STEPS_COMPLETED ratio).
   - PRE_FLIGHT passed before execution began.
   - Output files produced correspond to the expected output in the plan \
(check OUTPUT_FILES).
   - SANITY_CHECKS show no unresolved failures.
   If any planned step was skipped, the wrong skill was used, or no output was \
produced, flag it as a violation.
4. Verify the executor's outputs for correctness and completeness. Check that all \
output files exist and contain valid, non-empty data.
5. **Vendor Priority Enforcement**: Check if a vendor skill could have been used \
instead of a built-in tool. If yes, flag it as a violation.
6. Return your verdict in this exact format:

   VERDICT: approved | needs_revision | plan_revision_needed
   VENDOR_PRIORITY: satisfied | violated - [reason]
   PLAN_QUALITY: valid | violated - [reason] | revision_needed - [reason]
   PLAN_ADHERENCE: followed | violated - [reason]
   SUGGESTIONS: [list]

Use VERDICT: plan_revision_needed when the plan ITSELF is the root cause of \
failure — the executor cannot succeed without a better plan. Include specific \
suggestions for how the Planner should revise the plan.
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
