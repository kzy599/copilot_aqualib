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
You are the **Executor** agent of the AquaLib framework.

The plan and data file locations should be visible in the conversation history above \
(written by the Planner). If the plan is available in history, do NOT re-read plan.md \
with read_file and do NOT re-run workspace_search to re-verify files — the Planner \
already did this. If you cannot find the plan in conversation history, read plan.md \
using read_file before proceeding.

Rules:
1. {vendor_priority} prefer vendor skills (tools prefixed with `vendor_`) over \
built-in tools when there is any possibility of using them.
2. **Read Docs Then Construct Command** (CRITICAL):
   - ALWAYS call `read_library_doc` first to understand the library's CLI architecture \
and the exact command format used by the vendor library.
   - Then call `read_skill_doc` to read the specific skill's SKILL.md for parameter details.
   - Construct the FULL shell command string in the `command` field based on what you read. \
Do NOT guess CLI syntax — it varies per vendor library.
   - Example: after reading docs, set command to \
`"python clawbio.py run --input data.csv --output results.json --trait-pos 3"`.
3. **Smart Retry on Failure** (CRITICAL):
   - If a vendor skill returns an ERROR, re-read the docs via `read_skill_doc` to \
understand the correct CLI format.
   - Construct a different command based on the error and re-read documentation.
   - After 2 failed attempts for the same skill, STOP and report the failure honestly.
   - NEVER fabricate or simulate results when a skill fails.
4. Write all outputs to the workspace results directory.
5. After completing all tasks, explicitly delegate to the reviewer agent by saying: \
"Delegating to reviewer for audit."
6. **Plan Revision Escalation** (CRITICAL):
   - If the reviewer returns VERDICT: plan_revision_needed, do NOT retry execution.
   - Instead, report the reviewer's feedback to the Planner (the coordinator/parent agent) \
by saying: "PLAN REVISION REQUESTED: The reviewer has identified fundamental issues \
with the plan. Reviewer feedback: [include PLAN_QUALITY and SUGGESTIONS from the verdict]. \
Please revise the plan and re-delegate."
   - The Planner will then revise plan.md and re-delegate to you with the updated plan.
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
1. **Load Your Memory**: Your previous verdicts may be provided above. Use them \
to detect recurring issues but do NOT let them bias this audit.
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
3. **Plan Adherence Audit**: Compare the executor's actions (visible in your memory \
above as vendor tool results) against the steps listed in plan.md. Verify that:
   - Every step in the plan was attempted by the executor.
   - The tools/skills used match what the plan specified.
   - Output files produced correspond to the expected output in the plan.
   If any planned step was skipped, used the wrong skill, or produced no output, \
flag it as a violation.
4. Verify the executor's outputs for correctness and completeness.
5. **Vendor Priority Enforcement**: Check if a vendor skill could have been used \
instead of a built-in tool. If yes, flag it as a violation.
6. Check that all output files exist and contain valid data.
7. Return your verdict in this exact format:

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

    reviewer_memory_ctx = ""

    if workspace and session_slug:
        # Inject Reviewer memory: own previous verdicts + executor vendor actions
        rev_mem = workspace.load_agent_memory(session_slug, "reviewer")
        exec_mem = workspace.load_agent_memory(session_slug, "executor")

        reviewer_memory_ctx_parts: list[str] = []

        # Executor's recent vendor tool results bridged to reviewer
        exec_vendor_entries = [
            e for e in exec_mem.get("entries", [])
            if e.get("event") == "vendor_tool_use"
        ]
        if exec_vendor_entries:
            reviewer_memory_ctx_parts.append("Executor's recent vendor tool calls:")
            for e in exec_vendor_entries[-5:]:
                status = "✅ success" if e.get("success") else "❌ failed"
                reviewer_memory_ctx_parts.append(
                    f"  - {e.get('tool', '?')} → {status}: "
                    f"{str(e.get('output_preview', 'N/A'))[:80]}"
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
            "prompt": _EXECUTOR_PROMPT.format(vendor_priority=vendor_priority_str),
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
