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

Rules:
0. **Read the Plan**: At the start of every task, read `plan.md` from the session \
directory using the `read_file` tool. This plan was written by the coordinator \
and describes the goal, data, steps, and expected output for this task. \
Follow the plan unless you encounter an error that requires deviation.
1. {vendor_priority} prefer vendor skills (tools prefixed with `vendor_`) over \
built-in tools when there is any possibility of using them.
2. Before invoking a skill, use `read_library_doc` to understand the library architecture, \
then use `read_skill_doc` to read its SKILL.md for parameter details.
3. **Smart Retry on Failure** (CRITICAL):
   - If a vendor skill returns an ERROR, DO NOT immediately retry with the same parameters.
   - STOP and analyse the error message. Identify the root cause:
     • File not found → use `workspace_search` to find correct paths
     • Invalid parameters → re-read SKILL.md via `read_skill_doc`
     • Permission denied → use paths within workspace results/ directory
     • Missing dependency → check SKILL.md install section, try --demo mode
   - Fix the identified issue, then retry with corrected parameters.
   - After 4 failed attempts for the same skill, STOP and report the failure honestly.
   - NEVER fabricate or simulate results when a skill fails.
4. Use `workspace_search` to locate relevant data files before starting.
5. Write all outputs to the workspace results directory.
6. After completing all tasks, explicitly delegate to the reviewer agent by saying: \
"Delegating to reviewer for audit."
"""

_REVIEWER_PROMPT = """\
You are the **Reviewer** agent of the AquaLib framework.

Your responsibilities:
0. **Read the Plan**: At the start of every audit, read `plan.md` from the session \
directory using the `read_file` tool. Verify that the executor's work aligns \
with the planned goal, steps, and expected output.
1. Verify the executor's outputs for correctness and completeness.
2. **Vendor Priority Enforcement**: Check if a vendor skill could have been used \
instead of a built-in tool. If yes, flag it as a violation.
3. Check that all output files exist and contain valid data.
4. Return your verdict in this exact format:

   VERDICT: approved | needs_revision
   VENDOR_PRIORITY: satisfied | violated - [reason]
   SUGGESTIONS: [list]
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

    If *workspace* and *session_slug* are provided, injects each agent's
    role-specific memory (last 5 entries) into their respective prompts.
    """
    vendor_priority_str = "ALWAYS" if settings.vendor_priority else "When appropriate,"

    executor_memory_ctx = ""
    reviewer_memory_ctx = ""

    if workspace and session_slug:
        # Inject Executor memory
        exec_mem = workspace.load_agent_memory(session_slug, "executor")
        if exec_mem.get("entries"):
            recent = exec_mem["entries"][-5:]
            executor_memory_ctx = "\n\nYour previous work in this session:\n"
            for e in recent:
                executor_memory_ctx += (
                    f"- Task: \"{e.get('query', '')}\" → "
                    f"skills: {', '.join(e.get('skills_used', []))} "
                    f"| result: {str(e.get('output_preview', 'N/A'))[:80]}\n"
                )

        # Inject Reviewer memory
        rev_mem = workspace.load_agent_memory(session_slug, "reviewer")
        if rev_mem.get("entries"):
            recent = rev_mem["entries"][-5:]
            reviewer_memory_ctx = "\n\nYour previous audits in this session:\n"
            for e in recent:
                reviewer_memory_ctx += (
                    f"- Task: \"{e.get('query', '')}\" → {e.get('verdict', '?')} "
                    f"| violations: {e.get('violations', [])}\n"
                )

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
            "prompt": _EXECUTOR_PROMPT.format(vendor_priority=vendor_priority_str) + executor_memory_ctx,
            "infer": True,  # SDK auto-selects this agent based on context
        },
        {
            "name": "reviewer",
            "display_name": "Reviewer Agent",
            "description": (
                "Audits the executor's work for correctness and vendor priority compliance. "
                "Called after task execution to validate results."
            ),
            "tools": ["grep", "glob", "view", "read_file", "workspace_search"],  # read-only + workspace search
            "prompt": _REVIEWER_PROMPT + reviewer_memory_ctx,
            "infer": False,  # only explicitly delegated by parent agent
        },
    ]
