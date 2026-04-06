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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

_MAX_ADDITIONAL_CONTEXT_CHARS = 2000  # Hard cap for additionalContext injected at session start
# ---------------------------------------------------------------------------


def _save_reviewer_memory(
    workspace: "WorkspaceManager",
    session_slug: str,
    result_text: str,
) -> None:
    """Extract reviewer verdict fields from *result_text* and persist to memory.

    Parses VERDICT, VENDOR_PRIORITY, PLAN_QUALITY, PLAN_ADHERENCE, and SUGGESTIONS
    using regex so the reviewer's decisions accumulate in ``memory/reviewer.json``
    rather than being lost at session end.

    PLAN_QUALITY supports three states:
    - ``valid`` — plan is sound
    - ``violated`` — minor issues (missing files, bad params)
    - ``revision_needed`` — the plan is fundamentally flawed and must be revised
      by the Planner before re-execution
    """
    verdict_match = re.search(r"VERDICT\s*:\s*(\S+)", result_text, re.IGNORECASE)
    vendor_match = re.search(r"VENDOR_PRIORITY\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE)
    quality_match = re.search(r"PLAN_QUALITY\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE)
    adherence_match = re.search(r"PLAN_ADHERENCE\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE)
    suggestions_match = re.search(r"SUGGESTIONS\s*:\s*(.+?)(?:\n\n|$)", result_text, re.IGNORECASE | re.DOTALL)

    # Warn when the output doesn't match the expected reviewer format at all
    if not verdict_match:
        logger.warning("Reviewer memory: could not parse VERDICT from result text")
    if not vendor_match:
        logger.debug("Reviewer memory: could not parse VENDOR_PRIORITY from result text")
    if not quality_match:
        logger.debug("Reviewer memory: could not parse PLAN_QUALITY from result text")
    if not adherence_match:
        logger.debug("Reviewer memory: could not parse PLAN_ADHERENCE from result text")

    entry: dict[str, Any] = {
        "verdict": verdict_match.group(1).strip() if verdict_match else "unknown",
        "vendor_priority": vendor_match.group(1).strip() if vendor_match else "unknown",
        "plan_quality": quality_match.group(1).strip() if quality_match else "unknown",
        "plan_adherence": adherence_match.group(1).strip() if adherence_match else "unknown",
        "suggestions": suggestions_match.group(1).strip() if suggestions_match else "",
        "violations": [],
    }

    # Collect violations: "violated" = minor issue; "revision_needed" = plan must be revised
    if re.match(r"violated", entry["vendor_priority"], re.IGNORECASE):
        entry["violations"].append(f"vendor_priority: {entry['vendor_priority']}")
    if re.match(r"violated", entry["plan_quality"], re.IGNORECASE):
        entry["violations"].append(f"plan_quality: {entry['plan_quality']}")
    if re.match(r"revision_needed", entry["plan_quality"], re.IGNORECASE):
        entry["violations"].append(f"plan_quality: {entry['plan_quality']}")
    if re.match(r"violated", entry["plan_adherence"], re.IGNORECASE):
        entry["violations"].append(f"plan_adherence: {entry['plan_adherence']}")

    workspace.append_agent_memory_entry(session_slug, "reviewer", entry)


def _save_execution_report_memory(
    workspace: "WorkspaceManager",
    session_slug: str,
    result_text: str,
) -> None:
    """Parse EXECUTION_REPORT from executor output and save to both executor and reviewer memory.

    Parses five scalar fields — PRE_FLIGHT, STEPS_COMPLETED, TOTAL_VENDOR_CALLS,
    ERRORS_ENCOUNTERED, and SANITY_CHECKS — using regex, then persists as an
    ``"execution_report"`` event in both executor and reviewer memory so the
    reviewer receives a structured summary instead of raw 200-char output_preview
    snippets.  Multi-line sub-sections (STEP_DETAILS, OUTPUT_FILES) are intentionally
    omitted to keep the memory entry compact.
    """
    pre_flight_match = re.search(r"PRE_FLIGHT\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE)
    steps_match = re.search(r"STEPS_COMPLETED\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE)
    vendor_calls_match = re.search(
        r"TOTAL_VENDOR_CALLS\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE
    )
    errors_match = re.search(
        r"ERRORS_ENCOUNTERED\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE
    )
    sanity_match = re.search(r"SANITY_CHECKS\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE)

    entry: dict[str, Any] = {
        "event": "execution_report",
        "pre_flight": pre_flight_match.group(1).strip() if pre_flight_match else "unknown",
        "steps_completed": steps_match.group(1).strip() if steps_match else "unknown",
        "total_vendor_calls": vendor_calls_match.group(1).strip() if vendor_calls_match else "unknown",
        "errors_encountered": errors_match.group(1).strip() if errors_match else "unknown",
        "sanity_checks": sanity_match.group(1).strip() if sanity_match else "unknown",
    }

    workspace.append_agent_memory_entry(session_slug, "executor", entry)
    workspace.append_agent_memory_entry(session_slug, "reviewer", entry)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_hooks(
    settings: "Settings",
    workspace: "WorkspaceManager",
    session_slug: str | None = None,
    skill_metas: "list | None" = None,
) -> dict:
    """Build and return the complete hook dict for a Copilot SDK session."""
    # Shared state: tracks which doc tools the model has called this session.
    # Used by the doc-first gate to enforce read_library_doc/read_skill_doc
    # before any vendor_* tool invocation.
    _doc_tools_called: set[str] = set()

    return {
        "on_session_start": _make_session_start_hook(workspace, skill_metas=skill_metas),
        "on_user_prompt_submitted": _make_prompt_hook(workspace, session_slug),
        "on_pre_tool_use": _make_pre_tool_hook(settings, workspace, session_slug, _doc_tools_called),
        "on_post_tool_use": _make_post_tool_hook(workspace, session_slug, _doc_tools_called),
        "on_session_end": _make_session_end_hook(workspace, session_slug),
        "on_error_occurred": _make_error_hook(workspace, session_slug),
    }


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def _make_session_start_hook(workspace: "WorkspaceManager", skill_metas: "list | None" = None):
    async def on_session_start(input_data: dict[str, Any], invocation: Any) -> dict | None:
        """Inject project context + vendor skill overview into the session."""
        project = workspace.load_project()
        context_parts: list[str] = []

        if project:
            context_parts.append(f"Project: {project.get('name', 'unknown')}")
            if project.get("summary"):
                context_parts.append(f"History: {project['summary']}")

        # Last 20 context log entries (coordinator project history only)
        entries = workspace.load_context_log(tail=20)
        task_entries = [e for e in entries if e.get("query")]  # skip hook-audit entries
        if task_entries:
            context_parts.append(
                "Recent tasks (coordinator project history — "
                "sub-agents use their own role-specific memory):"
            )
            for e in task_entries[-5:]:
                icon = "✅" if e.get("status") in ("approved", "completed") else "⚠️"
                context_parts.append(
                    f"  {icon} \"{e.get('query', '')}\" → {e.get('status', 'unknown')} "
                    f"(skills: {', '.join(e.get('skills_used', []))})"
                )

        # Vendor skill overview: inject count + names only (progressive disclosure Level 1)
        # Use pre-scanned skill_metas if provided to avoid duplicate file I/O.
        try:
            if skill_metas is not None:
                skills = skill_metas
            else:
                from aqualib.skills.scanner import scan_all_skill_dirs

                skills = scan_all_skill_dirs(workspace.settings, workspace)
            if skills:
                skill_names = ", ".join(f"vendor_{s.name}" for s in skills)
                context_parts.append(
                    f"\nAvailable vendor skills ({len(skills)}): {skill_names}. "
                    "Use 'read_library_doc' then 'read_skill_doc' for full documentation."
                )
        except Exception:
            logger.debug("Could not load vendor skills for session context.", exc_info=True)

        # Library names only (not full doc content — use read_library_doc on demand)
        repo_vendor = Path(__file__).resolve().parent.parent.parent.parent / "vendor"
        if repo_vendor.is_dir():
            lib_dirs = [d for d in sorted(repo_vendor.iterdir()) if d.is_dir()]
            if lib_dirs:
                lib_names = ", ".join(d.name for d in lib_dirs)
                context_parts.append(
                    f"Vendor libraries: {lib_names}. "
                    "Use 'read_library_doc' for full library documentation."
                )

        if not context_parts:
            return None

        ctx = "\n".join(context_parts)
        return {"additionalContext": ctx[:_MAX_ADDITIONAL_CONTEXT_CHARS]}

    return on_session_start


def _make_prompt_hook(workspace: "WorkspaceManager", session_slug: str | None = None):
    async def on_user_prompt_submitted(input_data: dict[str, Any], invocation: Any) -> None:
        """Record the user prompt to context_log for project memory."""
        raw_ts = input_data.get("timestamp")
        if raw_ts is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        elif isinstance(raw_ts, str):
            timestamp = raw_ts
        elif isinstance(raw_ts, (int, float)):
            # SDK may pass Unix epoch in milliseconds or seconds.
            # Threshold: any value > 1e10 is treated as milliseconds (covers all
            # ms timestamps after April 1970); values <= 1e10 are seconds (covers
            # all second timestamps up to year 2286).
            epoch_s = raw_ts / 1000 if raw_ts > 1e10 else raw_ts
            timestamp = datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat()
        else:
            timestamp = str(raw_ts)
        entry: dict[str, Any] = {
            "event": "user_prompt",
            "query": input_data.get("prompt", ""),
            "timestamp": timestamp,
        }
        if session_slug:
            entry["session_slug"] = session_slug
        workspace.append_audit_entry(entry)
        return None  # do not modify the prompt

    return on_user_prompt_submitted


def _make_pre_tool_hook(
    settings: "Settings",
    workspace: "WorkspaceManager",
    session_slug: str | None = None,
    doc_tools_called: "set[str] | None" = None,
):
    if doc_tools_called is None:
        doc_tools_called = set()

    _vendor_reminder_sent = [False]  # mutable flag: fires once per session

    # Utility tools that should never trigger the vendor priority reminder
    _UTILITY_TOOLS = frozenset({"workspace_search", "read_skill_doc", "read_library_doc", "write_plan"})

    async def on_pre_tool_use(
        input_data: dict[str, Any], invocation: Any
    ) -> dict[str, Any]:
        """Doc-first gate + vendor priority check + pre-execution audit record.

        If a vendor_* tool is invoked before any documentation has been read
        in this session, allow the call but warn the model to read docs first.
        (Using 'allow' instead of 'block' avoids wasting a full LLM turn; the
        model gets an informative warning message alongside the tool result.)

        If vendor skills are available but the agent chose a non-utility built-in
        tool, return an ``additionalContext`` reminder once per session.
        """
        tool_name = input_data.get("toolName", "")

        entry: dict[str, Any] = {
            "event": "pre_tool_use",
            "tool": tool_name,
            "args_preview": str(input_data.get("toolArgs", {}))[:200],
        }
        if session_slug:
            entry["session_slug"] = session_slug
        workspace.append_audit_entry(entry)

        # Doc-first gate: warn (but allow) vendor tool calls until docs have been read.
        # `not doc_tools_called` is True when the set is empty (no docs read yet).
        if tool_name.startswith("vendor_") and not doc_tools_called:
            return {
                "permissionDecision": "allow",
                "additionalContext": (
                    "⚠️ DOC-FIRST WARNING: You are invoking a vendor tool before reading "
                    "documentation. Call read_library_doc first to understand the library's "
                    "CLI format, then read_skill_doc for the specific skill parameters. "
                    "Then retry the vendor tool call with the 'command' field set to the full "
                    "shell command string."
                ),
            }

        # Vendor priority reminder: fires once per session, skips utility tools
        if (
            settings.vendor_priority
            and not _vendor_reminder_sent[0]
            and not tool_name.startswith("vendor_")
            and tool_name not in _UTILITY_TOOLS
        ):
            vendor_skills = [
                t for t in input_data.get("availableTools", [])
                if str(t).startswith("vendor_")
            ]
            if vendor_skills:
                _vendor_reminder_sent[0] = True
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


def _make_post_tool_hook(
    workspace: "WorkspaceManager",
    session_slug: str | None = None,
    doc_tools_called: "set[str] | None" = None,
):
    if doc_tools_called is None:
        doc_tools_called = set()

    async def on_post_tool_use(input_data: dict[str, Any], invocation: Any) -> None:
        """Record tool execution result to the audit trail.

        Also tracks when read_skill_doc or read_library_doc are called so the
        doc-first gate in on_pre_tool_use can allow vendor tool invocations.

        Automatically captures reviewer verdicts and executor vendor-skill
        results into agent-role memory when a session_slug is available.
        """
        tool_name = input_data.get("toolName", "")
        result_text = str(input_data.get("toolResult", ""))

        # Track documentation reads for the doc-first gate
        if tool_name in ("read_skill_doc", "read_library_doc"):
            doc_tools_called.add(tool_name)

        entry: dict[str, Any] = {
            "event": "post_tool_use",
            "tool": tool_name,
            "success": not input_data.get("toolError"),
            "result_preview": result_text[:300],
        }
        if session_slug:
            entry["session_slug"] = session_slug
        workspace.append_audit_entry(entry)

        # Auto-capture reviewer memory when the result contains a VERDICT
        if session_slug and "VERDICT:" in result_text.upper():
            try:
                _save_reviewer_memory(workspace, session_slug, result_text)
            except Exception:
                logger.debug("Failed to save reviewer memory", exc_info=True)

        # Auto-capture execution report when the Executor produces one
        if session_slug and "EXECUTION_REPORT:" in result_text.upper():
            try:
                _save_execution_report_memory(workspace, session_slug, result_text)
            except Exception:
                logger.debug("Failed to save execution report memory", exc_info=True)

        # Auto-capture executor memory when a vendor_* tool completes,
        # and bridge the result to reviewer memory so the reviewer can
        # audit independently without needing the executor's conversation.
        if session_slug and tool_name.startswith("vendor_"):
            vendor_entry = {
                "event": "vendor_tool_use",
                "tool": tool_name,
                "success": not input_data.get("toolError"),
                "output_preview": result_text[:200],
            }
            try:
                workspace.append_agent_memory_entry(
                    session_slug,
                    "executor",
                    vendor_entry,
                )
            except Exception:
                logger.debug("Failed to save executor memory", exc_info=True)
            try:
                workspace.append_agent_memory_entry(
                    session_slug,
                    "reviewer",
                    vendor_entry,
                )
            except Exception:
                logger.debug("Failed to save reviewer memory for vendor tool", exc_info=True)

        return None

    return on_post_tool_use


def _make_session_end_hook(workspace: "WorkspaceManager", session_slug: str | None = None):
    async def on_session_end(input_data: dict[str, Any], invocation: Any) -> None:
        """Flush and finalise the workspace state after the session ends."""
        workspace.finalize_task()
        if session_slug:
            workspace.finalize_session_results(session_slug)
        return None

    return on_session_end


def _build_rethink_hint(error_context: str, error_msg: str, attempt: int, max_attempts: int) -> str:
    """Generate a concise rethink hint for the agent based on error patterns."""
    error_lower = error_msg.lower()

    if "permission denied" in error_lower:
        fix_suggestion = "Check file paths are within workspace data/ and outputs go to results/."
    elif "no such file" in error_lower or "not found" in error_lower:
        fix_suggestion = "Use workspace_search to verify correct file paths before retrying."
    elif "import" in error_lower or "module" in error_lower:
        fix_suggestion = "Read SKILL.md for required packages; try --demo flag if available."
    elif "timeout" in error_lower:
        fix_suggestion = "Try with a smaller input dataset or check if chunked processing is supported."
    elif "invalid" in error_lower and ("param" in error_lower or "arg" in error_lower):
        fix_suggestion = "Use read_skill_doc to verify parameter schema and types."
    else:
        fix_suggestion = "Re-read skill docs via read_skill_doc and adjust parameters."

    return (
        f"🔄 RETRY {attempt}/{max_attempts}: {error_msg[:150]} — "
        f"{fix_suggestion} Do NOT retry with identical parameters."
    )


def _make_error_hook(workspace: "WorkspaceManager", session_slug: str | None = None):
    _retry_counts: dict[str, int] = {}
    _MAX_RETRIES = 2  # Aligned with Executor prompt retry count

    async def on_error_occurred(
        input_data: dict[str, Any], invocation: Any
    ) -> dict[str, Any]:
        """Error handling with rethink guidance.

        - Any tool failure → retry up to _MAX_RETRIES times with error analysis
        - Each retry includes additionalContext with rethink hints
        - After all retries exhausted → skip with user-facing summary
        """
        error_context = input_data.get("errorContext", "")
        error_msg = input_data.get("error", "")
        error_context_str = str(error_context)
        error_msg_str = str(error_msg)[:500]

        entry: dict[str, Any] = {
            "event": "error",
            "context": error_context_str,
            "error": error_msg_str,
        }
        if session_slug:
            entry["session_slug"] = session_slug
        workspace.append_audit_entry(entry)

        retry_key = f"{error_context_str}:{error_msg_str[:100]}"
        count = _retry_counts.get(retry_key, 0) + 1
        _retry_counts[retry_key] = count

        if count <= _MAX_RETRIES:
            logger.info(
                "Error retry %d/%d for %s: %s",
                count, _MAX_RETRIES, error_context_str, error_msg_str[:200],
            )
            rethink_hint = _build_rethink_hint(error_context_str, error_msg_str, count, _MAX_RETRIES)
            return {
                "errorHandling": "retry",
                "additionalContext": rethink_hint,
            }
        else:
            logger.warning(
                "Retries exhausted (%d) for %s – skipping.",
                _MAX_RETRIES, error_context_str,
            )
            _retry_counts.pop(retry_key, None)
            return {
                "errorHandling": "skip",
                "additionalContext": (
                    f"⚠️ All {_MAX_RETRIES} retry attempts failed for '{error_context_str}'. "
                    f"Last error: {error_msg_str[:200]}. "
                    f"Report this failure to the user honestly. Do NOT fabricate results."
                ),
            }

    return on_error_occurred
