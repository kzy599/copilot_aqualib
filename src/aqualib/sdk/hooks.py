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
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _save_reviewer_memory(
    workspace: "WorkspaceManager",
    session_slug: str,
    result_text: str,
) -> None:
    """Extract reviewer verdict fields from *result_text* and persist to memory.

    Parses VERDICT, VENDOR_PRIORITY, PLAN_QUALITY, and SUGGESTIONS using
    regex so the reviewer's decisions accumulate in ``memory/reviewer.json``
    rather than being lost at session end.
    """
    verdict_match = re.search(r"VERDICT\s*:\s*(\S+)", result_text, re.IGNORECASE)
    vendor_match = re.search(r"VENDOR_PRIORITY\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE)
    quality_match = re.search(r"PLAN_QUALITY\s*:\s*(.+?)(?:\n|$)", result_text, re.IGNORECASE)
    suggestions_match = re.search(r"SUGGESTIONS\s*:\s*(.+?)(?:\n\n|$)", result_text, re.IGNORECASE | re.DOTALL)

    # Warn when the output doesn't match the expected reviewer format at all
    if not verdict_match:
        logger.warning("Reviewer memory: could not parse VERDICT from result text")
    if not vendor_match:
        logger.debug("Reviewer memory: could not parse VENDOR_PRIORITY from result text")
    if not quality_match:
        logger.debug("Reviewer memory: could not parse PLAN_QUALITY from result text")

    entry: dict[str, Any] = {
        "verdict": verdict_match.group(1).strip() if verdict_match else "unknown",
        "vendor_priority": vendor_match.group(1).strip() if vendor_match else "unknown",
        "plan_quality": quality_match.group(1).strip() if quality_match else "unknown",
        "suggestions": suggestions_match.group(1).strip() if suggestions_match else "",
        "violations": [],
    }

    # Collect violations: check that the field value *starts with* "violated"
    # to avoid false positives from phrases like "not violated".
    if re.match(r"violated", entry["vendor_priority"], re.IGNORECASE):
        entry["violations"].append(f"vendor_priority: {entry['vendor_priority']}")
    if re.match(r"violated", entry["plan_quality"], re.IGNORECASE):
        entry["violations"].append(f"plan_quality: {entry['plan_quality']}")

    workspace.append_agent_memory_entry(session_slug, "reviewer", entry)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_hooks(
    settings: "Settings",
    workspace: "WorkspaceManager",
    session_slug: str | None = None,
) -> dict:
    """Build and return the complete hook dict for a Copilot SDK session."""
    # Shared state: tracks which doc tools the model has called this session.
    # Used by the doc-first gate to enforce read_library_doc/read_skill_doc
    # before any vendor_* tool invocation.
    _doc_tools_called: set[str] = set()

    return {
        "on_session_start": _make_session_start_hook(workspace),
        "on_user_prompt_submitted": _make_prompt_hook(workspace, session_slug),
        "on_pre_tool_use": _make_pre_tool_hook(settings, workspace, session_slug, _doc_tools_called),
        "on_post_tool_use": _make_post_tool_hook(workspace, session_slug, _doc_tools_called),
        "on_session_end": _make_session_end_hook(workspace, session_slug),
        "on_error_occurred": _make_error_hook(workspace, session_slug),
    }


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def _make_session_start_hook(workspace: "WorkspaceManager"):
    async def on_session_start(input_data: dict[str, Any], invocation: Any) -> dict | None:
        """Inject project context + vendor skill overview into the session."""
        project = workspace.load_project()
        context_parts: list[str] = []

        if project:
            context_parts.append(f"Project: {project.get('name', 'unknown')}")
            if project.get("summary"):
                context_parts.append(f"History: {project['summary']}")

        # Last 5 context log entries (coordinator project history only)
        entries = workspace.load_context_log()
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

        # Vendor skill overview (progressive disclosure Level 1)
        try:
            from aqualib.skills.scanner import scan_all_skill_dirs

            skills = scan_all_skill_dirs(workspace.settings, workspace)
            if skills:
                context_parts.append(f"\nAvailable vendor skills ({len(skills)}):")
                for s in skills:
                    context_parts.append(f"  - vendor_{s.name}: {s.description[:80]}")
                context_parts.append(
                    "\nUse 'read_skill_doc' tool for full documentation before invoking."
                )
        except Exception:
            logger.debug("Could not load vendor skills for session context.", exc_info=True)

        # Library-level documentation (progressive disclosure Level 0)
        repo_vendor = Path(__file__).resolve().parent.parent.parent.parent / "vendor"
        if repo_vendor.is_dir():
            lib_dirs = [d for d in sorted(repo_vendor.iterdir()) if d.is_dir()]
            for lib_dir in lib_dirs:
                for doc_name in ("llms.txt", "AGENTS.md"):
                    doc = lib_dir / doc_name
                    if doc.exists():
                        context_parts.append(
                            f"\n## Library: {lib_dir.name}\n"
                            f"{doc.read_text(encoding='utf-8')[:500]}"
                        )
                        break
            if lib_dirs:
                context_parts.append(
                    "\nUse 'read_library_doc' tool for full library documentation "
                    "before reading individual SKILL.md files."
                )

        if not context_parts:
            return None

        return {"additionalContext": "\n".join(context_parts)}

    return on_session_start


def _make_prompt_hook(workspace: "WorkspaceManager", session_slug: str | None = None):
    async def on_user_prompt_submitted(input_data: dict[str, Any], invocation: Any) -> None:
        """Record the user prompt to context_log for project memory."""
        entry: dict[str, Any] = {
            "event": "user_prompt",
            "query": input_data.get("prompt", ""),
            "timestamp": input_data.get("timestamp"),
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

    async def on_pre_tool_use(
        input_data: dict[str, Any], invocation: Any
    ) -> dict[str, Any]:
        """Doc-first gate + vendor priority check + pre-execution audit record.

        If a vendor_* tool is invoked before any documentation has been read
        in this session, block the call and instruct the model to read docs first.

        If vendor skills are available but the agent chose a built-in tool,
        return an ``additionalContext`` reminder to steer the agent back.
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

        # Doc-first gate: block vendor tool calls until docs have been read.
        # `not doc_tools_called` is True when the set is empty (no docs read yet).
        if tool_name.startswith("vendor_") and not doc_tools_called:
            return {
                "permissionDecision": "block",
                "additionalContext": (
                    "⛔ DOC-FIRST GATE: You must read documentation before invoking vendor tools. "
                    "Call read_library_doc first to understand the library's CLI format and "
                    "architecture, then read_skill_doc for the specific skill parameters. "
                    "Then retry the vendor tool call with the 'command' field set to the full "
                    "shell command string."
                ),
            }

        # Vendor priority reminder
        if settings.vendor_priority and not tool_name.startswith("vendor_"):
            vendor_skills = [
                t for t in input_data.get("availableTools", [])
                if str(t).startswith("vendor_")
            ]
            if vendor_skills:
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

        # Auto-capture executor memory when a vendor_* tool completes
        if session_slug and tool_name.startswith("vendor_"):
            try:
                workspace.append_agent_memory_entry(
                    session_slug,
                    "executor",
                    {
                        "event": "vendor_tool_use",
                        "tool": tool_name,
                        "success": not input_data.get("toolError"),
                        "output_preview": result_text[:200],
                    },
                )
            except Exception:
                logger.debug("Failed to save executor memory", exc_info=True)

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
    """Generate a structured rethink hint for the agent based on error patterns."""
    error_lower = error_msg.lower()

    if "permission denied" in error_lower:
        fix_suggestion = (
            "The tool was denied file access. Before retrying:\n"
            "1. Use `workspace_search` to find accessible file paths\n"
            "2. Ensure output paths are within the workspace results/ directory\n"
            "3. Check that input files exist in workspace data/"
        )
    elif "no such file" in error_lower or "not found" in error_lower:
        fix_suggestion = (
            "A file or directory was not found. Before retrying:\n"
            "1. Use `workspace_search` to verify correct file paths\n"
            "2. Check for typos in file names\n"
            "3. Ensure the data directory exists and contains expected files"
        )
    elif "import" in error_lower or "module" in error_lower:
        fix_suggestion = (
            "A Python dependency is missing. Before retrying:\n"
            "1. Read the SKILL.md to check required packages\n"
            "2. Consider using the --demo flag if available\n"
            "3. Try an alternative skill that doesn't require this dependency"
        )
    elif "timeout" in error_lower:
        fix_suggestion = (
            "The operation timed out. Before retrying:\n"
            "1. Try with a smaller input dataset\n"
            "2. Check if the skill supports chunked processing\n"
            "3. Consider increasing timeout or using a subset of data"
        )
    elif "invalid" in error_lower and ("param" in error_lower or "arg" in error_lower):
        fix_suggestion = (
            "Parameters were invalid. Before retrying:\n"
            "1. Use `read_skill_doc` to read the correct parameter schema\n"
            "2. Verify parameter types (string vs int vs path)\n"
            "3. Check required vs optional parameters"
        )
    else:
        fix_suggestion = (
            "An unexpected error occurred. Before retrying:\n"
            "1. Use `read_skill_doc` to re-read the skill's documentation\n"
            "2. Verify all input parameters and file paths\n"
            "3. Try with different parameters or the --demo flag"
        )

    return (
        f"🔄 RETRY ATTEMPT {attempt}/{max_attempts} — RETHINK REQUIRED\n\n"
        f"Error: {error_msg[:300]}\n\n"
        f"Analysis & Fix Suggestions:\n{fix_suggestion}\n\n"
        f"IMPORTANT: Do NOT retry with the exact same parameters. "
        f"Analyse the error, adjust your approach, then try again."
    )


def _make_error_hook(workspace: "WorkspaceManager", session_slug: str | None = None):
    _retry_counts: dict[str, int] = {}
    _MAX_RETRIES = 4  # Aligned with tool_adapter's _MAX_SKILL_RETRIES

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
