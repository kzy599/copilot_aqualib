"""Reviewer agent – audits executor output with Clawbio-priority scepticism."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from aqualib.core.agent_base import BaseAgent
from aqualib.core.message import AuditReport, Role, SkillSource, Task, TaskStatus

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.skills.registry import SkillRegistry
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the **Reviewer** agent of the AquaLib framework.

Your responsibilities:
1. Audit the executor's work for correctness, completeness, and quality.
2. **Clawbio Priority Enforcement** – Maintain a sceptical stance.  Ask:
   "Could a Clawbio skill have been used instead of a generic tool?"
   If yes, the review MUST flag this as a priority violation.
3. Return your review as JSON:
   {
     "approved": true/false,
     "clawbio_priority_satisfied": true/false,
     "verdict": "...",
     "clawbio_priority_notes": "...",
     "suggestions": ["..."]
   }

Skill invocations to review:
{invocations_json}

Available Clawbio skills for reference:
{clawbio_skills_json}
"""


class ReviewerAgent(BaseAgent):
    """Audits the executor's work, enforcing Clawbio priority."""

    name = "Reviewer"
    role = Role.REVIEWER

    def __init__(
        self,
        settings: "Settings",
        registry: "SkillRegistry",
        workspace: "WorkspaceManager",
    ) -> None:
        super().__init__(settings)
        self.registry = registry
        self.workspace = workspace

    async def _execute(self, task: Task) -> Task:
        review = await self._review(task)

        approved = review.get("approved", False)
        clawbio_ok = review.get("clawbio_priority_satisfied", False)
        verdict = review.get("verdict", "")
        clawbio_notes = review.get("clawbio_priority_notes", "")

        task.review_passed = approved
        task.clawbio_priority_satisfied = clawbio_ok
        task.review_notes = verdict
        task.status = TaskStatus.APPROVED if approved else TaskStatus.NEEDS_REVISION

        task.add_message(self.role, f"Review verdict: {'✅ APPROVED' if approved else '❌ NEEDS REVISION'}")
        task.add_message(self.role, f"Clawbio priority: {'✅ OK' if clawbio_ok else '⚠️ VIOLATION – ' + clawbio_notes}")

        if review.get("suggestions"):
            task.add_message(self.role, "Suggestions:\n" + "\n".join(f"- {s}" for s in review["suggestions"]))

        # Build & persist audit report
        report = AuditReport(
            task_id=task.task_id,
            user_query=task.user_query,
            status=task.status,
            executor_summary=self._executor_summary(task),
            reviewer_verdict=verdict,
            clawbio_priority_check=clawbio_notes or ("Satisfied" if clawbio_ok else "Not evaluated"),
            skill_invocations=task.skill_invocations,
            messages=task.messages,
        )
        self.workspace.save_audit_report(report)
        self.workspace.save_task(task)
        return task

    # ------------------------------------------------------------------
    # LLM review
    # ------------------------------------------------------------------

    async def _review(self, task: Task) -> dict:
        invocations_json = json.dumps(
            [inv.model_dump(mode="json") for inv in task.skill_invocations], indent=2
        )
        clawbio_descs = [
            {"name": s.meta.name, "description": s.meta.description, "tags": s.meta.tags}
            for s in self.registry.list_clawbio()
        ]
        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(
                    invocations_json=invocations_json,
                    clawbio_skills_json=json.dumps(clawbio_descs, indent=2),
                ),
            },
            {"role": "user", "content": f"User query: {task.user_query}\n\nPlease review."},
        ]
        raw = await self._chat(messages)
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning("Reviewer LLM returned non-JSON: %s", raw[:200])
            return {"approved": False, "verdict": raw[:500], "clawbio_priority_satisfied": False}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _executor_summary(task: Task) -> str:
        ok = sum(1 for i in task.skill_invocations if i.success)
        total = len(task.skill_invocations)
        clawbio_count = sum(1 for i in task.skill_invocations if i.source == SkillSource.CLAWBIO)
        return (
            f"{ok}/{total} skill invocations succeeded. "
            f"{clawbio_count} Clawbio skill(s) used."
        )
