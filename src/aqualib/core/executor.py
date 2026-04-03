"""Executor agent – carries out the user's request by invoking skills."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aqualib.core.agent_base import BaseAgent
from aqualib.core.message import Role, SkillInvocation, SkillSource, Task, TaskStatus

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.skills.registry import SkillRegistry
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

# System prompt instructs the LLM to plan skill invocations
_SYSTEM_PROMPT = """\
You are the **Executor** agent of the AquaLib framework.

Your job:
1. Analyse the user query and the available skills.
2. **Always prefer Clawbio skills** over generic or external tools when there
   is *any* possibility of using them – even if a generic tool would be simpler.
3. Return a JSON array of skill invocation plans.  Each element:
   {"skill_name": "...", "parameters": {...}, "reason": "..."}
4. If no skill applies, return an empty array [].

Available skills (JSON):
{skills_json}
"""


class ExecutorAgent(BaseAgent):
    """Plans and executes skills, writing artefacts to the workspace."""

    name = "Executor"
    role = Role.EXECUTOR

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
        task.status = TaskStatus.RUNNING

        # Ask LLM to plan
        plan = await self._plan(task)
        task.add_message(self.role, f"Execution plan: {json.dumps(plan, indent=2)}")

        # Execute each planned skill
        for item in plan:
            invocation = await self._invoke_skill(task, item)
            task.skill_invocations.append(invocation)
            task.add_message(
                self.role,
                f"Skill `{invocation.skill_name}` ({'✅' if invocation.success else '❌'})"
                f" → {invocation.output_dir or 'N/A'}",
            )

        task.status = TaskStatus.COMPLETED
        self.workspace.save_task(task)
        return task

    # ------------------------------------------------------------------
    # LLM planning
    # ------------------------------------------------------------------

    async def _plan(self, task: Task) -> list[dict[str, Any]]:
        skills_json = json.dumps(self.registry.to_descriptions(), indent=2)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT.format(skills_json=skills_json)},
            {"role": "user", "content": task.user_query},
        ]
        raw = await self._chat(messages)

        # Parse JSON from the LLM response
        try:
            # Handle markdown code blocks
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError):
            logger.warning("Executor LLM returned non-JSON: %s", raw[:200])
            task.add_message(self.role, f"LLM plan parse error – raw: {raw[:500]}")
            return []

    # ------------------------------------------------------------------
    # Skill invocation
    # ------------------------------------------------------------------

    async def _invoke_skill(self, task: Task, plan_item: dict[str, Any]) -> SkillInvocation:
        skill_name = plan_item.get("skill_name", "")
        params = plan_item.get("parameters", {})

        inv = SkillInvocation(
            skill_name=skill_name,
            source=SkillSource.GENERIC,
            parameters=params,
        )

        skill = self.registry.get(skill_name)
        if skill is None:
            inv.error = f"Skill '{skill_name}' not found in registry."
            inv.finished_at = datetime.now(timezone.utc)
            return inv

        inv.source = skill.meta.source
        out_dir = self.workspace.skill_invocation_dir(task.task_id, inv.invocation_id)
        inv.output_dir = str(out_dir.relative_to(self.workspace.dirs.results))

        try:
            result = await skill.execute(params, out_dir)
            inv.output = result
            inv.success = True
            # Write meta for reviewer
            (out_dir / "invocation_meta.json").write_text(inv.model_dump_json(indent=2))
        except Exception as exc:
            inv.error = str(exc)
            logger.exception("Skill %s failed", skill_name)
        finally:
            inv.finished_at = datetime.now(timezone.utc)

        return inv
