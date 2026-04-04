"""Orchestrator – coordinates the Searcher → Executor → Reviewer pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aqualib.core.message import Role, Task, TaskStatus

if TYPE_CHECKING:
    from aqualib.core.executor import ExecutorAgent
    from aqualib.core.reviewer import ReviewerAgent
    from aqualib.core.searcher import SearcherAgent
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Runs the canonical pipeline:

    1. **Searcher** – retrieve context (RAG + progressive disclosure).
    2. **Executor** – plan & invoke skills.
    3. **Reviewer** – audit results (vendor priority check).

    If the reviewer requests revision (up to ``max_retries``), the executor
    re-runs with feedback.
    """

    def __init__(
        self,
        searcher: "SearcherAgent",
        executor: "ExecutorAgent",
        reviewer: "ReviewerAgent",
        workspace: "WorkspaceManager",
        *,
        max_retries: int = 2,
    ) -> None:
        self.searcher = searcher
        self.executor = executor
        self.reviewer = reviewer
        self.workspace = workspace
        self.max_retries = max_retries

    def _build_project_context(self) -> str:
        """Load project metadata and recent task history for agent context."""
        meta = self.workspace.load_project()
        if meta is None:
            return ""

        parts = [f"Project: {meta.get('name', 'unknown')}"]

        summary = meta.get("summary", "")
        if summary:
            parts.append(f"History: {summary}")

        # Include the last 5 context log entries for agent awareness
        entries = self.workspace.load_context_log()
        if entries:
            recent = entries[-5:]
            parts.append("Recent tasks:")
            for e in recent:
                status_icon = "✅" if e.get("status") == "approved" else "⚠️"
                parts.append(
                    f"  - {status_icon} \"{e.get('query', '')}\" → "
                    f"{e.get('status', 'unknown')} "
                    f"(skills: {', '.join(e.get('skills_used', []))})"
                )

        return "\n".join(parts)

    async def run(self, user_query: str) -> Task:
        """Execute the full pipeline and return the final Task."""
        task = Task(user_query=user_query)
        task.add_message(Role.USER, user_query)

        # Inject project context if available
        project_context = self._build_project_context()
        if project_context:
            task.add_message(Role.ORCHESTRATOR, f"Project context:\n{project_context}")

        task.add_message(Role.ORCHESTRATOR, "Pipeline started.")

        # Step 1: Search / RAG
        logger.info("[Orchestrator] Step 1 – Searcher")
        task = await self.searcher.run(task)

        for attempt in range(1, self.max_retries + 2):  # +1 for initial, +1 for range
            # Step 2: Execute
            logger.info("[Orchestrator] Step 2 – Executor (attempt %d)", attempt)
            task = await self.executor.run(task)

            # Step 3: Review
            logger.info("[Orchestrator] Step 3 – Reviewer")
            task = await self.reviewer.run(task)

            if task.status == TaskStatus.APPROVED:
                task.add_message(Role.ORCHESTRATOR, f"Task approved after {attempt} attempt(s). ✅")
                break

            if attempt <= self.max_retries:
                task.add_message(
                    Role.ORCHESTRATOR,
                    f"Reviewer requested revision (attempt {attempt}/{self.max_retries}). Re-running executor.",
                )
                task.status = TaskStatus.PENDING  # reset for re-execution
            else:
                task.add_message(Role.ORCHESTRATOR, "Max retries reached – returning for manual review.")

        self.workspace.save_task(task)
        self.workspace.update_project_after_task(task)
        return task
