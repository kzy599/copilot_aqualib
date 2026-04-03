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
    3. **Reviewer** – audit results (Clawbio priority check).

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

    async def run(self, user_query: str) -> Task:
        """Execute the full pipeline and return the final Task."""
        task = Task(user_query=user_query)
        task.add_message(Role.USER, user_query)
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
        return task
