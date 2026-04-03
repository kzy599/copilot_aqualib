"""Abstract base class for all AquaLib agents."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from aqualib.core.message import Role, Task

if TYPE_CHECKING:
    from aqualib.config import Settings


class BaseAgent(abc.ABC):
    """Every agent (Executor, Reviewer, Searcher) inherits from this."""

    name: str = "base"
    role: Role = Role.SYSTEM

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task: Task) -> Task:
        """Execute the agent's logic and return the (mutated) task."""
        task.add_message(self.role, f"[{self.name}] Starting …")
        task = await self._execute(task)
        task.add_message(self.role, f"[{self.name}] Finished.")
        return task

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def _execute(self, task: Task) -> Task:
        """Override with agent-specific logic."""
        ...

    # ------------------------------------------------------------------
    # Helpers – LLM calls
    # ------------------------------------------------------------------

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        """Call the configured LLM and return the assistant reply."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.settings.llm.api_key or None,
            base_url=self.settings.llm.base_url,
        )
        resp = await client.chat.completions.create(
            model=self.settings.llm.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.settings.llm.temperature,
            max_tokens=self.settings.llm.max_tokens,
        )
        return resp.choices[0].message.content or ""
