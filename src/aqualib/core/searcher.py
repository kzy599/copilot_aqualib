"""Searcher (RAG) agent – progressive-disclosure information retrieval."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from aqualib.core.agent_base import BaseAgent
from aqualib.core.message import Role, Task

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.rag.retriever import Retriever

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the **Searcher** agent (RAG proxy) of the AquaLib framework.

You receive a user query and retrieved context chunks.  Your job:
1. Synthesise the chunks into a clear, concise information brief.
2. Highlight any **Clawbio skills** that appear relevant.
3. Suggest next steps for the Executor.
4. If the context is insufficient, say so and recommend what data the user
   should add to the ``data/`` directory.

Retrieved context (progressive disclosure – summaries first):
{context_json}
"""


class SearcherAgent(BaseAgent):
    """Retrieves relevant context via RAG and summarises it for the pipeline."""

    name = "Searcher"
    role = Role.SEARCHER

    def __init__(self, settings: "Settings", retriever: "Retriever") -> None:
        super().__init__(settings)
        self.retriever = retriever

    async def _execute(self, task: Task) -> Task:
        # Progressive disclosure: summaries first
        summaries = await self.retriever.query_summaries(task.user_query)
        task.add_message(self.role, f"RAG summaries ({len(summaries)} chunks): {json.dumps(summaries, indent=2)}")

        if not summaries:
            task.add_message(self.role, "No relevant context found in RAG index.")
            return task

        # If we have hits, get full details for the LLM
        full_results = await self.retriever.query_full(task.user_query)
        brief = await self._synthesise(task.user_query, full_results)
        task.add_message(self.role, f"Information brief:\n{brief}")

        return task

    async def _synthesise(self, query: str, chunks: list[dict]) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT.format(context_json=json.dumps(chunks, indent=2))},
            {"role": "user", "content": query},
        ]
        return await self._chat(messages)
