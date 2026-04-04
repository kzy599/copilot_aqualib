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
    from aqualib.skills.registry import SkillRegistry
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the **Searcher** agent (RAG proxy) of the AquaLib framework.

You receive a user query and retrieved context chunks.  Your job:
1. Synthesise the chunks into a clear, concise information brief.
2. Highlight any **vendor skills** that appear relevant.
3. Suggest next steps for the Executor.
4. If the context is insufficient, say so and recommend what data the user
   should add to the ``data/`` directory.

Context may come from RAG vector search, registry keyword matching, or
workspace file scanning.  Treat all sources equally but note the origin.

Retrieved context:
{context_json}
"""


class SearcherAgent(BaseAgent):
    """Retrieves relevant context via RAG and summarises it for the pipeline."""

    name = "Searcher"
    role = Role.SEARCHER

    def __init__(
        self,
        settings: "Settings",
        retriever: "Retriever",
        registry: "SkillRegistry" = None,
        workspace: "WorkspaceManager" = None,
    ) -> None:
        super().__init__(settings)
        self.retriever = retriever
        self.registry = registry
        self.workspace = workspace

    async def _execute(self, task: Task) -> Task:
        # Progressive disclosure: summaries first
        summaries = await self.retriever.query_summaries(task.user_query)
        task.add_message(self.role, f"RAG summaries ({len(summaries)} chunks): {json.dumps(summaries, indent=2)}")

        if summaries:
            # Normal RAG path – progressive disclosure Level 2
            full_results = await self.retriever.query_full(task.user_query)
            brief = await self._synthesise(task.user_query, full_results)
            task.add_message(self.role, f"Information brief:\n{brief}")
            return task

        # --- Fallback discovery (no RAG results) ---
        task.add_message(self.role, "No RAG context available \u2013 activating fallback discovery.")

        fallback_context: list[dict] = []

        # Tier 1: Registry-based skill matching
        if self.registry is not None:
            candidates = self.registry.resolve(task.user_query)
            if candidates:
                skill_briefs = [
                    {
                        "name": s.meta.name,
                        "source": s.meta.source.value,
                        "description": s.meta.description,
                        "tags": s.meta.tags,
                        "relevance": "keyword_match",
                    }
                    for s in candidates[:5]  # Top 5 matches
                ]
                fallback_context.extend(skill_briefs)
                task.add_message(
                    self.role,
                    f"Fallback Tier 1 \u2013 Registry match: {len(skill_briefs)} candidate skill(s) found.",
                )

        # Tier 2: Workspace file scanning (grep)
        if self.workspace is not None:
            file_hits = self.workspace.scan_data_files(task.user_query)
            if file_hits:
                fallback_context.extend(
                    {"type": "file_scan", **hit} for hit in file_hits[:5]
                )
                task.add_message(
                    self.role,
                    f"Fallback Tier 2 \u2013 File scan: {len(file_hits)} file(s) matched keywords.",
                )

        # Tier 3: LLM synthesis (only if we found anything)
        if fallback_context:
            brief = await self._synthesise(task.user_query, fallback_context)
            task.add_message(self.role, f"Fallback brief:\n{brief}")
        else:
            task.add_message(
                self.role,
                "No relevant context found via RAG, registry, or file scan. "
                "Executor will proceed with skill list only.",
            )

        return task

    async def _synthesise(self, query: str, chunks: list[dict]) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT.format(context_json=json.dumps(chunks, indent=2))},
            {"role": "user", "content": query},
        ]
        return await self._chat(messages)
