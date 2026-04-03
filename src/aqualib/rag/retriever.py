"""Retriever component – wraps LlamaIndex query engine for the Searcher agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single chunk returned by the retriever."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever:
    """Thin wrapper around a LlamaIndex ``VectorStoreIndex`` query engine.

    Supports *progressive disclosure*: first returns summaries, then full
    content on demand.
    """

    def __init__(self, index: Any, top_k: int = 5) -> None:
        self._index = index
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    async def query(self, question: str) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for *question*."""
        if self._index is None:
            logger.warning("RAG index is None – returning empty results.")
            return []

        engine = self._index.as_query_engine(similarity_top_k=self.top_k)
        response = await _async_query(engine, question)
        results: list[RetrievalResult] = []
        for node in response.source_nodes:
            results.append(
                RetrievalResult(
                    text=node.text,
                    score=node.score if node.score is not None else 0.0,
                    metadata=node.metadata,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Progressive disclosure helpers
    # ------------------------------------------------------------------

    async def query_summaries(self, question: str, max_chars: int = 200) -> list[dict]:
        """Return short summaries first (progressive disclosure level 1)."""
        chunks = await self.query(question)
        return [
            {
                "summary": c.text[:max_chars] + ("…" if len(c.text) > max_chars else ""),
                "score": round(c.score, 4),
                "metadata": c.metadata,
            }
            for c in chunks
        ]

    async def query_full(self, question: str) -> list[dict]:
        """Return full chunk text (progressive disclosure level 2)."""
        chunks = await self.query(question)
        return [
            {"text": c.text, "score": round(c.score, 4), "metadata": c.metadata}
            for c in chunks
        ]


# ---------------------------------------------------------------------------
# Async helper (LlamaIndex query engines are sync by default)
# ---------------------------------------------------------------------------

async def _async_query(engine: Any, question: str) -> Any:
    """Run the sync query engine in a thread to avoid blocking the loop."""
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, engine.query, question)
