"""Factory that wires all components together into a ready-to-use Orchestrator."""

from __future__ import annotations

from aqualib.config import Settings, get_settings
from aqualib.core.executor import ExecutorAgent
from aqualib.core.orchestrator import Orchestrator
from aqualib.core.reviewer import ReviewerAgent
from aqualib.core.searcher import SearcherAgent
from aqualib.rag.indexer import RAGIndexer
from aqualib.rag.retriever import Retriever
from aqualib.skills.clawbio.skills import ALL_CLAWBIO_SKILLS
from aqualib.skills.registry import SkillRegistry
from aqualib.workspace.manager import WorkspaceManager


def build_registry(settings: Settings) -> SkillRegistry:
    """Create and populate the skill registry (Clawbio skills auto-registered)."""
    registry = SkillRegistry(clawbio_priority=settings.clawbio_priority)
    for cls in ALL_CLAWBIO_SKILLS:
        registry.register(cls())
    return registry


async def build_orchestrator(
    settings: Settings | None = None,
    *,
    skip_rag_index: bool = False,
) -> Orchestrator:
    """Construct the full agent pipeline, ready to call ``orchestrator.run(query)``."""
    if settings is None:
        settings = get_settings()

    registry = build_registry(settings)
    workspace = WorkspaceManager(settings)

    # RAG
    indexer = RAGIndexer(settings, registry)
    if not skip_rag_index:
        await indexer.load_or_build()
    retriever = Retriever(indexer.index, top_k=settings.rag.similarity_top_k)

    # Agents
    searcher = SearcherAgent(settings, retriever)
    executor = ExecutorAgent(settings, registry, workspace)
    reviewer = ReviewerAgent(settings, registry, workspace)

    return Orchestrator(searcher, executor, reviewer, workspace)
