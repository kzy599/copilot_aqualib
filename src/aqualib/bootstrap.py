"""Factory that wires all components together into a ready-to-use Orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

from aqualib.config import Settings, get_settings
from aqualib.core.executor import ExecutorAgent
from aqualib.core.orchestrator import Orchestrator
from aqualib.core.reviewer import ReviewerAgent
from aqualib.core.searcher import SearcherAgent
from aqualib.rag.indexer import RAGIndexer
from aqualib.rag.retriever import Retriever
from aqualib.skills.clawbio.skills import ALL_CLAWBIO_SKILLS
from aqualib.skills.loader import mount_vendor_skills, scan_vendor_directory
from aqualib.skills.registry import SkillRegistry
from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


def build_registry(settings: Settings) -> SkillRegistry:
    """Create and populate the skill registry.

    Registration order (vendor-first):
      1. Scan the runtime mount point (``skills/vendor/``) for externally
         provided skills – highest priority (user customisation).
      2. Scan **all** subdirectories in the ``vendor/`` directory at the repo
         root for repo-shipped skill libraries.
      3. Register the bundled example skills as a fallback so the framework
         is usable out of the box.
    """
    registry = SkillRegistry(vendor_priority=settings.vendor_priority)

    # 1. Dynamic mount-point scan (highest priority — user customisation)
    mounted = mount_vendor_skills(settings.directories.skills_vendor, registry)
    logger.info("Mounted %d vendor skill(s) from %s", mounted, settings.directories.skills_vendor)

    # 2. Vendor directory scan (all repo-shipped libraries under vendor/)
    vendor_root = Path(__file__).resolve().parent.parent.parent / "vendor"
    if vendor_root.is_dir():
        for lib_dir in sorted(vendor_root.iterdir()):
            if not lib_dir.is_dir():
                continue
            vendor_mounted = 0
            for skill in scan_vendor_directory(lib_dir):
                if registry.get(skill.meta.name) is None:
                    registry.register(skill)
                    vendor_mounted += 1
            if vendor_mounted:
                logger.info(
                    "Mounted %d vendor skill(s) from %s", vendor_mounted, lib_dir
                )

    # 3. Bundled example skills (lowest priority — only register those not already present)
    for cls in ALL_CLAWBIO_SKILLS:
        instance = cls()
        if registry.get(instance.meta.name) is None:
            registry.register(instance)

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
    searcher = SearcherAgent(settings, retriever, registry=registry, workspace=workspace)
    executor = ExecutorAgent(settings, registry, workspace)
    reviewer = ReviewerAgent(settings, registry, workspace)

    return Orchestrator(searcher, executor, reviewer, workspace)
