"""Document & skill indexer for the RAG pipeline (LlamaIndex-backed)."""

from __future__ import annotations

import json
import logging
from typing import Any

from aqualib.config import Settings
from aqualib.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class RAGIndexer:
    """Build and persist a LlamaIndex vector-store index.

    Sources:
    * Files under ``data/`` (user-provided documents).
    * Skill descriptions from the :class:`SkillRegistry`.
    """

    def __init__(self, settings: Settings, registry: SkillRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._index: Any = None
        self._index_path = settings.directories.work / ".rag_index"

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    async def build_index(self) -> None:
        """(Re-)build the in-memory vector index."""
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core import Settings as LISettings
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        # Configure LlamaIndex globals
        LISettings.llm = OpenAI(
            model=self.settings.llm.model,
            api_key=self.settings.llm.api_key or None,
            api_base=self.settings.llm.base_url,
            temperature=self.settings.llm.temperature,
        )
        LISettings.embed_model = OpenAIEmbedding(
            model_name=self.settings.rag.embed_model,
            api_key=self.settings.llm.api_key or None,
            api_base=self.settings.llm.base_url,
        )
        LISettings.chunk_size = self.settings.rag.chunk_size
        LISettings.chunk_overlap = self.settings.rag.chunk_overlap

        docs: list[Document] = []

        # 1. Skill descriptions
        for desc in self.registry.to_descriptions():
            docs.append(
                Document(
                    text=json.dumps(desc, indent=2),
                    metadata={"type": "skill", "name": desc["name"], "source": desc["source"]},
                )
            )

        # 2. Files in the data directory
        data_dir = self.settings.directories.data
        if data_dir.exists():
            for fp in data_dir.rglob("*"):
                if fp.is_file() and fp.suffix in {".txt", ".md", ".json", ".csv", ".yaml", ".yml"}:
                    try:
                        text = fp.read_text(errors="replace")
                        docs.append(
                            Document(
                                text=text[:50_000],  # safety cap
                                metadata={"type": "file", "path": str(fp.relative_to(data_dir))},
                            )
                        )
                    except Exception as exc:
                        logger.warning("Skipping %s: %s", fp, exc)

        if not docs:
            logger.warning("No documents to index – RAG will return empty results.")
            return

        self._index = VectorStoreIndex.from_documents(docs)
        logger.info("RAG index built with %d documents.", len(docs))

    # ------------------------------------------------------------------
    # Persist / load  (optional – keeps index between runs)
    # ------------------------------------------------------------------

    def persist(self) -> None:
        if self._index is not None:
            self._index_path.mkdir(parents=True, exist_ok=True)
            self._index.storage_context.persist(persist_dir=str(self._index_path))
            logger.info("RAG index persisted → %s", self._index_path)

    async def load_or_build(self) -> None:
        """Load a persisted index or build from scratch."""
        if self._index_path.exists():
            try:
                from llama_index.core import StorageContext, load_index_from_storage

                storage_ctx = StorageContext.from_defaults(persist_dir=str(self._index_path))
                self._index = load_index_from_storage(storage_ctx)
                logger.info("RAG index loaded from %s", self._index_path)
                return
            except Exception as exc:
                logger.warning("Failed to load persisted index: %s – rebuilding.", exc)
        await self.build_index()

    # ------------------------------------------------------------------
    # Public accessor
    # ------------------------------------------------------------------

    @property
    def index(self) -> Any:
        return self._index
