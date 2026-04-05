"""Document & skill indexer for the RAG pipeline (LlamaIndex-backed)."""

from __future__ import annotations

import collections
import json
import logging
from typing import TYPE_CHECKING, Any

from aqualib.config import Settings
from aqualib.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


class RAGIndexer:
    """Build and persist a LlamaIndex vector-store index.

    Sources:
    * Skill descriptions from the :class:`SkillRegistry`.
    * Files under ``data/`` (user-provided documents).
    * SKILL.md files under ``skills/vendor/`` (vendor skill documentation).
    * Vendor execution traces from ``results/vendor_traces/``.
    * Task history from ``context_log.jsonl``.
    """

    def __init__(
        self,
        settings: Settings,
        registry: SkillRegistry,
        workspace: "WorkspaceManager | None" = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.workspace = workspace
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
        # Resolve credentials: RAG-specific > LLM fallback
        rag_api_key = self.settings.rag.api_key or self.settings.llm.api_key or None
        rag_base_url = self.settings.rag.base_url or self.settings.llm.base_url

        LISettings.llm = OpenAI(
            model=self.settings.llm.model,
            api_key=self.settings.llm.api_key or None,
            api_base=self.settings.llm.base_url,
            temperature=self.settings.llm.temperature,
        )
        LISettings.embed_model = OpenAIEmbedding(
            model_name=self.settings.rag.embed_model,
            api_key=rag_api_key,
            api_base=rag_base_url,
        )
        LISettings.chunk_size = self.settings.rag.chunk_size
        LISettings.chunk_overlap = self.settings.rag.chunk_overlap

        docs: list[Document] = []

        # 1. Skill descriptions (optional — registry may be empty in SDK path)
        skill_descs = self.registry.to_descriptions()
        if skill_descs:
            for desc in skill_descs:
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

        # 3. SKILL.md files from skills/vendor/
        skills_vendor_dir = self.settings.directories.skills_vendor
        if skills_vendor_dir.exists():
            for fp in skills_vendor_dir.rglob("SKILL.md"):
                try:
                    text = fp.read_text(errors="replace")
                    docs.append(
                        Document(
                            text=text[:50_000],
                            metadata={
                                "type": "skill_doc",
                                "path": str(fp.relative_to(skills_vendor_dir)),
                            },
                        )
                    )
                except Exception as exc:
                    logger.warning("Skipping vendor SKILL.md %s: %s", fp, exc)

        # 4. Vendor traces (last 50 most recent)
        vendor_traces_dir = self.settings.directories.vendor_traces
        if vendor_traces_dir.exists():
            try:
                trace_files = sorted(
                    [f for f in vendor_traces_dir.iterdir() if f.is_file() and f.suffix == ".json"]
                )[-50:]
                for fp in trace_files:
                    try:
                        text = fp.read_text(errors="replace")
                        docs.append(
                            Document(
                                text=text[:10_000],
                                metadata={"type": "vendor_trace", "path": fp.name},
                            )
                        )
                    except Exception as exc:
                        logger.warning("Skipping vendor trace %s: %s", fp, exc)
            except Exception as exc:
                logger.warning("Failed to scan vendor traces: %s", exc)

        # 5. Context log (last 100 entries)
        context_log_path = self.settings.directories.context_log
        if context_log_path.exists():
            try:
                with context_log_path.open(errors="replace") as fh:
                    last_lines = list(collections.deque(fh, maxlen=100))
                combined = "".join(last_lines)
                docs.append(
                    Document(
                        text=combined[:50_000],
                        metadata={"type": "context_log"},
                    )
                )
            except Exception as exc:
                logger.warning("Failed to read context log: %s", exc)

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
        self.persist()

    # ------------------------------------------------------------------
    # Public accessor
    # ------------------------------------------------------------------

    @property
    def index(self) -> Any:
        return self._index
