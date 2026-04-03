"""Centralised configuration for AquaLib.

All settings are resolved in a single place so that every module can
``from aqualib.config import get_settings`` and receive the same instance.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_BASE = Path.cwd() / "aqualib_workspace"


class DirectorySettings(BaseModel):
    """Strict separation of work / results / data / skills directories."""

    base: Path = Field(default_factory=lambda: _DEFAULT_BASE)
    work: Path = Field(default=Path("work"), description="Intermediate / scratch files")
    results: Path = Field(default=Path("results"), description="Final outputs & audit reports")
    data: Path = Field(default=Path("data"), description="Input data & RAG corpus")
    skills_clawbio: Path = Field(
        default=Path("skills/clawbio"),
        description="Mount point for the external Clawbio skill library.",
    )
    clawbio_traces: Path = Field(
        default=Path("results/clawbio_traces"),
        description="Dedicated trace logs for every Clawbio skill invocation.",
    )

    def resolve(self) -> "DirectorySettings":
        """Return a copy with all paths resolved relative to *base*."""
        return DirectorySettings(
            base=self.base.resolve(),
            work=(self.base / self.work).resolve(),
            results=(self.base / self.results).resolve(),
            data=(self.base / self.data).resolve(),
            skills_clawbio=(self.base / self.skills_clawbio).resolve(),
            clawbio_traces=(self.base / self.clawbio_traces).resolve(),
        )


class LLMSettings(BaseModel):
    """LLM provider configuration – OpenAI-compatible by default."""

    api_key: str = Field(default="", description="API key (falls back to OPENAI_API_KEY env var)")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for the LLM API")
    model: str = Field(default="gpt-4o", description="Model identifier")
    temperature: float = 0.2
    max_tokens: int = 4096


class RAGSettings(BaseModel):
    """Settings for the LlamaIndex-backed RAG pipeline."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    similarity_top_k: int = 5
    embed_model: str = "text-embedding-3-small"


class Settings(BaseModel):
    """Root settings object – the single source of truth."""

    directories: DirectorySettings = Field(default_factory=DirectorySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    clawbio_priority: bool = Field(
        default=True,
        description="When True the framework *always* prefers Clawbio skills over generic tools.",
    )
    verbose: bool = False


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global *Settings* singleton (lazy-initialised)."""
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = _load_settings()
    return _settings


def reset_settings() -> None:
    """Force re-load on next access (useful in tests)."""
    global _settings  # noqa: PLW0603
    _settings = None


def _load_settings() -> Settings:
    """Load settings from ``aqualib.yaml`` (if present) merged with env vars."""
    cfg_path = Path(os.getenv("AQUALIB_CONFIG", "aqualib.yaml"))
    data: dict = {}
    if cfg_path.is_file():
        with open(cfg_path) as fh:
            data = yaml.safe_load(fh) or {}

    settings = Settings(**data)

    # Env-var overrides
    if not settings.llm.api_key:
        settings.llm.api_key = os.getenv("OPENAI_API_KEY", "")
    if os.getenv("AQUALIB_BASE_DIR"):
        settings.directories.base = Path(os.getenv("AQUALIB_BASE_DIR", "."))

    settings.directories = settings.directories.resolve()
    return settings
