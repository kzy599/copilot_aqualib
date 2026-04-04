"""Centralised configuration for AquaLib.

All settings are resolved in a single place so that every module can
``from aqualib.config import get_settings`` and receive the same instance.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

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
    skills_vendor: Path = Field(
        default=Path("skills/vendor"),
        description="Mount point for external vendor skill libraries.",
    )
    vendor_traces: Path = Field(
        default=Path("results/vendor_traces"),
        description="Dedicated trace logs for every vendor skill invocation.",
    )

    @property
    def project_file(self) -> Path:
        """Path to the project manifest file."""
        return self.base / "project.json"

    @property
    def context_log(self) -> Path:
        """Path to the context log (JSONL) file."""
        return self.base / "context_log.jsonl"

    def resolve(self) -> "DirectorySettings":
        """Return a copy with all paths resolved relative to *base*."""
        return DirectorySettings(
            base=self.base.resolve(),
            work=(self.base / self.work).resolve(),
            results=(self.base / self.results).resolve(),
            data=(self.base / self.data).resolve(),
            skills_vendor=(self.base / self.skills_vendor).resolve(),
            vendor_traces=(self.base / self.vendor_traces).resolve(),
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

    api_key: str = Field(
        default="",
        description="Embedding API key (falls back to AQUALIB_RAG_API_KEY env var, then llm.api_key)",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for the embedding API "
        "(falls back to AQUALIB_RAG_BASE_URL env var, then llm.base_url)",
    )
    chunk_size: int = 512
    chunk_overlap: int = 64
    similarity_top_k: int = 5
    embed_model: str = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Copilot SDK settings (new in v0.2.0)
# ---------------------------------------------------------------------------


class AzureConfig(BaseModel):
    """Azure-specific configuration for BYOK mode."""

    api_version: str = "2024-10-21"


class ProviderConfig(BaseModel):
    """BYOK (Bring Your Own Key) provider configuration."""

    type: Literal["openai", "azure", "anthropic"] = "openai"
    base_url: str = Field(default="", description="Provider base URL (e.g. http://localhost:11434/v1)")
    api_key: str = Field(default="", description="Provider API key (also reads AQUALIB_PROVIDER_API_KEY)")
    azure: Optional[AzureConfig] = None


class CopilotSettings(BaseModel):
    """GitHub Copilot SDK session and authentication configuration."""

    auth: Literal["github", "token", "byok"] = Field(
        default="github",
        description="Authentication mode: 'github' uses logged-in user, 'token' uses GH_TOKEN, 'byok' uses provider",
    )
    github_token: str = Field(
        default="",
        description="GitHub token for 'token' auth mode (also reads GH_TOKEN / GITHUB_TOKEN)",
    )
    provider: Optional[ProviderConfig] = Field(
        default=None,
        description="BYOK provider config (required when auth='byok')",
    )
    model: str = Field(default="gpt-4o", description="Default model for sessions")
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort: 'low' | 'medium' | 'high' | 'xhigh' | null",
    )
    streaming: bool = Field(default=False, description="Enable streaming responses")
    cli_path: Optional[str] = Field(
        default=None,
        description="Custom Copilot CLI path (also reads COPILOT_CLI_PATH)",
    )
    use_stdio: bool = Field(default=True, description="Use stdio transport (vs TCP)")


class Settings(BaseModel):
    """Root settings object – the single source of truth."""

    copilot: CopilotSettings = Field(default_factory=CopilotSettings)
    directories: DirectorySettings = Field(default_factory=DirectorySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    vendor_priority: bool = Field(
        default=True,
        description="When True the framework *always* prefers vendor skills over generic tools.",
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

    # ---------------------------------------------------------------------------
    # Copilot SDK env-var overrides
    # ---------------------------------------------------------------------------
    if not settings.copilot.github_token:
        settings.copilot.github_token = (
            os.getenv("GH_TOKEN", "") or os.getenv("GITHUB_TOKEN", "")
        )
    if settings.copilot.cli_path is None:
        cli_path_env = os.getenv("COPILOT_CLI_PATH", "")
        if cli_path_env:
            settings.copilot.cli_path = cli_path_env
    if settings.copilot.provider is None:
        settings.copilot.provider = ProviderConfig()
    if not settings.copilot.provider.api_key:
        settings.copilot.provider.api_key = os.getenv("AQUALIB_PROVIDER_API_KEY", "")
    if not settings.copilot.provider.base_url:
        settings.copilot.provider.base_url = os.getenv("AQUALIB_PROVIDER_BASE_URL", "")

    # ---------------------------------------------------------------------------
    # Legacy LLM / RAG env-var overrides (kept for backward compatibility)
    # ---------------------------------------------------------------------------
    if not settings.llm.api_key:
        settings.llm.api_key = os.getenv("OPENAI_API_KEY", "")
    if settings.llm.base_url is None:
        llm_base = os.getenv("AQUALIB_LLM_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", "")
        if llm_base:
            settings.llm.base_url = llm_base
    if os.getenv("AQUALIB_BASE_DIR"):
        settings.directories.base = Path(os.getenv("AQUALIB_BASE_DIR", "."))

    # RAG env-var fallbacks
    if not settings.rag.api_key:
        settings.rag.api_key = os.getenv("AQUALIB_RAG_API_KEY", "") or settings.llm.api_key
    if settings.rag.base_url is None:
        rag_base = os.getenv("AQUALIB_RAG_BASE_URL", "")
        if rag_base:
            settings.rag.base_url = rag_base
        # If still None, leave it — indexer.py will fall back to llm.base_url at runtime

    settings.directories = settings.directories.resolve()
    return settings
