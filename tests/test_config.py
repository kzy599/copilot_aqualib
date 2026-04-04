"""Unit tests for configuration loading."""

from pathlib import Path

from aqualib.config import Settings, get_settings, reset_settings


def test_default_settings():
    s = Settings()
    assert s.vendor_priority is True
    assert s.llm.model == "gpt-4o"
    assert s.rag.chunk_size == 512


def test_directory_resolve(tmp_path: Path):
    from aqualib.config import DirectorySettings

    dirs = DirectorySettings(base=tmp_path).resolve()
    assert dirs.work == (tmp_path / "work").resolve()
    assert dirs.results == (tmp_path / "results").resolve()
    assert dirs.data == (tmp_path / "data").resolve()
    assert dirs.skills_vendor == (tmp_path / "skills" / "vendor").resolve()
    assert dirs.vendor_traces == (tmp_path / "results" / "vendor_traces").resolve()


def test_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AQUALIB_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    reset_settings()
    s = get_settings()
    assert s.directories.base == tmp_path.resolve()
    assert s.llm.api_key == "test-key-123"
    reset_settings()  # cleanup


def test_rag_api_key_falls_back_to_llm(monkeypatch):
    """When no RAG-specific key is set, rag.api_key falls back to llm.api_key."""
    monkeypatch.setenv("OPENAI_API_KEY", "shared-key")
    monkeypatch.delenv("AQUALIB_RAG_API_KEY", raising=False)
    reset_settings()
    s = get_settings()
    assert s.rag.api_key == "shared-key"
    assert s.llm.api_key == "shared-key"
    reset_settings()


def test_rag_api_key_env_override(monkeypatch):
    """AQUALIB_RAG_API_KEY takes precedence over llm.api_key for RAG."""
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("AQUALIB_RAG_API_KEY", "rag-key")
    reset_settings()
    s = get_settings()
    assert s.llm.api_key == "llm-key"
    assert s.rag.api_key == "rag-key"
    reset_settings()


def test_llm_base_url_env_override(monkeypatch):
    """AQUALIB_LLM_BASE_URL env var sets llm.base_url."""
    monkeypatch.setenv("AQUALIB_LLM_BASE_URL", "https://my-llm.example.com")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    reset_settings()
    s = get_settings()
    assert s.llm.base_url == "https://my-llm.example.com"
    reset_settings()


def test_openai_base_url_env_fallback(monkeypatch):
    """OPENAI_BASE_URL env var sets llm.base_url when AQUALIB_LLM_BASE_URL is unset."""
    monkeypatch.delenv("AQUALIB_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-compat.example.com")
    reset_settings()
    s = get_settings()
    assert s.llm.base_url == "https://openai-compat.example.com"
    reset_settings()


def test_rag_base_url_env_override(monkeypatch):
    """AQUALIB_RAG_BASE_URL env var sets rag.base_url."""
    monkeypatch.setenv("AQUALIB_RAG_BASE_URL", "https://embed.example.com")
    reset_settings()
    s = get_settings()
    assert s.rag.base_url == "https://embed.example.com"
    reset_settings()


def test_rag_base_url_stays_none_without_env(monkeypatch):
    """rag.base_url remains None when no env var is set (runtime fallback to llm.base_url)."""
    monkeypatch.delenv("AQUALIB_RAG_BASE_URL", raising=False)
    reset_settings()
    s = get_settings()
    assert s.rag.base_url is None
    reset_settings()


def test_rag_settings_has_credential_fields():
    """RAGSettings exposes api_key and base_url fields."""
    s = Settings()
    assert hasattr(s.rag, "api_key")
    assert hasattr(s.rag, "base_url")
    assert s.rag.api_key == ""
    assert s.rag.base_url is None
