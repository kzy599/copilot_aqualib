"""Unit tests for configuration loading."""

from pathlib import Path

from aqualib.config import Settings, get_settings, reset_settings


def test_default_settings():
    s = Settings()
    assert s.clawbio_priority is True
    assert s.llm.model == "gpt-4o"
    assert s.rag.chunk_size == 512


def test_directory_resolve(tmp_path: Path):
    from aqualib.config import DirectorySettings

    dirs = DirectorySettings(base=tmp_path).resolve()
    assert dirs.work == (tmp_path / "work").resolve()
    assert dirs.results == (tmp_path / "results").resolve()
    assert dirs.data == (tmp_path / "data").resolve()
    assert dirs.skills_clawbio == (tmp_path / "skills" / "clawbio").resolve()
    assert dirs.clawbio_traces == (tmp_path / "results" / "clawbio_traces").resolve()


def test_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AQUALIB_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    reset_settings()
    s = get_settings()
    assert s.directories.base == tmp_path.resolve()
    assert s.llm.api_key == "test-key-123"
    reset_settings()  # cleanup
