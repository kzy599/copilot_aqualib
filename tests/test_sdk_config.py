"""Tests for the new Copilot SDK configuration (CopilotSettings, ProviderConfig, AzureConfig)."""

from __future__ import annotations

from aqualib.config import (
    AzureConfig,
    CopilotSettings,
    ProviderConfig,
    Settings,
    get_settings,
    reset_settings,
)

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------


def test_copilot_settings_defaults():
    s = CopilotSettings()
    assert s.auth == "github"
    assert s.github_token == ""
    assert s.model == "gpt-4o"
    assert s.reasoning_effort is None
    assert s.streaming is False
    assert s.cli_path is None
    assert s.use_stdio is True


def test_provider_config_defaults():
    p = ProviderConfig()
    assert p.type == "openai"
    assert p.base_url == ""
    assert p.api_key == ""
    assert p.azure is None


def test_azure_config_defaults():
    a = AzureConfig()
    assert a.api_version == "2024-10-21"


def test_settings_has_copilot_field():
    s = Settings()
    assert hasattr(s, "copilot")
    assert isinstance(s.copilot, CopilotSettings)


def test_settings_copilot_default_auth():
    s = Settings()
    assert s.copilot.auth == "github"


# ---------------------------------------------------------------------------
# BYOK provider config
# ---------------------------------------------------------------------------


def test_byok_provider_config_roundtrip():
    p = ProviderConfig(
        type="azure",
        base_url="https://my-azure.openai.azure.com",
        api_key="secret",
        azure=AzureConfig(api_version="2024-05-01-preview"),
    )
    s = Settings(copilot=CopilotSettings(auth="byok", provider=p))
    assert s.copilot.auth == "byok"
    assert s.copilot.provider.type == "azure"
    assert s.copilot.provider.azure.api_version == "2024-05-01-preview"


def test_byok_anthropic_provider():
    p = ProviderConfig(type="anthropic", base_url="https://api.anthropic.com", api_key="ant-key")
    s = Settings(copilot=CopilotSettings(auth="byok", provider=p))
    assert s.copilot.provider.type == "anthropic"
    assert s.copilot.provider.api_key == "ant-key"


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


def test_gh_token_env_var(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "ghp_test_token")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    reset_settings()
    s = get_settings()
    assert s.copilot.github_token == "ghp_test_token"
    reset_settings()


def test_github_token_env_var_fallback(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_test")
    reset_settings()
    s = get_settings()
    assert s.copilot.github_token == "github_pat_test"
    reset_settings()


def test_gh_token_takes_priority_over_github_token(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "primary_token")
    monkeypatch.setenv("GITHUB_TOKEN", "secondary_token")
    reset_settings()
    s = get_settings()
    assert s.copilot.github_token == "primary_token"
    reset_settings()


def test_aqualib_provider_api_key_env_var(monkeypatch):
    monkeypatch.setenv("AQUALIB_PROVIDER_API_KEY", "byok-api-key-123")
    reset_settings()
    s = get_settings()
    assert s.copilot.provider is not None
    assert s.copilot.provider.api_key == "byok-api-key-123"
    reset_settings()


def test_aqualib_provider_base_url_env_var(monkeypatch):
    monkeypatch.setenv("AQUALIB_PROVIDER_BASE_URL", "http://localhost:11434/v1")
    reset_settings()
    s = get_settings()
    assert s.copilot.provider is not None
    assert s.copilot.provider.base_url == "http://localhost:11434/v1"
    reset_settings()


def test_copilot_cli_path_env_var(monkeypatch):
    monkeypatch.setenv("COPILOT_CLI_PATH", "/usr/local/bin/github-copilot-cli")
    reset_settings()
    s = get_settings()
    assert s.copilot.cli_path == "/usr/local/bin/github-copilot-cli"
    reset_settings()


def test_provider_env_vars_not_set_gives_empty_strings(monkeypatch):
    monkeypatch.delenv("AQUALIB_PROVIDER_API_KEY", raising=False)
    monkeypatch.delenv("AQUALIB_PROVIDER_BASE_URL", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    reset_settings()
    s = get_settings()
    assert s.copilot.provider.api_key == ""
    assert s.copilot.provider.base_url == ""
    assert s.copilot.github_token == ""
    reset_settings()


# ---------------------------------------------------------------------------
# Backward-compatibility: existing config keys still work
# ---------------------------------------------------------------------------


def test_legacy_llm_settings_still_available():
    s = Settings()
    assert hasattr(s, "llm")
    assert s.llm.model == "gpt-4o"
    assert s.llm.temperature == 0.2


def test_legacy_rag_settings_still_available():
    s = Settings()
    assert hasattr(s, "rag")
    assert s.rag.chunk_size == 512
    assert s.rag.embed_model == "text-embedding-3-small"


def test_legacy_directories_still_available():
    s = Settings()
    assert hasattr(s, "directories")
    assert s.directories is not None


def test_openai_api_key_still_applies_to_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-openai-key")
    reset_settings()
    s = get_settings()
    assert s.llm.api_key == "legacy-openai-key"
    reset_settings()
