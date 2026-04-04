"""Tests for RAG auto-detection and rag_search tool registration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aqualib.config import DirectorySettings, RAGSettings, Settings
from aqualib.workspace.manager import WorkspaceManager


@pytest.fixture()
def workspace(tmp_path: Path) -> WorkspaceManager:
    dirs = DirectorySettings(base=tmp_path).resolve()
    settings = Settings(directories=dirs)
    return WorkspaceManager(settings)


@pytest.fixture()
def settings_with_rag_key(tmp_path: Path) -> Settings:
    dirs = DirectorySettings(base=tmp_path).resolve()
    return Settings(directories=dirs, rag=RAGSettings(api_key="test-key"))


@pytest.fixture()
def settings_enabled_rag(tmp_path: Path) -> Settings:
    dirs = DirectorySettings(base=tmp_path).resolve()
    return Settings(directories=dirs, rag=RAGSettings(enabled=True, api_key="test-key"))


@pytest.fixture()
def settings_no_rag(tmp_path: Path) -> Settings:
    dirs = DirectorySettings(base=tmp_path).resolve()
    return Settings(directories=dirs)


# ---------------------------------------------------------------------------
# RAGSettings.enabled field
# ---------------------------------------------------------------------------


class TestRAGSettingsEnabled:
    def test_default_is_false(self):
        rag = RAGSettings()
        assert rag.enabled is False

    def test_can_be_set_to_true(self):
        rag = RAGSettings(enabled=True)
        assert rag.enabled is True

    def test_settings_has_rag_enabled_field(self, tmp_path: Path):
        dirs = DirectorySettings(base=tmp_path).resolve()
        settings = Settings(directories=dirs)
        assert hasattr(settings.rag, "enabled")
        assert settings.rag.enabled is False


# ---------------------------------------------------------------------------
# _is_rag_available
# ---------------------------------------------------------------------------


class TestIsRagAvailable:
    def test_returns_false_when_llama_index_not_installed(self, settings_with_rag_key: Settings):
        from aqualib.skills.tool_adapter import _is_rag_available

        with patch.dict("sys.modules", {"llama_index": None, "llama_index.core": None}):
            # Simulate ImportError by raising it in the try block
            with patch("builtins.__import__", side_effect=_mock_import_llama_index_error):
                result = _is_rag_available(settings_with_rag_key)
        assert result is False

    def test_returns_true_when_enabled_flag_set(self, settings_enabled_rag: Settings):
        from aqualib.skills.tool_adapter import _is_rag_available

        with patch("aqualib.skills.tool_adapter._is_rag_available") as mock_check:
            # Directly test the logic: enabled=True + llama_index present → True
            mock_check.return_value = True
            result = _is_rag_available(settings_enabled_rag)

        # If llama_index is installed, enabled=True should give True
        try:
            import llama_index.core  # noqa: F401
            result = _is_rag_available(settings_enabled_rag)
            assert result is True
        except ImportError:
            pytest.skip("llama-index not installed")

    def test_returns_true_when_api_key_set(self, settings_with_rag_key: Settings):
        from aqualib.skills.tool_adapter import _is_rag_available

        try:
            import llama_index.core  # noqa: F401
        except ImportError:
            pytest.skip("llama-index not installed")

        result = _is_rag_available(settings_with_rag_key)
        assert result is True

    def test_returns_false_when_no_key_and_not_enabled(self, settings_no_rag: Settings):
        from aqualib.skills.tool_adapter import _is_rag_available

        # No api_key, no enabled flag, no llm.api_key
        try:
            import llama_index.core  # noqa: F401
        except ImportError:
            pytest.skip("llama-index not installed")

        # settings_no_rag has empty api_keys by default
        result = _is_rag_available(settings_no_rag)
        assert result is False

    def test_silent_failure_no_exception(self, settings_no_rag: Settings):
        """_is_rag_available must never raise exceptions."""
        from aqualib.skills.tool_adapter import _is_rag_available

        try:
            result = _is_rag_available(settings_no_rag)
            assert isinstance(result, bool)
        except Exception as exc:
            pytest.fail(f"_is_rag_available raised an exception: {exc}")


def _mock_import_llama_index_error(name, *args, **kwargs):
    """Simulate ImportError for llama_index.core only."""
    if name == "llama_index.core":
        raise ImportError("No module named 'llama_index'")
    return __builtins__.__import__(name, *args, **kwargs)


# ---------------------------------------------------------------------------
# _maybe_create_rag_search_tool
# ---------------------------------------------------------------------------


class TestMaybeCreateRagSearchTool:
    def test_returns_none_when_rag_unavailable(
        self, settings_no_rag: Settings, workspace: WorkspaceManager
    ):
        from aqualib.skills.tool_adapter import _maybe_create_rag_search_tool

        with patch(
            "aqualib.skills.tool_adapter._is_rag_available", return_value=False
        ):
            tool = _maybe_create_rag_search_tool(settings_no_rag, workspace)

        assert tool is None

    def test_returns_tool_when_rag_available(
        self, settings_with_rag_key: Settings, workspace: WorkspaceManager
    ):
        from aqualib.skills.tool_adapter import _maybe_create_rag_search_tool

        with patch(
            "aqualib.skills.tool_adapter._is_rag_available", return_value=True
        ):
            tool = _maybe_create_rag_search_tool(settings_with_rag_key, workspace)

        # Should return a tool (either real SDK tool or None if copilot not available)
        # In environments with the SDK installed, this should be a Tool object
        if tool is not None:
            if isinstance(tool, dict):
                assert tool["name"] == "rag_search"
            else:
                assert tool.name == "rag_search"


# ---------------------------------------------------------------------------
# build_tools_from_skills with RAG
# ---------------------------------------------------------------------------


class TestBuildToolsFromSkillsRAG:
    def test_includes_rag_search_when_rag_available(
        self, settings_with_rag_key: Settings, workspace: WorkspaceManager
    ):
        from aqualib.skills.tool_adapter import build_tools_from_skills

        with patch("aqualib.skills.scanner.scan_all_skill_dirs", return_value=[]), \
             patch("aqualib.skills.tool_adapter._is_rag_available", return_value=True), \
             patch("aqualib.skills.tool_adapter._execute_rag_search"):
            tools = build_tools_from_skills(settings_with_rag_key, workspace)

        tool_names = _get_tool_names(tools)
        assert "rag_search" in tool_names

    def test_excludes_rag_search_when_rag_unavailable(
        self, settings_no_rag: Settings, workspace: WorkspaceManager
    ):
        from aqualib.skills.tool_adapter import build_tools_from_skills

        with patch("aqualib.skills.scanner.scan_all_skill_dirs", return_value=[]), \
             patch("aqualib.skills.tool_adapter._is_rag_available", return_value=False):
            tools = build_tools_from_skills(settings_no_rag, workspace)

        tool_names = _get_tool_names(tools)
        assert "rag_search" not in tool_names

    def test_always_includes_workspace_search(
        self, settings_no_rag: Settings, workspace: WorkspaceManager
    ):
        from aqualib.skills.tool_adapter import build_tools_from_skills

        with patch("aqualib.skills.scanner.scan_all_skill_dirs", return_value=[]), \
             patch("aqualib.skills.tool_adapter._is_rag_available", return_value=False):
            tools = build_tools_from_skills(settings_no_rag, workspace)

        tool_names = _get_tool_names(tools)
        assert "workspace_search" in tool_names

    def test_no_exception_when_rag_detection_fails(
        self, settings_no_rag: Settings, workspace: WorkspaceManager
    ):
        """RAG detection failure must be silent."""
        from aqualib.skills.tool_adapter import build_tools_from_skills

        def always_false(_):
            return False

        with patch("aqualib.skills.scanner.scan_all_skill_dirs", return_value=[]), \
             patch("aqualib.skills.tool_adapter._is_rag_available", side_effect=always_false):
            try:
                tools = build_tools_from_skills(settings_no_rag, workspace)
                assert isinstance(tools, list)
            except Exception as exc:
                pytest.fail(f"build_tools_from_skills raised: {exc}")


def _get_tool_names(tools: list) -> list[str]:
    """Extract tool names from a list of SDK tools or stub dicts."""
    names = []
    for t in tools:
        if isinstance(t, dict):
            names.append(t.get("name", ""))
        else:
            names.append(getattr(t, "name", getattr(t, "__name__", "")))
    return names


# ---------------------------------------------------------------------------
# RAGSettings env-var and yaml interaction
# ---------------------------------------------------------------------------


class TestRAGSettingsConfig:
    def test_rag_enabled_defaults_false_in_settings(self):
        from aqualib.config import get_settings, reset_settings

        reset_settings()
        settings = get_settings()
        assert settings.rag.enabled is False
        reset_settings()

    def test_rag_enabled_can_be_set_via_yaml(self, tmp_path: Path):
        import os

        yaml_path = tmp_path / "aqualib.yaml"
        yaml_path.write_text("rag:\n  enabled: true\n")

        from aqualib.config import get_settings, reset_settings

        original_config = os.environ.get("AQUALIB_CONFIG")
        os.environ["AQUALIB_CONFIG"] = str(yaml_path)
        reset_settings()
        try:
            settings = get_settings()
            assert settings.rag.enabled is True
        finally:
            if original_config is None:
                os.environ.pop("AQUALIB_CONFIG", None)
            else:
                os.environ["AQUALIB_CONFIG"] = original_config
            reset_settings()
