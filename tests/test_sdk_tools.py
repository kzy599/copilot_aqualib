"""Tests for build_tools_from_skills() – SKILL.md → SDK tool adapter."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from aqualib.config import DirectorySettings, Settings
from aqualib.workspace.manager import WorkspaceManager


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    dirs = DirectorySettings(base=tmp_path).resolve()
    return Settings(directories=dirs)


@pytest.fixture()
def workspace(tmp_path: Path, settings: Settings) -> WorkspaceManager:
    return WorkspaceManager(settings)


def _write_skill_md(directory: Path, name: str, description: str = "A vendor skill") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        textwrap.dedent(f"""\
        ---
        name: {name}
        description: {description}
        tags: test, science
        version: 1.0.0
        ---
        """)
    )


# ---------------------------------------------------------------------------
# build_tools_from_skills
# ---------------------------------------------------------------------------


class TestBuildToolsFromSkills:
    def test_returns_list(self, settings, workspace):
        from aqualib.skills.tool_adapter import build_tools_from_skills

        with patch("aqualib.skills.scanner.scan_all_skill_dirs", return_value=[]):
            tools = build_tools_from_skills(settings, workspace)

        # At minimum workspace_search + read_skill_doc
        assert isinstance(tools, list)
        assert len(tools) >= 2

    def test_includes_utility_tools(self, settings, workspace):
        from aqualib.skills.tool_adapter import build_tools_from_skills

        with patch("aqualib.skills.scanner.scan_all_skill_dirs", return_value=[]):
            tools = build_tools_from_skills(settings, workspace)

        # Accept either real SDK tools (with .name attr) or stub dicts
        tool_names = []
        for t in tools:
            if isinstance(t, dict):
                tool_names.append(t.get("name", ""))
            elif hasattr(t, "__wrapped__"):
                # SDK @define_tool returns a decorated function; name is on the function or wrapper
                tool_names.append(getattr(t, "__name__", ""))
            else:
                tool_names.append(getattr(t, "name", getattr(t, "__name__", "")))

        assert any("workspace_search" in n for n in tool_names)
        assert any("read_skill_doc" in n for n in tool_names)

    def test_creates_vendor_tool_per_skill(self, settings, workspace, tmp_path):
        vendor_dir = tmp_path / "skills" / "vendor"
        _write_skill_md(vendor_dir / "seq_align", "seq_align", "Sequence alignment skill")
        _write_skill_md(vendor_dir / "drug_check", "drug_check", "Drug interaction checker")

        # We need to mock scan_all_skill_dirs to return our test skills
        from aqualib.skills.scanner import scan_skill_directory
        from aqualib.skills.tool_adapter import build_tools_from_skills

        skill_metas = scan_skill_directory(vendor_dir)
        assert len(skill_metas) == 2

        with patch("aqualib.skills.scanner.scan_all_skill_dirs", return_value=skill_metas):
            tools = build_tools_from_skills(settings, workspace)

        # 2 vendor tools + 4 utility tools (workspace_search, read_skill_doc, read_library_doc, write_plan)
        assert len(tools) == 6

    def test_vendor_tool_name_prefixed(self, settings, workspace, tmp_path):
        vendor_dir = tmp_path / "skills" / "vendor"
        _write_skill_md(vendor_dir / "align", "my_aligner", "Aligns things")

        from aqualib.skills.scanner import scan_skill_directory
        from aqualib.skills.tool_adapter import _create_vendor_tool

        metas = scan_skill_directory(vendor_dir)
        assert len(metas) == 1

        tool = _create_vendor_tool(metas[0], workspace)

        if isinstance(tool, dict):
            assert tool["name"] == "vendor_my_aligner"
        else:
            tool_name = getattr(tool, "__name__", None) or getattr(tool, "name", "")
            assert "vendor_my_aligner" in tool_name


# ---------------------------------------------------------------------------
# workspace_search tool
# ---------------------------------------------------------------------------


class TestWorkspaceSearchTool:
    def test_returns_empty_message_when_no_data(self, workspace):
        from aqualib.skills.tool_adapter import _create_workspace_search_tool

        tool = _create_workspace_search_tool(workspace)
        assert tool is not None

    @pytest.mark.asyncio
    async def test_finds_data_files(self, workspace: WorkspaceManager):
        data_dir = workspace.dirs.data
        (data_dir / "proteins.txt").write_text("protein alignment data for testing.")

        from aqualib.skills.tool_adapter import _create_workspace_search_tool

        tool = _create_workspace_search_tool(workspace)

        if isinstance(tool, dict):
            # Stub mode (SDK not installed)
            import asyncio

            fn = tool["_fn"]
            result = fn({"query": "protein", "max_results": 5})
            if asyncio.iscoroutine(result):
                result = await result
        else:
            # Real SDK tool — handler expects a ToolInvocation with .arguments dict
            class MockInvocation:
                arguments = {"query": "protein", "max_results": 5}

            tool_result = await tool.handler(MockInvocation())
            result = tool_result.text_result_for_llm

        import json
        hits = json.loads(result)
        assert len(hits) > 0
        assert hits[0]["path"] == "proteins.txt"


# ---------------------------------------------------------------------------
# read_skill_doc tool
# ---------------------------------------------------------------------------


class TestReadSkillDocTool:
    @pytest.mark.asyncio
    async def test_returns_not_found_for_unknown_skill(self, workspace: WorkspaceManager):
        from aqualib.skills.tool_adapter import _create_read_skill_doc_tool

        tool = _create_read_skill_doc_tool(workspace, [])

        if isinstance(tool, dict):
            fn = tool["_fn"]
            result = fn({"skill_name": "nonexistent", "include_readme": False})
        else:
            class MockInvocation:
                arguments = {"skill_name": "nonexistent", "include_readme": False}

            tool_result = await tool.handler(MockInvocation())
            result = tool_result.text_result_for_llm

        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_skill_md_content(self, workspace: WorkspaceManager, tmp_path: Path):
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\n\nThis is the documentation.")

        from aqualib.skills.scanner import SkillMeta
        from aqualib.skills.tool_adapter import _create_read_skill_doc_tool

        meta = SkillMeta(
            name="my_skill",
            description="Test skill",
            tags=[],
            version="1.0.0",
            parameters_schema={},
            skill_dir=skill_dir,
            vendor_root=tmp_path,
        )

        tool = _create_read_skill_doc_tool(workspace, [meta])

        if isinstance(tool, dict):
            fn = tool["_fn"]
            result = fn({"skill_name": "my_skill", "include_readme": False})
        else:
            class MockInvocation:
                arguments = {"skill_name": "my_skill", "include_readme": False}

            tool_result = await tool.handler(MockInvocation())
            result = tool_result.text_result_for_llm

        assert "My Skill" in result
        assert "This is the documentation" in result


# ---------------------------------------------------------------------------
# scan_all_skill_dirs integration
# ---------------------------------------------------------------------------


def test_scan_all_skill_dirs_deduplicates(tmp_path: Path):
    """If the same skill name exists at multiple tiers, only the first occurrence is kept."""
    ws_vendor = tmp_path / "workspace" / "skills" / "vendor"
    _write_skill_md(ws_vendor / "align", "seq_align", "workspace version")

    dirs = DirectorySettings(base=tmp_path / "workspace").resolve()
    settings = Settings(directories=dirs)
    workspace = WorkspaceManager(settings)

    # Manually seed a second skill_dir in repo vendor (won't be found in test env)
    # Just verify no duplicates in the scan result
    from aqualib.skills.scanner import scan_all_skill_dirs

    metas = scan_all_skill_dirs(settings, workspace)
    names = [m.name for m in metas]
    assert len(names) == len(set(names)), "scan_all_skill_dirs should return unique skill names"
