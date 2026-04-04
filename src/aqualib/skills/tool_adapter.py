"""Vendor SKILL.md → Copilot SDK @define_tool adapter.

This module converts the AquaLib SKILL.md-discovered vendor skills into
Copilot SDK tool definitions that can be passed to ``create_session(tools=...)``.

The SDK will call these tools in its built-in ReAct loop when the agent decides
to invoke a vendor skill.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.skills.scanner import SkillMeta
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models at module scope (required for get_type_hints in @define_tool)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel
    from pydantic import Field as PydanticField

    class VendorSkillParams(BaseModel):
        parameters: dict = PydanticField(
            default_factory=dict,
            description="Parameters to pass to the vendor skill CLI",
        )

    class SearchParams(BaseModel):
        query: str = PydanticField(description="Search keywords to find in workspace data files")
        max_results: int = PydanticField(default=5, description="Maximum results to return")

    class ReadSkillParams(BaseModel):
        skill_name: str = PydanticField(
            description="Name of the vendor skill to read documentation for"
        )
        include_readme: bool = PydanticField(
            default=False, description="Also read README.md and AGENTS.md if present"
        )

    class RAGSearchParams(BaseModel):
        query: str = PydanticField(description="Semantic search query")
        top_k: int = PydanticField(default=5, description="Number of results")

except ImportError:
    VendorSkillParams = None  # type: ignore[assignment,misc]
    SearchParams = None  # type: ignore[assignment,misc]
    ReadSkillParams = None  # type: ignore[assignment,misc]
    RAGSearchParams = None  # type: ignore[assignment,misc]

def build_tools_from_skills(
    settings: "Settings",
    workspace: "WorkspaceManager",
    session_slug: str | None = None,
) -> list:
    """Scan all SKILL.md files and convert each vendor skill to a Copilot SDK tool.

    Returns a list of tool callables decorated with ``@define_tool`` (or equivalent
    dict specs when the SDK is not installed, for testing purposes).
    """
    from aqualib.skills.scanner import scan_all_skill_dirs

    skill_metas = scan_all_skill_dirs(settings, workspace)
    tools: list = []

    for meta in skill_metas:
        tool = _create_vendor_tool(meta, workspace, session_slug=session_slug)
        tools.append(tool)

    tools.append(_create_workspace_search_tool(workspace))
    tools.append(_create_read_skill_doc_tool(workspace, skill_metas))

    # Auto-detect and register RAG tool if available
    rag_tool = _maybe_create_rag_search_tool(settings, workspace)
    if rag_tool is not None:
        tools.append(rag_tool)

    logger.info("Built %d SDK tools (%d vendor + 2 utility)", len(tools), len(skill_metas))
    return tools


# ---------------------------------------------------------------------------
# Individual tool factories
# ---------------------------------------------------------------------------


def _create_vendor_tool(meta: "SkillMeta", workspace: "WorkspaceManager", session_slug: str | None = None) -> Any:
    """Create a Copilot SDK tool for a single vendor skill."""
    try:
        from copilot import define_tool

        @define_tool(
            name=f"vendor_{meta.name}",
            description=f"[VENDOR] {meta.description}. Tags: {', '.join(meta.tags)}",
        )
        async def vendor_skill_tool(params: VendorSkillParams) -> str:
            return await _run_vendor_skill(meta, workspace, params.parameters, session_slug=session_slug)

        return vendor_skill_tool

    except ImportError:
        # Fallback for environments without the SDK (e.g. test runs without the package)
        return _make_stub_tool(
            name=f"vendor_{meta.name}",
            description=f"[VENDOR] {meta.description}. Tags: {', '.join(meta.tags)}",
            fn=lambda params: _run_vendor_skill(
                meta, workspace, params.get("parameters", {}),
                session_slug=session_slug,
            ),
        )


def _create_workspace_search_tool(workspace: "WorkspaceManager") -> Any:
    """Create a tool that searches workspace data/ for relevant files."""
    try:
        from copilot import define_tool

        @define_tool(
            name="workspace_search",
            description=(
                "Search through project data files (CSV, FASTA, JSON, etc.) in the workspace. "
                "Use this to find relevant data before invoking a vendor skill."
            ),
        )
        async def workspace_search(params: SearchParams) -> str:
            hits = workspace.scan_data_files(params.query, max_files=params.max_results)
            if not hits:
                return "No matching files found in workspace data/."
            return json.dumps(hits, indent=2)

        return workspace_search

    except ImportError:
        return _make_stub_tool(
            name="workspace_search",
            description="Search workspace data files.",
            fn=lambda params: json.dumps(
                workspace.scan_data_files(params.get("query", ""), max_files=params.get("max_results", 5)),
                indent=2,
            ),
        )


def _create_read_skill_doc_tool(workspace: "WorkspaceManager", skill_metas: "list[SkillMeta]") -> Any:
    """Create a progressive-disclosure tool for reading full SKILL.md documents.

    Progressive disclosure levels:
    - Level 1: skill name + description in ``custom_agents`` description
    - Level 2: this tool returns the full SKILL.md content
    - Level 3: README.md / AGENTS.md from the same directory
    """
    try:
        from copilot import define_tool

        @define_tool(
            name="read_skill_doc",
            description=(
                "Read the full SKILL.md documentation for a vendor skill. "
                "Use this to understand parameters, constraints, and usage examples "
                "BEFORE invoking a vendor skill."
            ),
            skip_permission=True,
        )
        async def read_skill_doc(params: ReadSkillParams) -> str:
            return _read_skill_documentation(skill_metas, params.skill_name, params.include_readme)

        return read_skill_doc

    except ImportError:
        return _make_stub_tool(
            name="read_skill_doc",
            description="Read SKILL.md documentation for a vendor skill.",
            fn=lambda params: _read_skill_documentation(
                skill_metas,
                params.get("skill_name", ""),
                params.get("include_readme", False),
            ),
        )


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------


async def _run_vendor_skill(
    meta: "SkillMeta",
    workspace: "WorkspaceManager",
    parameters: dict,
    session_slug: str | None = None,
) -> str:
    """Execute a vendor skill via subprocess and return the result as a string."""
    entry = _resolve_entry_point(meta)
    output_dir = await workspace.next_invocation_dir()
    input_file = output_dir / "input.json"
    output_file = output_dir / "output.json"

    input_file.write_text(json.dumps(parameters, indent=2))

    proc = await asyncio.create_subprocess_exec(
        "python",
        str(entry),
        "run",
        str(input_file),
        "--output",
        str(output_file),
        "--skill",
        meta.name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(meta.vendor_root),
    )
    _VENDOR_TIMEOUT = 300  # seconds

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_VENDOR_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        workspace.save_sdk_vendor_trace(
            meta.name,
            {"returncode": -1, "stdout": "", "stderr": f"Timeout after {_VENDOR_TIMEOUT}s"},
            session_slug=session_slug,
        )
        return f"ERROR: Vendor skill '{meta.name}' timed out after {_VENDOR_TIMEOUT}s."

    workspace.save_sdk_vendor_trace(
        meta.name,
        {
            "returncode": proc.returncode,
            "stdout": stdout.decode(errors="replace")[:2000],
            "stderr": stderr.decode(errors="replace")[:2000],
        },
        session_slug=session_slug,
    )

    if proc.returncode != 0:
        return (
            f"ERROR: Vendor skill '{meta.name}' failed (exit {proc.returncode}): "
            f"{stderr.decode(errors='replace')[:500]}"
        )

    if output_file.exists():
        return output_file.read_text(encoding="utf-8")[:4000]
    return stdout.decode(errors="replace")[:4000]


def _resolve_entry_point(meta: "SkillMeta") -> Path:
    """Locate the vendor CLI entry point in the library root."""
    candidates = [
        meta.vendor_root / "cli.py",
        meta.vendor_root / "main.py",
        meta.vendor_root / "clawbio.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return meta.vendor_root / "cli.py"


def _read_skill_documentation(
    skill_metas: "list[SkillMeta]",
    skill_name: str,
    include_readme: bool,
) -> str:
    """Read and return documentation for a named skill."""
    for meta in skill_metas:
        if meta.name == skill_name:
            result = f"# SKILL.md for {meta.name}\n\n"
            skill_md = meta.skill_dir / "SKILL.md"
            if skill_md.exists():
                result += skill_md.read_text(encoding="utf-8")

            if include_readme:
                for extra_name in ("README.md", "AGENTS.md"):
                    extra = meta.skill_dir / extra_name
                    if extra.exists():
                        result += f"\n\n# {extra_name}\n\n{extra.read_text(encoding='utf-8')}"

            return result[:8000]

    return f"Skill '{skill_name}' not found."


def _make_stub_tool(name: str, description: str, fn: Any) -> dict:
    """Return a plain dict stub when the SDK is not installed (used in tests)."""
    return {"name": name, "description": description, "_fn": fn}


# ---------------------------------------------------------------------------
# RAG auto-detection and tool registration
# ---------------------------------------------------------------------------


def _maybe_create_rag_search_tool(settings: "Settings", workspace: "WorkspaceManager") -> Any:
    """If RAG is configured and llama-index is installed, create a rag_search SDK tool."""
    if not _is_rag_available(settings):
        return None

    try:
        from copilot import define_tool

        @define_tool(
            name="rag_search",
            description=(
                "Semantic search over workspace data files using vector embeddings. "
                "More powerful than workspace_search for conceptual queries. "
                "Use when keyword search returns poor results."
            ),
        )
        async def rag_search(params: RAGSearchParams) -> str:
            return await _execute_rag_search(settings, workspace, params.query, params.top_k)

        return rag_search
    except ImportError:
        return None


def _is_rag_available(settings: "Settings") -> bool:
    """Check if RAG dependencies are installed and configured."""
    try:
        import llama_index.core  # noqa: F401
    except ImportError:
        return False

    rag = settings.rag
    if rag.enabled:
        return True
    # Only activate if user explicitly set a RAG-specific API key
    if rag.api_key and rag.api_key != settings.llm.api_key:
        return True
    return False


async def _execute_rag_search(
    settings: "Settings", workspace: "WorkspaceManager", query: str, top_k: int
) -> str:
    """Execute a RAG query, automatically building/loading the index as needed."""
    from aqualib.rag.indexer import RAGIndexer
    from aqualib.rag.retriever import Retriever
    from aqualib.skills.registry import SkillRegistry

    empty_registry = SkillRegistry()
    indexer = RAGIndexer(settings, empty_registry)
    await indexer.load_or_build()

    if indexer.index is None:
        return "RAG index is empty — no documents to search."

    retriever = Retriever(indexer.index, top_k=top_k)
    results = await retriever.query_summaries(query)
    return json.dumps(results, indent=2, ensure_ascii=False)
