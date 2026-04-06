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
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.skills.scanner import SkillMeta
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

# Maximum wall-clock time for a single vendor skill invocation.
# Scientific workflows (genome assembly, GWAS, phylogenetics) can run
# for hours; this timeout is a last-resort safeguard, not a task limit.
_VENDOR_TIMEOUT_SECONDS = 43200  # 12 hours

_MAX_DOC_LENGTH = 8000  # Maximum characters returned by doc-reading tools
_MAX_DOC_FILE_CHARS = 2000  # Per-file character cap when doc_type="all"

# Module-level cache for RAGIndexer instances, keyed by index path.
# Avoids rebuilding/deserializing the vector index on every rag_search call.
_rag_indexer_cache: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Pydantic models at module scope (required for get_type_hints in @define_tool)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel
    from pydantic import Field as PydanticField

    class VendorSkillParams(BaseModel):
        command: str = PydanticField(
            default="",
            description=(
                "Full shell command to execute. Construct this after reading "
                "read_library_doc and read_skill_doc. "
                "Example: 'python clawbio.py run --input /path/to/data.csv "
                "--output /path/to/results.json --trait-pos 3'"
            ),
        )
        parameters: dict = PydanticField(
            default_factory=dict,
            description=(
                "[DEPRECATED] Legacy parameter dict. Use 'command' field instead. "
                "Only used when 'command' is empty."
            ),
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

    class WritePlanParams(BaseModel):
        plan: str = PydanticField(
            description=(
                "The execution plan in Markdown format. Should include: "
                "Goal, Data, Steps (ordered), and Expected Output."
            )
        )

    class ReadLibraryParams(BaseModel):
        library_name: str = PydanticField(
            description="Name of the vendor skill library (directory name under vendor/)"
        )
        doc_type: str = PydanticField(
            default="all",
            description="Which doc to read: 'agents_md', 'readme', 'catalog', 'llms_txt', or 'all'",
        )

except ImportError:
    VendorSkillParams = None  # type: ignore[assignment,misc]
    SearchParams = None  # type: ignore[assignment,misc]
    ReadSkillParams = None  # type: ignore[assignment,misc]
    RAGSearchParams = None  # type: ignore[assignment,misc]
    WritePlanParams = None  # type: ignore[assignment,misc]
    ReadLibraryParams = None  # type: ignore[assignment,misc]

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
    tools.append(_create_read_library_doc_tool())
    tools.append(_create_write_plan_tool(workspace, session_slug))

    # Auto-detect and register RAG tool if available
    rag_tool = _maybe_create_rag_search_tool(settings, workspace)
    if rag_tool is not None:
        tools.append(rag_tool)

    utility_count = len(tools) - len(skill_metas)
    logger.info("Built %d SDK tools (%d vendor + %d utility)", len(tools), len(skill_metas), utility_count)
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
            description=(
                f"[VENDOR] {meta.description}. Tags: {', '.join(meta.tags)}. "
                f"IMPORTANT: First call read_library_doc then read_skill_doc to learn the exact "
                f"CLI syntax, then set 'command' to the full shell command string."
            ),
        )
        async def vendor_skill_tool(params: VendorSkillParams) -> str:
            return await _run_vendor_skill_with_retry(
                meta, workspace,
                command=params.command,
                parameters=params.parameters if not params.command else None,
                session_slug=session_slug,
            )

        return vendor_skill_tool

    except ImportError:
        # Fallback for environments without the SDK (e.g. test runs without the package).
        # Default arguments capture loop variables at definition time, avoiding late binding.
        async def _stub_vendor_fn(params, _meta=meta, _ws=workspace, _slug=session_slug):
            cmd = params.get("command", "")
            legacy_params = params.get("parameters") if not cmd else None
            return await _run_vendor_skill_with_retry(
                _meta, _ws, command=cmd, parameters=legacy_params,
                session_slug=_slug,
            )
        return _make_stub_tool(
            name=f"vendor_{meta.name}",
            description=(
                f"[VENDOR] {meta.description}. Tags: {', '.join(meta.tags)}. "
                f"IMPORTANT: First call read_library_doc then read_skill_doc to learn the exact "
                f"CLI syntax, then set 'command' to the full shell command string."
            ),
            fn=_stub_vendor_fn,
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


def _create_read_library_doc_tool() -> Any:
    """Create a tool for reading top-level documentation of a vendor skill library."""
    try:
        from copilot import define_tool

        @define_tool(
            name="read_library_doc",
            description=(
                "Read top-level documentation of a vendor skill library. "
                "When doc_type='all' (default), reads ALL .md and .txt files in the "
                "library root (AGENTS.md, CLAUDE.md, README.md, llms.txt, etc.) plus "
                "skills/catalog.json. Use this FIRST to understand the library's "
                "architecture, CLI commands, and skill map BEFORE reading individual "
                "SKILL.md files."
            ),
            skip_permission=True,
        )
        async def read_library_doc(params: ReadLibraryParams) -> str:
            return _read_library_documentation(params.library_name, params.doc_type)

        return read_library_doc

    except ImportError:
        return _make_stub_tool(
            name="read_library_doc",
            description="Read top-level documentation of a vendor skill library.",
            fn=lambda params: _read_library_documentation(
                params.get("library_name", ""),
                params.get("doc_type", "all"),
            ),
        )


def _read_library_documentation(library_name: str, doc_type: str) -> str:
    """Read and return top-level documentation for a named vendor library."""
    repo_vendor = Path(__file__).resolve().parent.parent.parent.parent / "vendor"
    lib_dir = repo_vendor / library_name
    if not lib_dir.is_dir():
        return f"Library '{library_name}' not found under vendor/."

    _DOC_FILES = {
        "llms_txt": lib_dir / "llms.txt",
        "agents_md": lib_dir / "AGENTS.md",
        "readme": lib_dir / "README.md",
        "catalog": lib_dir / "skills" / "catalog.json",
    }

    if doc_type != "all":
        doc_path = _DOC_FILES.get(doc_type)
        if doc_path is None:
            return f"Unknown doc_type '{doc_type}'. Choose from: agents_md, readme, catalog, llms_txt, all."
        if not doc_path.exists():
            return f"Document '{doc_type}' not found for library '{library_name}'."
        return doc_path.read_text(encoding="utf-8")[:_MAX_DOC_LENGTH]

    # doc_type == "all": scan all .md/.txt files in root + skills/catalog.json
    parts: list[str] = []
    for f in sorted(lib_dir.iterdir()):
        if f.is_file() and f.suffix in (".md", ".txt"):
            content = f.read_text(encoding="utf-8")[:_MAX_DOC_FILE_CHARS]  # per-file cap
            parts.append(f"# {f.name}\n\n{content}")
    # Also include skills/catalog.json if present
    catalog = lib_dir / "skills" / "catalog.json"
    if catalog.exists():
        parts.append(f"# catalog.json\n\n{catalog.read_text(encoding='utf-8')}")

    combined = "\n\n".join(parts)
    return combined[:_MAX_DOC_LENGTH]


def _create_write_plan_tool(workspace: "WorkspaceManager", session_slug: str | None = None) -> Any:
    """Create a tool that writes the task execution plan to the session directory.

    The plan is written to ``sessions/<slug>/plan.md`` and can be read by
    executor and reviewer agents via the built-in ``read_file`` tool.
    """
    try:
        from copilot import define_tool

        @define_tool(
            name="write_plan",
            description=(
                "Write the task execution plan to the session's plan.md file. "
                "MUST be called before delegating to executor or reviewer agents. "
                "The plan should include: Goal, Data, Steps, and Expected Output."
            ),
            skip_permission=True,
        )
        async def write_plan(params: WritePlanParams) -> str:
            return _write_plan_to_session(workspace, session_slug, params.plan)

        return write_plan

    except ImportError:
        return _make_stub_tool(
            name="write_plan",
            description="Write task execution plan to session plan.md.",
            fn=lambda params: _write_plan_to_session(
                workspace, session_slug, params.get("plan", ""),
            ),
        )


def _write_plan_to_session(
    workspace: "WorkspaceManager",
    session_slug: str | None,
    plan_content: str,
) -> str:
    """Write plan.md to the session directory. Returns the file path for reference."""
    if not session_slug:
        # Fallback: write to workspace root if no session
        plan_path = workspace.dirs.base / "plan.md"
    else:
        plan_dir = workspace.session_dir(session_slug)
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / "plan.md"

    plan_path.write_text(plan_content, encoding="utf-8")
    logger.info("Plan written → %s", plan_path)
    return f"Plan saved to {plan_path}. Executor and reviewer will read this file."


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------


async def _run_vendor_skill_with_retry(
    meta: "SkillMeta",
    workspace: "WorkspaceManager",
    command: str = "",
    parameters: dict | None = None,
    session_slug: str | None = None,
) -> str:
    """Execute a vendor skill once and return the result.

    On failure, returns the ERROR string immediately so the SDK's
    on_error_occurred hook can handle retries with rethink hints,
    prompting the model to re-read docs and construct a different command.
    This avoids the "N retries × N retries" multiplication problem.
    """
    result = await _run_vendor_skill(
        meta, workspace,
        command=command,
        parameters=parameters,
        session_slug=session_slug,
    )

    if result.startswith("ERROR:"):
        logger.warning("Skill '%s' failed: %s", meta.name, result[:200])
        workspace.append_audit_entry({
            "event": "skill_error",
            "skill": meta.name,
            "error_preview": result[:500],
            "session_slug": session_slug,
        })

    return result


async def _run_vendor_skill(
    meta: "SkillMeta",
    workspace: "WorkspaceManager",
    command: str = "",
    parameters: dict | None = None,
    session_slug: str | None = None,
) -> str:
    """Execute a vendor skill via subprocess and return the result as a string.

    If ``command`` is provided, execute it directly via shell with
    ``cwd=meta.vendor_root`` — the model constructs this after reading docs.

    If ``command`` is empty, fall back to the legacy hardcoded argparse-style
    invocation using ``parameters`` (deprecated — a warning is logged).
    """
    output_dir = await workspace.next_invocation_dir(session_slug=session_slug)
    legacy_output_file: "Path | None" = None

    if command:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(meta.vendor_root),
            start_new_session=True,
        )
    else:
        logger.warning(
            "DEPRECATED: Vendor skill '%s' invoked without 'command' field. "
            "Read docs via read_library_doc/read_skill_doc and use 'command' "
            "for model-driven CLI construction.",
            meta.name,
        )
        if parameters is None:
            parameters = {}
        entry = _resolve_entry_point(meta)
        input_file = output_dir / "input.json"
        legacy_output_file = output_dir / "output.json"
        input_file.write_text(json.dumps(parameters, indent=2))

        proc = await asyncio.create_subprocess_exec(
            "python",
            str(entry),
            "run",
            str(input_file),
            "--output",
            str(legacy_output_file),
            "--skill",
            meta.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(meta.vendor_root),
            start_new_session=True,
        )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_VENDOR_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
        else:
            proc.kill()
        await proc.wait()
        workspace.save_sdk_vendor_trace(
            meta.name,
            {"returncode": -1, "stdout": "", "stderr": f"Timeout after {_VENDOR_TIMEOUT_SECONDS}s"},
            session_slug=session_slug,
        )
        return f"ERROR: Vendor skill '{meta.name}' timed out after {_VENDOR_TIMEOUT_SECONDS}s."

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

    # On success, copy output to session results directory for canonical storage
    if session_slug:
        import shutil
        results_dir = workspace.session_results_dir(session_slug)
        output_dest = results_dir / f"{meta.name}_output.json"
        if legacy_output_file is not None and legacy_output_file.exists():
            shutil.copy2(legacy_output_file, output_dest)
        else:
            stdout_text = stdout.decode(errors="replace")
            output_dest.write_text(
                json.dumps({"stdout": stdout_text[:4000]}, indent=2), encoding="utf-8"
            )
        logger.info("Skill output saved → %s", output_dest)

    if legacy_output_file is not None and legacy_output_file.exists():
        return legacy_output_file.read_text(encoding="utf-8")[:4000]
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
    logger.warning(
        "No CLI entry point found in %s (tried %s) — subprocess will fail.",
        meta.vendor_root,
        ", ".join(c.name for c in candidates),
    )
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

            return result[:_MAX_DOC_LENGTH]

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
        # Fallback for environments without the SDK (e.g. test runs).
        async def _stub_rag_fn(params, _s=settings, _w=workspace):
            query = params.get("query", "")
            top_k = params.get("top_k", 5)
            return await _execute_rag_search(_s, _w, query, top_k)

        return _make_stub_tool(
            name="rag_search",
            description=(
                "Semantic search over workspace data files using vector embeddings. "
                "More powerful than workspace_search for conceptual queries. "
                "Use when keyword search returns poor results."
            ),
            fn=_stub_rag_fn,
        )


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

    cache_key = str(settings.directories.work / ".rag_index")

    if cache_key not in _rag_indexer_cache:
        from aqualib.skills.registry import SkillRegistry
        registry = SkillRegistry()
        indexer = RAGIndexer(settings, registry, workspace=workspace)
        await indexer.load_or_build()
        _rag_indexer_cache[cache_key] = indexer

    indexer = _rag_indexer_cache[cache_key]

    if indexer.index is None:
        return "RAG index is empty — no documents to search."

    retriever = Retriever(indexer.index, top_k=top_k)
    results = await retriever.query_summaries(query)
    return json.dumps(results, indent=2, ensure_ascii=False)
