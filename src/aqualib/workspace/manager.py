"""Workspace directory manager.

Creates and maintains the canonical directory layout:

    <base>/
    ├── work/                   # scratch / intermediate files
    ├── data/                   # input data & RAG corpus
    ├── skills/
    │   └── vendor/             # mount point for external vendor libraries
    └── results/
        ├── vendor_traces/      # standardised logs of every vendor skill invocation
        └── <task_id>/
            ├── audit_report.json
            ├── audit_report.md
            └── skills/
                ├── <skill_invocation_id>/   # one sub-dir per invocation
                │   └── ...artefacts...
                └── ...
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union, overload

from aqualib.config import Settings

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Owns the on-disk layout and persistence of audit artefacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dirs = settings.directories
        self._ensure_dirs()
        self._invocation_counter: int = 0
        self._invocation_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        for d in (
            self.dirs.work,
            self.dirs.results,
            self.dirs.data,
            self.dirs.skills_vendor,
            self.dirs.vendor_traces,
        ):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Workspace ready at %s", self.dirs.base)

    def task_dir(self, task_id: str) -> Path:
        p = self.dirs.results / task_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def skills_dir(self, task_id: str) -> Path:
        p = self.task_dir(task_id) / "skills"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def skill_invocation_dir(self, task_id: str, invocation_id: str) -> Path:
        p = self.skills_dir(task_id) / invocation_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    async def next_invocation_dir(self, session_slug: str | None = None) -> Path:
        """Create and return a new sequential invocation directory under work/.

        Used by the SDK tool adapter to provide an isolated scratch space for
        each vendor skill call within the current session. Thread-safe via asyncio.Lock.

        When *session_slug* is provided the canonical directory is created under
        ``sessions/<slug>/work/inv_NNNN/`` (authoritative storage), and a symlink
        is placed at ``work/<slug>/inv_NNNN/`` for an aggregated cross-session
        view.  The canonical path is returned so all tool writes go to the
        session directory.

        When *session_slug* is ``None`` the legacy behaviour
        (``work/inv_NNNN/``) is preserved for backward compatibility.
        """
        async with self._invocation_lock:
            self._invocation_counter += 1
            counter = self._invocation_counter

        if session_slug:
            # Canonical location: session subdirectory
            canonical_dir = (
                self.dirs.base / "sessions" / session_slug / "work"
                / f"inv_{counter:04d}"
            )
            canonical_dir.mkdir(parents=True, exist_ok=True)

            # Aggregated view: symlink in work/<slug>/inv_NNNN → canonical
            # Use a relative path so the symlink remains valid if the workspace is moved.
            link_parent = self.dirs.work / session_slug
            link_parent.mkdir(parents=True, exist_ok=True)
            link_path = link_parent / f"inv_{counter:04d}"
            try:
                import os
                rel_target = Path(os.path.relpath(canonical_dir, link_parent))
                link_path.symlink_to(rel_target)
            except (OSError, NotImplementedError):
                # Symlinks not supported on this system; log and continue
                logger.debug(
                    "Symlink not supported; aggregated view at %s will be absent", link_path
                )

            return canonical_dir
        else:
            inv_dir = self.dirs.work / f"inv_{counter:04d}"
            inv_dir.mkdir(parents=True, exist_ok=True)
            return inv_dir

    # ------------------------------------------------------------------
    # Vendor trace logging
    # ------------------------------------------------------------------

    def save_vendor_trace(
        self,
        task_id: str,
        invocation: "Any",
    ) -> Path:
        """Write a standardised trace record for a vendor skill invocation (legacy path).

        Every vendor execution gets a JSON file under
        ``results/vendor_traces/<task_id>_<invocation_id>.json``.

        For the Copilot SDK path, use :meth:`save_sdk_vendor_trace` instead.
        """

        trace_dir = self.dirs.vendor_traces
        trace_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{task_id}_{invocation.invocation_id}.json"
        trace_path = trace_dir / filename
        trace_data = {
            "task_id": task_id,
            "invocation_id": invocation.invocation_id,
            "skill_name": invocation.skill_name,
            "source": invocation.source.value,
            "parameters": invocation.parameters,
            "output": invocation.output,
            "output_dir": invocation.output_dir,
            "success": invocation.success,
            "error": invocation.error,
            "started_at": invocation.started_at.isoformat(),
            "finished_at": invocation.finished_at.isoformat() if invocation.finished_at else None,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        trace_path.write_text(json.dumps(trace_data, indent=2))
        logger.info("Vendor trace saved → %s", trace_path)
        return trace_path

    def save_sdk_vendor_trace(self, skill_name: str, trace: dict, session_slug: str | None = None) -> Path:
        """Write a vendor trace dict produced by the SDK tool adapter.

        Used by the Copilot SDK integration layer (``sdk/tools.py``) where
        there is no ``SkillInvocation`` object – just a simple dict with the
        subprocess result.

        When *session_slug* is provided the canonical file is written to
        ``sessions/<slug>/vendor_traces/`` and a symlink (or fallback copy) is
        placed in ``results/vendor_traces/`` for aggregated cross-session views.
        The canonical path is returned.

        When *session_slug* is ``None`` the file is written directly to
        ``results/vendor_traces/`` (legacy fallback).
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        invocation_id = uuid.uuid4().hex[:8]
        filename = f"{skill_name}_{ts}_{invocation_id}.json"
        trace_data = {
            "skill_name": skill_name,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **trace,
        }
        serialized = json.dumps(trace_data, indent=2)

        if session_slug:
            # Canonical location: session subdirectory
            canonical_dir = self.session_vendor_traces_dir(session_slug)
            canonical_path = canonical_dir / filename
            canonical_path.write_text(serialized)
            logger.info("SDK vendor trace saved → %s", canonical_path)

            # Aggregated view: symlink in results/vendor_traces/ → canonical
            # Use a relative path so the symlink remains valid if the workspace is moved.
            global_dir = self.dirs.vendor_traces
            global_dir.mkdir(parents=True, exist_ok=True)
            link_path = global_dir / filename
            try:
                import os
                rel_target = Path(os.path.relpath(canonical_path, global_dir))
                link_path.symlink_to(rel_target)
            except (OSError, NotImplementedError):
                # Fall back to a copy on systems without symlink support
                logger.debug(
                    "Symlink not supported; writing copy at %s", link_path
                )
                link_path.write_text(serialized)

            return canonical_path
        else:
            # Legacy path: write directly to results/vendor_traces/
            trace_dir = self.dirs.vendor_traces
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / filename
            trace_path.write_text(serialized)
            logger.info("SDK vendor trace saved → %s", trace_path)
            return trace_path

    # Backward-compatible alias
    save_clawbio_trace = save_vendor_trace

    def list_vendor_traces(self, task_id: str | None = None) -> list[dict]:
        """List vendor trace files, optionally filtered by task_id."""
        trace_dir = self.dirs.vendor_traces
        if not trace_dir.exists():
            return []
        results = []
        for f in sorted(trace_dir.iterdir()):
            if not f.is_file() or not f.suffix == ".json":
                continue
            if task_id and not f.name.startswith(f"{task_id}_"):
                continue
            results.append(json.loads(f.read_text()))
        return results

    # Backward-compatible alias
    list_clawbio_traces = list_vendor_traces

    # ------------------------------------------------------------------
    # Project metadata
    # ------------------------------------------------------------------

    def create_project(self, name: str | None = None, description: str = "") -> dict[str, Any]:
        """Create a new ``project.json`` at the workspace root.

        Returns the project metadata dict.
        """
        project_name = name or self.dirs.base.name
        meta: dict[str, Any] = {
            "project_id": uuid.uuid4().hex[:8],
            "name": project_name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "task_count": 0,
            "tags": [],
            "summary": "",
        }
        self.save_project(meta)
        return meta

    def load_project(self) -> dict[str, Any] | None:
        """Load ``project.json`` from the workspace root, or *None* if absent."""
        pf = self.dirs.project_file
        if not pf.exists():
            return None
        return json.loads(pf.read_text())

    def save_project(self, meta: dict[str, Any]) -> None:
        """Write *meta* to ``project.json``."""
        self.dirs.project_file.write_text(json.dumps(meta, indent=2))

    def update_project(self, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Merge *updates* into ``project.json`` and write it back.

        Returns the updated metadata, or *None* if no project exists.
        This is used by the SDK session manager to store ``session_id``.
        """
        meta = self.load_project()
        if meta is None:
            return None
        meta.update(updates)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_project(meta)
        return meta

    def append_context_log(self, entry: dict[str, Any]) -> None:
        """Append a single JSON line to ``context_log.jsonl``."""
        with open(self.dirs.context_log, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def load_context_log(self, tail: int | None = None) -> list[dict[str, Any]]:
        """Read entries from ``context_log.jsonl``.

        Args:
            tail: If given, return only the last *tail* entries.
        """
        cl = self.dirs.context_log
        if not cl.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in cl.read_text().splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        if tail is not None:
            return entries[-tail:]
        return entries

    def append_audit_entry(self, entry: dict[str, Any]) -> None:
        """Append a hook audit entry to ``context_log.jsonl``.

        Used by the SDK hooks (``on_pre_tool_use``, ``on_post_tool_use``, etc.)
        to maintain a real-time audit trail.
        """
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.append_context_log(entry)

    def build_project_summary(self) -> str:
        """Build a human-readable cumulative summary from ``context_log.jsonl``."""
        entries = self.load_context_log()
        if not entries:
            return ""

        total = len(entries)
        status_counts: Counter[str] = Counter()
        skill_counts: Counter[str] = Counter()
        last_timestamp = ""

        for entry in entries:
            status_counts[entry.get("status", "unknown")] += 1
            for skill in entry.get("skills_used", []):
                skill_counts[skill] += 1
            last_timestamp = entry.get("timestamp", last_timestamp)

        status_parts = [f"{count} {status}" for status, count in status_counts.most_common()]
        skill_parts = [f"{name} ({count}×)" for name, count in skill_counts.most_common()]
        last_date = last_timestamp[:10] if last_timestamp else "unknown"

        return (
            f"{total} tasks completed ({', '.join(status_parts)}). "
            f"Skills used: {', '.join(skill_parts) if skill_parts else 'none'}. "
            f"Last run: {last_date}."
        )

    @overload
    def update_project_after_task(self, task: "Any") -> None: ...

    @overload
    def update_project_after_task(self, task_or_query: str, messages: list | None = None) -> None: ...

    def update_project_after_task(
        self,
        task_or_query: "Union[Any, str]",
        messages: list | None = None,
    ) -> None:
        """Increment counters, append context log, and regenerate summary.

        Accepts two call signatures:

        * **Legacy** (registry-based pipeline):
          ``update_project_after_task(task: Task)``
        * **SDK** (Copilot SDK path):
          ``update_project_after_task(query: str, messages: list)``
        """
        if isinstance(task_or_query, str):
            # SDK path: query + result messages
            self._update_project_after_sdk_task(task_or_query, messages or [])
        else:
            # Legacy path: Task object
            self._update_project_after_legacy_task(task_or_query)

    def _update_project_after_legacy_task(self, task: "Any") -> None:
        """Called after every ``save_task`` in the legacy orchestrator pipeline."""
        meta = self.load_project()
        if meta is None:
            return

        meta["task_count"] = meta.get("task_count", 0) + 1
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()

        skills_used = [inv.skill_name for inv in task.skill_invocations]
        entry: dict[str, Any] = {
            "task_id": task.task_id,
            "query": task.user_query,
            "status": task.status.value,
            "skills_used": skills_used,
            "vendor_priority_satisfied": task.vendor_priority_satisfied or False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.append_context_log(entry)

        meta["summary"] = self.build_project_summary()
        self.save_project(meta)

    def _update_project_after_sdk_task(self, query: str, messages: list) -> None:
        """Called by the SDK CLI path after a session completes."""
        meta = self.load_project()
        if meta is None:
            return

        meta["task_count"] = meta.get("task_count", 0) + 1
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()

        entry: dict[str, Any] = {
            "task_id": uuid.uuid4().hex[:8],
            "query": query,
            "status": "completed",
            "skills_used": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.append_context_log(entry)

        meta["summary"] = self.build_project_summary()
        self.save_project(meta)

    def finalize_task(self) -> None:
        """Post-task cleanup called from the ``on_session_end`` hook.

        Currently a no-op placeholder — future implementations may flush
        buffers, compact logs, or snapshot state.
        """
        logger.info("Task finalised – workspace state is up-to-date.")

    # ------------------------------------------------------------------
    # Data-file scanning (fallback when RAG is unavailable)
    # ------------------------------------------------------------------

    def scan_data_files(
        self,
        query: str,
        *,
        max_files: int = 10,
        max_results: int | None = None,
        max_chars_per_file: int = 500,
        extensions: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Grep-like scan of data/ files for query keywords.

        Returns a list of dicts with file path and matching snippets.
        ``max_results`` is an alias for ``max_files`` (used by SDK tool adapter).
        """
        if max_results is not None:
            max_files = max_results

        if extensions is None:
            extensions = {".txt", ".md", ".json", ".csv", ".yaml", ".yml"}

        data_dir = self.dirs.data
        if not data_dir.exists():
            return []

        keywords = [w.lower() for w in query.split() if len(w) > 2]
        if not keywords:
            return []

        results: list[dict[str, Any]] = []
        for fp in sorted(data_dir.rglob("*")):
            if not fp.is_file() or fp.suffix not in extensions:
                continue
            if fp.stat().st_size > 50_000:
                continue
            try:
                text = fp.read_text(errors="replace")
            except Exception:
                continue

            text_lower = text.lower()
            matched_keywords = [kw for kw in keywords if kw in text_lower]
            if not matched_keywords:
                continue

            first_kw = matched_keywords[0]
            idx = text_lower.find(first_kw)
            start = max(0, idx - 100)
            end = min(len(text), idx + max_chars_per_file)
            snippet = text[start:end].strip()

            results.append({
                "path": str(fp.relative_to(data_dir)),
                "matched_keywords": matched_keywords,
                "keyword_count": len(matched_keywords),
                "snippet": snippet,
            })

            if len(results) >= max_files:
                break

        results.sort(key=lambda r: r["keyword_count"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_audit_report(self, report: "Any") -> Path:
        """Write both JSON and Markdown versions of the audit report."""
        td = self.task_dir(report.task_id)
        json_path = td / "audit_report.json"
        md_path = td / "audit_report.md"

        json_path.write_text(report.model_dump_json(indent=2))
        md_path.write_text(report.to_markdown())
        logger.info("Audit report saved → %s", td)
        return td

    def save_task(self, task: "Any") -> Path:
        """Persist the full task state as JSON."""
        td = self.task_dir(task.task_id)
        path = td / "task_state.json"
        path.write_text(task.model_dump_json(indent=2))
        return path

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def load_task(self, task_id: str) -> "Any | None":
        from aqualib.core.message import Task

        path = self.task_dir(task_id) / "task_state.json"
        if not path.exists():
            return None
        return Task.model_validate_json(path.read_text())

    def list_tasks(self) -> list[str]:
        """Return task IDs that have results directories."""
        if not self.dirs.results.exists():
            return []
        return sorted(
            d.name for d in self.dirs.results.iterdir() if d.is_dir() and (d / "task_state.json").exists()
        )

    def load_audit_report(self, task_id: str) -> "Any | None":
        from aqualib.core.message import AuditReport

        path = self.task_dir(task_id) / "audit_report.json"
        if not path.exists():
            return None
        return AuditReport.model_validate_json(path.read_text())

    def list_skill_outputs(self, task_id: str) -> list[dict]:
        """List all skill invocation sub-directories for a task."""
        sd = self.skills_dir(task_id)
        results = []
        for inv_dir in sorted(sd.iterdir()):
            if inv_dir.is_dir():
                files = [f.name for f in inv_dir.iterdir() if f.is_file()]
                meta_path = inv_dir / "invocation_meta.json"
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                results.append({"invocation_id": inv_dir.name, "files": files, "meta": meta})
        return results

    # ------------------------------------------------------------------
    # Multi-session management
    # ------------------------------------------------------------------

    def _generate_session_slug(self, name: str | None = None) -> str:
        """Generate a URL-friendly slug for a new session."""
        import re

        base = re.sub(r"[^a-z0-9]+", "-", (name or "session").lower()).strip("-")[:30]
        suffix = uuid.uuid4().hex[:8]
        return f"{base}-{suffix}"

    def create_session(self, name: str | None = None) -> dict[str, Any]:
        """Create a new session directory structure and session.json.

        Returns the session metadata dict.
        """
        slug = self._generate_session_slug(name)
        session_dir = self.dirs.base / "sessions" / slug
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "memory").mkdir(exist_ok=True)
        (session_dir / "results").mkdir(exist_ok=True)
        (session_dir / "vendor_traces").mkdir(exist_ok=True)
        (session_dir / "work").mkdir(exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        meta: dict[str, Any] = {
            "session_id": f"aqualib-{slug}",
            "slug": slug,
            "name": name or slug,
            "created_at": now,
            "updated_at": now,
            "task_count": 0,
            "status": "active",
            "summary": "",
        }
        (session_dir / "session.json").write_text(json.dumps(meta, indent=2))

        # Mark as active session in project.json
        self.update_project({"active_session": slug})
        logger.info("Created session %s", slug)
        return meta

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions sorted by most recently updated."""
        sessions_dir = self.dirs.base / "sessions"
        if not sessions_dir.exists():
            return []
        results: list[dict[str, Any]] = []
        for d in sessions_dir.iterdir():
            if d.is_dir() and (d / "session.json").exists():
                try:
                    results.append(json.loads((d / "session.json").read_text()))
                except Exception:
                    pass
        return sorted(results, key=lambda s: s.get("updated_at", ""), reverse=True)

    def load_session(self, slug: str) -> dict[str, Any] | None:
        """Load a session's metadata by slug."""
        path = self.dirs.base / "sessions" / slug / "session.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def find_session_by_prefix(self, prefix: str) -> dict[str, Any] | None:
        """Find a session by slug prefix (returns the most recent match)."""
        for s in self.list_sessions():
            if s["slug"].startswith(prefix):
                return s
        return None

    def get_active_session(self) -> dict[str, Any] | None:
        """Return the active session metadata, or None if not set."""
        project = self.load_project()
        if not project or not project.get("active_session"):
            return None
        return self.load_session(project["active_session"])

    def session_dir(self, slug: str) -> Path:
        """Return the path to a session's directory."""
        return self.dirs.base / "sessions" / slug

    def session_results_dir(self, slug: str) -> Path:
        """Return (and create) the results directory for a session."""
        d = self.session_dir(slug) / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def session_vendor_traces_dir(self, slug: str) -> Path:
        """Return (and create) the vendor_traces directory for a session."""
        d = self.session_dir(slug) / "vendor_traces"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def finalize_session_results(self, slug: str) -> None:
        """Create symlinks in ``results/<slug>/`` pointing to ``sessions/<slug>/results/`` contents.

        Called from the ``on_session_end`` hook after task completion so that
        results are accessible under both the canonical session path and the
        aggregated ``results/`` tree for cross-session views.

        On systems that don't support symlinks the method logs a debug message
        and skips creating the aggregated view.
        """
        session_results = self.session_results_dir(slug)
        results_slug_dir = self.dirs.results / slug
        results_slug_dir.mkdir(parents=True, exist_ok=True)
        import os
        for item in session_results.iterdir():
            link = results_slug_dir / item.name
            if not link.exists() and not link.is_symlink():
                try:
                    # Use a relative path so the symlink remains valid if the workspace is moved.
                    rel_target = Path(os.path.relpath(item, results_slug_dir))
                    link.symlink_to(rel_target)
                except (OSError, NotImplementedError):
                    logger.debug(
                        "Symlink not supported; aggregated result at %s will be absent", link
                    )

    # ------------------------------------------------------------------
    # Agent (role) memory
    # ------------------------------------------------------------------

    def load_agent_memory(self, slug: str, agent_name: str) -> dict[str, Any]:
        """Load the memory for a specific agent within a session.

        Returns an empty memory structure if the file does not exist.
        """
        path = self.session_dir(slug) / "memory" / f"{agent_name}.json"
        if not path.exists():
            return {"agent": agent_name, "session_slug": slug, "entries": []}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {"agent": agent_name, "session_slug": slug, "entries": []}

    def save_agent_memory(self, slug: str, agent_name: str, memory: dict[str, Any]) -> None:
        """Save agent memory, automatically compacting to the most recent 20 entries."""
        entries = memory.get("entries", [])
        if len(entries) > 20:
            memory["entries"] = entries[-20:]
        path = self.session_dir(slug) / "memory" / f"{agent_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(memory, indent=2, ensure_ascii=False))

    def append_agent_memory_entry(
        self, slug: str, agent_name: str, entry: dict[str, Any]
    ) -> None:
        """Append a memory entry for an agent, automatically compacting to 20 entries."""
        memory = self.load_agent_memory(slug, agent_name)
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        memory["entries"].append(entry)
        self.save_agent_memory(slug, agent_name, memory)

    def update_session_after_task(
        self, slug: str, query: str, messages: list, skills_used: list[str] | None = None
    ) -> None:
        """Update session.json counters and summary; also update global project.json."""
        session_meta = self.load_session(slug)
        if session_meta:
            session_meta["task_count"] = session_meta.get("task_count", 0) + 1
            session_meta["updated_at"] = datetime.now(timezone.utc).isoformat()
            session_meta["summary"] = f"Last: {query[:100]}"
            (self.session_dir(slug) / "session.json").write_text(
                json.dumps(session_meta, indent=2, ensure_ascii=False)
            )

        # Update global project.json
        project = self.load_project()
        if project:
            project["task_count"] = project.get("task_count", 0) + 1
            project["updated_at"] = datetime.now(timezone.utc).isoformat()
            project["active_session"] = slug
            project["summary"] = self.build_project_summary()
            self.save_project(project)

        # Append to global context_log with session tag
        self.append_context_log({
            "session_slug": slug,
            "task_id": uuid.uuid4().hex[:8],
            "query": query,
            "status": "completed",
            "skills_used": skills_used or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
