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

import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aqualib.config import Settings
from aqualib.core.message import AuditReport, SkillInvocation, Task

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Owns the on-disk layout and persistence of audit artefacts."""

    def __init__(self, settings: Settings) -> None:
        self.dirs = settings.directories
        self._ensure_dirs()

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

    # ------------------------------------------------------------------
    # Vendor trace logging
    # ------------------------------------------------------------------

    def save_vendor_trace(self, task_id: str, invocation: SkillInvocation) -> Path:
        """Write a standardised trace record for a vendor skill invocation.

        Every vendor execution gets a JSON file under
        ``results/vendor_traces/<task_id>_<invocation_id>.json``
        so the Reviewer (and humans) can easily inspect the trail.
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

    def append_context_log(self, entry: dict[str, Any]) -> None:
        """Append a single JSON line to ``context_log.jsonl``."""
        with open(self.dirs.context_log, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def load_context_log(self) -> list[dict[str, Any]]:
        """Read all entries from ``context_log.jsonl``."""
        cl = self.dirs.context_log
        if not cl.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in cl.read_text().splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries

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

    def update_project_after_task(self, task: Task) -> None:
        """Increment counters, append context log, and regenerate summary.

        Called after every ``save_task`` in the orchestrator pipeline.
        """
        meta = self.load_project()
        if meta is None:
            return  # no project initialised – skip silently

        meta["task_count"] = meta.get("task_count", 0) + 1
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Build context log entry
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_audit_report(self, report: AuditReport) -> Path:
        """Write both JSON and Markdown versions of the audit report."""
        td = self.task_dir(report.task_id)
        json_path = td / "audit_report.json"
        md_path = td / "audit_report.md"

        json_path.write_text(report.model_dump_json(indent=2))
        md_path.write_text(report.to_markdown())
        logger.info("Audit report saved → %s", td)
        return td

    def save_task(self, task: Task) -> Path:
        """Persist the full task state as JSON."""
        td = self.task_dir(task.task_id)
        path = td / "task_state.json"
        path.write_text(task.model_dump_json(indent=2))
        return path

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def load_task(self, task_id: str) -> Task | None:
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

    def load_audit_report(self, task_id: str) -> AuditReport | None:
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
