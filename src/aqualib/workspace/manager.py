"""Workspace directory manager.

Creates and maintains the canonical directory layout:

    <base>/
    ├── work/                   # scratch / intermediate files
    ├── data/                   # input data & RAG corpus
    ├── skills/
    │   └── clawbio/            # mount point for external Clawbio library
    └── results/
        ├── clawbio_traces/     # standardised logs of every Clawbio invocation
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
from datetime import datetime, timezone
from pathlib import Path

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
            self.dirs.skills_clawbio,
            self.dirs.clawbio_traces,
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
    # Clawbio trace logging
    # ------------------------------------------------------------------

    def save_clawbio_trace(self, task_id: str, invocation: SkillInvocation) -> Path:
        """Write a standardised trace record for a Clawbio skill invocation.

        Every Clawbio execution gets a JSON file under
        ``results/clawbio_traces/<task_id>_<invocation_id>.json``
        so the Reviewer (and humans) can easily inspect the trail.
        """
        trace_dir = self.dirs.clawbio_traces
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
        logger.info("Clawbio trace saved → %s", trace_path)
        return trace_path

    def list_clawbio_traces(self, task_id: str | None = None) -> list[dict]:
        """List Clawbio trace files, optionally filtered by task_id."""
        trace_dir = self.dirs.clawbio_traces
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
