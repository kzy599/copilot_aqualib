"""Workspace directory manager.

Creates and maintains the canonical directory layout:

    <base>/
    ├── work/                   # scratch / intermediate files
    ├── data/                   # input data & RAG corpus
    └── results/
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
from pathlib import Path

from aqualib.config import Settings
from aqualib.core.message import AuditReport, Task

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
        for d in (self.dirs.work, self.dirs.results, self.dirs.data):
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
