"""SKILL.md scanner for vendor skill discovery.

This module provides the canonical way to discover vendor skills
from SKILL.md files on the file system.  It is used by both the
legacy registry-based pipeline and the new Copilot SDK tool adapter.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

# The SKILL.md parser lives in loader.py and is imported here for
# backward compatibility (scanner is the newer canonical location).
from aqualib.skills.loader import parse_skill_md  # noqa: F401

if TYPE_CHECKING:
    from aqualib.config import Settings
    from aqualib.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SkillMeta (lightweight version for SDK tool building)
# ---------------------------------------------------------------------------


class SkillMeta:
    """Lightweight metadata for a single vendor skill discovered from SKILL.md."""

    __slots__ = ("name", "description", "tags", "version", "parameters_schema", "skill_dir", "vendor_root")

    def __init__(
        self,
        name: str,
        description: str,
        tags: list[str],
        version: str,
        parameters_schema: dict[str, Any],
        skill_dir: Path,
        vendor_root: Path,
    ) -> None:
        self.name = name
        self.description = description
        self.tags = tags
        self.version = version
        self.parameters_schema = parameters_schema
        self.skill_dir = skill_dir
        self.vendor_root = vendor_root

    def __repr__(self) -> str:
        return f"SkillMeta(name={self.name!r}, tags={self.tags!r})"


# ---------------------------------------------------------------------------
# Low-level directory scanner
# ---------------------------------------------------------------------------


def scan_skill_directory(directory: Path) -> list[SkillMeta]:
    """Walk *directory* for ``SKILL.md`` files and return :class:`SkillMeta` instances.

    If the directory does not exist or contains no ``SKILL.md`` files, an
    empty list is returned – the framework degrades gracefully.
    """
    if not directory.exists() or not directory.is_dir():
        logger.info("Skill directory %s does not exist or is not a directory – skipping.", directory)
        return []

    skill_files = sorted(directory.rglob("SKILL.md"))
    if not skill_files:
        logger.info("No SKILL.md files found in %s", directory)
        return []

    skills: list[SkillMeta] = []
    for md_file in skill_files:
        try:
            meta = _load_meta_from_md(md_file, vendor_root=directory)
            if meta is not None:
                skills.append(meta)
        except Exception:
            logger.exception("Failed to load skill metadata from %s", md_file)

    logger.info("Skill scan: %d skill(s) in %s", len(skills), directory)
    return skills


def _load_meta_from_md(md_file: Path, *, vendor_root: Path) -> SkillMeta | None:
    """Parse a single ``SKILL.md`` and return a :class:`SkillMeta`."""
    text = md_file.read_text(encoding="utf-8")
    meta_dict = parse_skill_md(text)

    name = meta_dict.get("name")
    if not name:
        logger.warning("SKILL.md at %s has no name – skipping.", md_file)
        return None

    name = re.sub(r"\s+", "_", name.strip().lower())

    tags = meta_dict.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    parameters = meta_dict.get("parameters", {})
    if isinstance(parameters, str):
        parameters = {}

    return SkillMeta(
        name=name,
        description=meta_dict.get("description", ""),
        tags=tags,
        version=meta_dict.get("version", "0.1.0"),
        parameters_schema=parameters,
        skill_dir=md_file.parent,
        vendor_root=vendor_root,
    )


# ---------------------------------------------------------------------------
# Multi-directory scanner (three-tier priority)
# ---------------------------------------------------------------------------


def scan_all_skill_dirs(settings: "Settings", workspace: "WorkspaceManager") -> list[SkillMeta]:
    """Scan all skill directories and return a deduplicated list of :class:`SkillMeta`.

    Three-tier priority (highest → lowest):
    1. ``workspace/skills/vendor/``  – per-project customisation
    2. ``<repo>/vendor/*/``          – repo-shipped submodule libraries
    3. Internal ``skills/`` directory (not used by scanner, handled elsewhere)

    Skills with duplicate names are resolved in priority order (first wins).
    """
    seen: dict[str, SkillMeta] = {}

    # Tier 1: workspace per-project vendor mount
    ws_vendor = workspace.dirs.skills_vendor
    if ws_vendor.exists():
        for meta in scan_skill_directory(ws_vendor):
            if meta.name not in seen:
                seen[meta.name] = meta

    # Tier 2: repo vendor/ submodule libraries
    repo_vendor = Path(__file__).resolve().parent.parent.parent.parent / "vendor"
    if repo_vendor.is_dir():
        for lib_dir in sorted(repo_vendor.iterdir()):
            if lib_dir.is_dir() and any(lib_dir.rglob("SKILL.md")):
                for meta in scan_skill_directory(lib_dir):
                    if meta.name not in seen:
                        seen[meta.name] = meta

    return list(seen.values())
