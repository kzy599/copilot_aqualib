"""Dynamic skill loader – scans a directory and registers discovered skills.

The Clawbio mount point (``skills/clawbio/``) is scanned at runtime.  Every
Python module found is inspected for concrete ``BaseSkill`` subclasses.
Discovered skills are tagged with ``SkillSource.CLAWBIO`` and registered
into the shared ``SkillRegistry``.

This treats the Clawbio library as a *black box*: the framework never
modifies or defines the skills themselves – it only discovers and loads them.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from aqualib.core.message import SkillSource
from aqualib.skills.skill_base import BaseSkill

if TYPE_CHECKING:
    from aqualib.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


def scan_clawbio_directory(directory: Path) -> list[BaseSkill]:
    """Walk *directory* and return instances of every ``BaseSkill`` found.

    Modules are loaded via ``importlib`` so they do not need to be part of
    an installed package.  Each concrete ``BaseSkill`` subclass discovered
    in a ``.py`` file is instantiated once.

    If the directory does not exist or contains no valid skills, an empty
    list is returned – this is intentional so the framework degrades
    gracefully when no external Clawbio library is mounted.
    """
    if not directory.is_dir():
        logger.info("Clawbio mount point %s does not exist – skipping scan.", directory)
        return []

    skills: list[BaseSkill] = []
    py_files = sorted(directory.rglob("*.py"))

    if not py_files:
        logger.info("No Python files found in %s", directory)
        return skills

    for py_file in py_files:
        if py_file.name.startswith("_"):
            # Skip __init__.py and private modules
            continue
        try:
            found = _load_skills_from_file(py_file)
            skills.extend(found)
        except Exception:
            logger.exception("Failed to load skills from %s", py_file)

    logger.info(
        "Clawbio scan complete: %d skill(s) discovered in %s",
        len(skills),
        directory,
    )
    return skills


def mount_clawbio_skills(
    directory: Path,
    registry: "SkillRegistry",
) -> int:
    """Scan *directory* and register every discovered skill.

    Returns the number of skills registered.
    """
    skills = scan_clawbio_directory(directory)
    for skill in skills:
        # Ensure the source is set to Clawbio regardless of what the module
        # declared – this directory is the Clawbio mount point.
        skill.meta.source = SkillSource.CLAWBIO
        registry.register(skill)
    return len(skills)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_skills_from_file(py_file: Path) -> list[BaseSkill]:
    """Import a single Python file and return instances of any BaseSkill subclasses."""
    module_name = f"_clawbio_ext_.{py_file.stem}"

    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        logger.warning("Could not create module spec for %s", py_file)
        return []

    module = importlib.util.module_from_spec(spec)
    # Temporarily add the file's parent to sys.path so intra-library imports
    # within the Clawbio directory can work.
    parent_str = str(py_file.parent)
    added_to_path = parent_str not in sys.path
    if added_to_path:
        sys.path.insert(0, parent_str)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        logger.exception("Error executing module %s", py_file)
        sys.modules.pop(module_name, None)
        return []
    finally:
        if added_to_path and parent_str in sys.path:
            sys.path.remove(parent_str)

    # Discover concrete BaseSkill subclasses defined in this module
    skills: list[BaseSkill] = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, BaseSkill)
            and obj is not BaseSkill
            and not inspect.isabstract(obj)
            and hasattr(obj, "meta")
        ):
            try:
                instance = obj()
                skills.append(instance)
                logger.debug("Discovered Clawbio skill: %s in %s", instance.meta.name, py_file.name)
            except Exception:
                logger.exception("Failed to instantiate %s from %s", _name, py_file)

    return skills
