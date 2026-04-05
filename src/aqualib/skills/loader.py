"""Markdown-driven skill loader for vendor CLI libraries.

Vendor libraries are **CLI-first, Markdown-driven** skill collections.  The
source of truth for each skill is its ``SKILL.md`` file – *not* a Python
module.  Skills are executed via the library's CLI entry point::

    python <entry_point> run <input_file> --output <output_file> --skill <name>

This loader:

1. Recursively scans a vendor mount point for ``SKILL.md`` files.
2. Parses each ``SKILL.md`` to extract structured metadata (name,
   description, tags, parameters schema).
3. Creates a thin :class:`VendorCliSkill` adapter that implements the
   standard :class:`BaseSkill` interface by delegating execution to the
   vendor CLI via ``asyncio.create_subprocess_exec``.

The framework never imports or modifies vendor library internals – it treats
each library as a *black box* discovered purely through its Markdown contracts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from aqualib.core.message import SkillSource
from aqualib.skills.skill_base import BaseSkill, SkillMeta

if TYPE_CHECKING:
    from aqualib.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# Maximum wall-clock time for a single vendor skill invocation.
# Scientific workflows (genome assembly, GWAS, phylogenetics) can run
# for hours; this timeout is a last-resort safeguard, not a task limit.
_VENDOR_TIMEOUT_SECONDS = 43200  # 12 hours

# ---------------------------------------------------------------------------
# SKILL.md parser
# ---------------------------------------------------------------------------

# Regex for a simple YAML-style frontmatter block (``---`` delimited).
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_skill_md(text: str) -> dict[str, Any]:
    """Parse a ``SKILL.md`` file and return structured metadata.

    Supported frontmatter keys (all optional except *name*):
        name, description, version, tags (comma-separated), parameters
        (JSON string or YAML-ish key:value lines).

    If no frontmatter is present the parser falls back to extracting the
    first ``# Heading`` as the name and the following paragraph as the
    description.
    """
    meta: dict[str, Any] = {}

    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match:
        try:
            parsed = yaml.safe_load(fm_match.group(1))
            if isinstance(parsed, dict):
                meta = {k.lower(): v for k, v in parsed.items() if v is not None}
        except yaml.YAMLError:
            logger.warning("Failed to parse SKILL.md frontmatter as YAML, falling back to line parser.")
            # fallback to existing line-by-line parser
            for line in fm_match.group(1).splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "tags":
                    meta["tags"] = [t.strip() for t in value.split(",") if t.strip()]
                elif key == "parameters":
                    try:
                        meta["parameters"] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        meta["parameters"] = value
                else:
                    meta[key] = value

        # Normalize tags: SKILL.md convention is comma-separated string
        if "tags" in meta and isinstance(meta["tags"], str):
            meta["tags"] = [t.strip() for t in meta["tags"].split(",") if t.strip()]

        # Normalize parameters: try JSON parse if string
        if "parameters" in meta and isinstance(meta["parameters"], str):
            try:
                meta["parameters"] = json.loads(meta["parameters"])
            except (json.JSONDecodeError, TypeError):
                pass

        body = text[fm_match.end():]
    else:
        body = text

    # Fallback: extract heading + first paragraph from body
    if "name" not in meta:
        heading = re.search(r"^#\s+(.+)", body, re.MULTILINE)
        if heading:
            meta["name"] = heading.group(1).strip()

    if "description" not in meta:
        # First non-empty paragraph after the heading
        paragraphs = re.split(r"\n{2,}", body.strip())
        for para in paragraphs:
            stripped = para.strip()
            if stripped and not stripped.startswith("#"):
                meta["description"] = stripped.replace("\n", " ")
                break

    return meta


# ---------------------------------------------------------------------------
# CLI adapter skill
# ---------------------------------------------------------------------------


class VendorCliSkill(BaseSkill):
    """Adapter that wraps a single vendor skill discovered via ``SKILL.md``.

    Execution delegates to the vendor CLI entry point via subprocess::

        python <vendor_entry> run <input_file> --output <output_file> \\
            --skill <skill_name>
    """

    def __init__(self, skill_meta: SkillMeta, skill_dir: Path, vendor_root: Path) -> None:
        self.meta = skill_meta
        self._skill_dir = skill_dir
        self._vendor_root = vendor_root

    async def execute(self, params: dict[str, Any], output_dir: Path) -> Any:
        """Run the skill via the vendor CLI, writing artefacts to *output_dir*."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the CLI entry point
        entry = self._resolve_entry_point()

        # Write parameters as a JSON input file for the CLI
        input_file = output_dir / "input_params.json"
        input_file.write_text(json.dumps(params, indent=2))

        output_file = output_dir / "vendor_output.json"

        cmd = [
            "python",
            str(entry),
            "run",
            str(input_file),
            "--output",
            str(output_file),
            "--skill",
            self.meta.name,
        ]

        logger.info("Vendor CLI invocation: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._vendor_root),
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
            timeout_msg = f"Vendor CLI '{self.meta.name}' timed out after {_VENDOR_TIMEOUT_SECONDS}s"
            (output_dir / "cli_stdout.txt").write_text("")
            (output_dir / "cli_stderr.txt").write_text(timeout_msg)
            raise RuntimeError(timeout_msg)

        result: dict[str, Any] = {
            "skill": self.meta.name,
            "returncode": proc.returncode,
            "stdout": stdout.decode(errors="replace") if stdout else "",
            "stderr": stderr.decode(errors="replace") if stderr else "",
        }

        # Persist raw CLI output for traceability
        (output_dir / "cli_stdout.txt").write_text(result["stdout"])
        (output_dir / "cli_stderr.txt").write_text(result["stderr"])

        if proc.returncode != 0:
            raise RuntimeError(
                f"Vendor CLI exited with code {proc.returncode}: {result['stderr'][:500]}"
            )

        # Try to read structured output
        if output_file.exists():
            try:
                result["output"] = json.loads(output_file.read_text())
            except json.JSONDecodeError:
                result["output"] = output_file.read_text()

        result["status"] = "completed"
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_entry_point(self) -> Path:
        """Locate the vendor CLI entry point in the library root."""
        candidates = [
            self._vendor_root / "cli.py",
            self._vendor_root / "main.py",
            self._vendor_root / "clawbio.py",  # legacy ClawBio compat
        ]
        for c in candidates:
            if c.is_file():
                return c
        logger.warning(
            "No CLI entry point found in %s (tried %s) — subprocess will fail.",
            self._vendor_root,
            ", ".join(c.name for c in candidates),
        )
        # Fallback: use root/cli.py even if not yet present (will
        # produce a clear subprocess error at runtime).
        return self._vendor_root / "cli.py"


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------


def scan_vendor_directory(directory: Path) -> list[BaseSkill]:
    """Walk *directory* for ``SKILL.md`` files and return adapter instances.

    If the directory does not exist or contains no ``SKILL.md`` files, an
    empty list is returned – the framework degrades gracefully.
    """
    if not directory.exists():
        logger.info("Vendor mount point %s does not exist – skipping scan.", directory)
        return []
    if not directory.is_dir():
        logger.warning("Vendor mount point %s exists but is not a directory – skipping scan.", directory)
        return []

    skill_files = sorted(directory.rglob("SKILL.md"))

    if not skill_files:
        logger.info("No SKILL.md files found in %s", directory)
        return []

    skills: list[BaseSkill] = []
    for md_file in skill_files:
        try:
            skill = _load_skill_from_md(md_file, vendor_root=directory)
            if skill is not None:
                skills.append(skill)
        except Exception:
            logger.exception("Failed to load skill from %s", md_file)

    logger.info(
        "Vendor scan complete: %d skill(s) discovered in %s",
        len(skills),
        directory,
    )
    return skills


def mount_vendor_skills(
    directory: Path,
    registry: "SkillRegistry",
) -> int:
    """Scan *directory* for ``SKILL.md`` files and register discovered skills.

    Returns the number of skills registered.
    """
    skills = scan_vendor_directory(directory)
    for skill in skills:
        skill.meta.source = SkillSource.VENDOR
        registry.register(skill)
    return len(skills)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_skill_from_md(md_file: Path, *, vendor_root: Path) -> BaseSkill | None:
    """Parse a single ``SKILL.md`` and return a :class:`VendorCliSkill`.

    The skill name extracted from the Markdown is sanitised for use as a
    registry key: whitespace is replaced with underscores and the result is
    lowercased (e.g. ``"Gene Expression"`` → ``"gene_expression"``).
    """
    text = md_file.read_text(encoding="utf-8")
    meta_dict = parse_skill_md(text)

    name = meta_dict.get("name")
    if not name:
        logger.warning("SKILL.md at %s has no name – skipping.", md_file)
        return None

    # Sanitise the name for use as a registry key
    name = re.sub(r"\s+", "_", name.strip().lower())

    description = meta_dict.get("description", "")
    tags = meta_dict.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    version = meta_dict.get("version", "0.1.0")
    parameters = meta_dict.get("parameters", {})
    if isinstance(parameters, str):
        parameters = {}

    skill_meta = SkillMeta(
        name=name,
        description=description,
        source=SkillSource.VENDOR,
        tags=tags,
        version=version,
        parameters_schema=parameters,
    )

    skill = VendorCliSkill(
        skill_meta=skill_meta,
        skill_dir=md_file.parent,
        vendor_root=vendor_root,
    )
    logger.debug("Loaded vendor skill '%s' from %s", name, md_file)
    return skill


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# Keep old names available so existing external code won't break immediately.
ClawBioCliSkill = VendorCliSkill
scan_clawbio_directory = scan_vendor_directory
mount_clawbio_skills = mount_vendor_skills
