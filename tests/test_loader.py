"""Unit tests for the dynamic Clawbio skill loader."""

import textwrap
from pathlib import Path

import pytest

from aqualib.core.message import SkillSource
from aqualib.skills.loader import mount_clawbio_skills, scan_clawbio_directory
from aqualib.skills.registry import SkillRegistry


def _write_skill_module(directory: Path, filename: str, content: str) -> Path:
    """Write a Python file into the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(textwrap.dedent(content))
    return path


@pytest.fixture()
def clawbio_dir(tmp_path: Path) -> Path:
    """Create a temporary Clawbio mount point with sample skills."""
    d = tmp_path / "skills" / "clawbio"
    _write_skill_module(d, "sample_skill.py", """\
        import json
        from pathlib import Path
        from typing import Any

        from aqualib.core.message import SkillSource
        from aqualib.skills.skill_base import BaseSkill, SkillMeta


        class SampleSkill(BaseSkill):
            meta = SkillMeta(
                name="sample_clawbio_skill",
                description="A sample Clawbio skill for testing.",
                source=SkillSource.CLAWBIO,
                tags=["test", "sample"],
            )

            async def execute(self, params: dict[str, Any], output_dir: Path) -> Any:
                output_dir.mkdir(parents=True, exist_ok=True)
                result = {"status": "ok", "params": params}
                (output_dir / "sample_out.json").write_text(json.dumps(result))
                return result
    """)
    return d


def test_scan_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    skills = scan_clawbio_directory(empty)
    assert skills == []


def test_scan_nonexistent_directory(tmp_path: Path):
    skills = scan_clawbio_directory(tmp_path / "nope")
    assert skills == []


def test_scan_discovers_skill(clawbio_dir: Path):
    skills = scan_clawbio_directory(clawbio_dir)
    assert len(skills) == 1
    assert skills[0].meta.name == "sample_clawbio_skill"
    assert skills[0].meta.source == SkillSource.CLAWBIO


def test_mount_registers_skills(clawbio_dir: Path):
    registry = SkillRegistry(clawbio_priority=True)
    count = mount_clawbio_skills(clawbio_dir, registry)
    assert count == 1
    skill = registry.get("sample_clawbio_skill")
    assert skill is not None
    assert skill.meta.source == SkillSource.CLAWBIO


def test_mount_forces_clawbio_source(tmp_path: Path):
    """Even if the external module sets source=GENERIC, mount overrides to CLAWBIO."""
    d = tmp_path / "skills" / "clawbio"
    _write_skill_module(d, "generic_labelled.py", """\
        from pathlib import Path
        from typing import Any

        from aqualib.core.message import SkillSource
        from aqualib.skills.skill_base import BaseSkill, SkillMeta


        class MislabelledSkill(BaseSkill):
            meta = SkillMeta(
                name="mislabelled_skill",
                description="This claims to be generic but is in the clawbio dir.",
                source=SkillSource.GENERIC,
                tags=["test"],
            )

            async def execute(self, params: dict[str, Any], output_dir: Path) -> Any:
                return {"ok": True}
    """)
    registry = SkillRegistry(clawbio_priority=True)
    mount_clawbio_skills(d, registry)
    skill = registry.get("mislabelled_skill")
    assert skill is not None
    # Source should be forced to CLAWBIO regardless of module declaration
    assert skill.meta.source == SkillSource.CLAWBIO


def test_scan_skips_init_files(tmp_path: Path):
    """Files starting with underscore (like __init__.py) should be skipped."""
    d = tmp_path / "skills" / "clawbio"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("# init")
    (d / "_private.py").write_text("# private")
    skills = scan_clawbio_directory(d)
    assert skills == []


def test_scan_handles_broken_module(tmp_path: Path):
    """A module with a syntax error should be skipped, not crash the loader."""
    d = tmp_path / "skills" / "clawbio"
    d.mkdir(parents=True)
    (d / "broken.py").write_text("this is not valid python !!!")
    skills = scan_clawbio_directory(d)
    assert skills == []


@pytest.mark.asyncio
async def test_discovered_skill_is_executable(clawbio_dir: Path, tmp_path: Path):
    """Verify that a dynamically discovered skill can actually be executed."""
    skills = scan_clawbio_directory(clawbio_dir)
    assert len(skills) == 1
    out_dir = tmp_path / "output"
    result = await skills[0].execute({"key": "value"}, out_dir)
    assert result["status"] == "ok"
    assert (out_dir / "sample_out.json").exists()


def test_scan_nested_directory(tmp_path: Path):
    """Skills in sub-directories of the mount point should also be discovered."""
    d = tmp_path / "skills" / "clawbio"
    sub = d / "submodule"
    _write_skill_module(sub, "nested_skill.py", """\
        from pathlib import Path
        from typing import Any

        from aqualib.core.message import SkillSource
        from aqualib.skills.skill_base import BaseSkill, SkillMeta


        class NestedSkill(BaseSkill):
            meta = SkillMeta(
                name="nested_clawbio_skill",
                description="Nested in a sub-directory.",
                source=SkillSource.CLAWBIO,
                tags=["nested"],
            )

            async def execute(self, params: dict[str, Any], output_dir: Path) -> Any:
                return {"nested": True}
    """)
    skills = scan_clawbio_directory(d)
    assert len(skills) == 1
    assert skills[0].meta.name == "nested_clawbio_skill"
