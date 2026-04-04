"""Unit tests for the Markdown-driven vendor skill loader."""

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aqualib.core.message import SkillSource
from aqualib.skills.loader import (
    VendorCliSkill,
    mount_vendor_skills,
    parse_skill_md,
    scan_vendor_directory,
)
from aqualib.skills.registry import SkillRegistry


def _write_skill_md(directory: Path, content: str) -> Path:
    """Write a SKILL.md into the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(textwrap.dedent(content))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vendor_dir(tmp_path: Path) -> Path:
    """Create a temporary vendor mount point with a sample SKILL.md."""
    d = tmp_path / "skills" / "vendor"
    _write_skill_md(
        d / "alignment",
        """\
        ---
        name: sequence_alignment
        description: Perform pairwise or multiple sequence alignment.
        tags: alignment, sequence, bioinformatics
        version: 1.0.0
        ---
        # Sequence Alignment

        Runs a CLI-based alignment pipeline via the vendor entry point.
        """,
    )
    # Also place a dummy CLI entry point
    (d / "cli.py").write_text("# stub entry point")
    return d


# ---------------------------------------------------------------------------
# parse_skill_md
# ---------------------------------------------------------------------------


def test_parse_frontmatter():
    text = textwrap.dedent("""\
    ---
    name: my_skill
    description: A test skill.
    tags: foo, bar, baz
    version: 2.0.0
    ---
    # My Skill

    Extra body text.
    """)
    meta = parse_skill_md(text)
    assert meta["name"] == "my_skill"
    assert meta["description"] == "A test skill."
    assert meta["tags"] == ["foo", "bar", "baz"]
    assert meta["version"] == "2.0.0"


def test_parse_heading_fallback():
    """When no frontmatter is present, extract name from first heading."""
    text = textwrap.dedent("""\
    # Gene Expression

    Analyse differential gene-expression data.
    """)
    meta = parse_skill_md(text)
    assert meta["name"] == "Gene Expression"
    assert "gene-expression" in meta["description"].lower()


def test_parse_empty_file():
    meta = parse_skill_md("")
    assert "name" not in meta


def test_parse_parameters_json():
    text = textwrap.dedent("""\
    ---
    name: with_params
    parameters: {"input": {"type": "string"}}
    ---
    """)
    meta = parse_skill_md(text)
    assert isinstance(meta["parameters"], dict)
    assert "input" in meta["parameters"]


# ---------------------------------------------------------------------------
# scan_vendor_directory
# ---------------------------------------------------------------------------


def test_scan_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    skills = scan_vendor_directory(empty)
    assert skills == []


def test_scan_nonexistent_directory(tmp_path: Path):
    skills = scan_vendor_directory(tmp_path / "nope")
    assert skills == []


def test_scan_discovers_skill(vendor_dir: Path):
    skills = scan_vendor_directory(vendor_dir)
    assert len(skills) == 1
    assert skills[0].meta.name == "sequence_alignment"
    assert skills[0].meta.source == SkillSource.VENDOR


def test_scan_multiple_skills(tmp_path: Path):
    d = tmp_path / "skills" / "vendor"
    _write_skill_md(d / "skill_a", "---\nname: skill_a\ndescription: A\n---\n")
    _write_skill_md(d / "skill_b", "---\nname: skill_b\ndescription: B\n---\n")
    skills = scan_vendor_directory(d)
    assert len(skills) == 2
    names = {s.meta.name for s in skills}
    assert names == {"skill_a", "skill_b"}


def test_scan_nested_skill(tmp_path: Path):
    """SKILL.md in a sub-directory should be discovered."""
    d = tmp_path / "skills" / "vendor"
    _write_skill_md(d / "nested" / "deep", "---\nname: deep_skill\ndescription: deep\n---\n")
    skills = scan_vendor_directory(d)
    assert len(skills) == 1
    assert skills[0].meta.name == "deep_skill"


def test_scan_skips_invalid_md(tmp_path: Path):
    """A SKILL.md with no name should be skipped gracefully."""
    d = tmp_path / "skills" / "vendor" / "bad"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("No frontmatter, no heading, just text.")
    skills = scan_vendor_directory(d.parent.parent)
    assert skills == []


# ---------------------------------------------------------------------------
# mount_vendor_skills
# ---------------------------------------------------------------------------


def test_mount_registers_skills(vendor_dir: Path):
    registry = SkillRegistry(vendor_priority=True)
    count = mount_vendor_skills(vendor_dir, registry)
    assert count == 1
    skill = registry.get("sequence_alignment")
    assert skill is not None
    assert skill.meta.source == SkillSource.VENDOR


def test_mount_forces_vendor_source(tmp_path: Path):
    """Even if SKILL.md doesn't declare source, mount sets it to VENDOR."""
    d = tmp_path / "skills" / "vendor"
    _write_skill_md(d / "s1", "---\nname: some_tool\ndescription: test\n---\n")
    registry = SkillRegistry(vendor_priority=True)
    mount_vendor_skills(d, registry)
    skill = registry.get("some_tool")
    assert skill is not None
    assert skill.meta.source == SkillSource.VENDOR


# ---------------------------------------------------------------------------
# VendorCliSkill adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cli_skill_writes_input_and_calls_subprocess(vendor_dir: Path, tmp_path: Path):
    """Verify the adapter writes input params and calls the CLI."""
    skills = scan_vendor_directory(vendor_dir)
    assert len(skills) == 1
    skill = skills[0]

    out_dir = tmp_path / "output"
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(
        return_value=(json.dumps({"status": "ok"}).encode(), b"")
    )

    with patch("aqualib.skills.loader.asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await skill.execute({"sequences": ["ATCG"]}, out_dir)

    # Input file should have been written
    assert (out_dir / "input_params.json").exists()
    params = json.loads((out_dir / "input_params.json").read_text())
    assert params == {"sequences": ["ATCG"]}

    # subprocess should have been called with expected args
    mock_exec.assert_called_once()
    call_args = mock_exec.call_args[0]
    assert call_args[0] == "python"
    assert "run" in call_args
    assert "--skill" in call_args
    assert "sequence_alignment" in call_args

    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_cli_skill_raises_on_nonzero_exit(vendor_dir: Path, tmp_path: Path):
    """Verify the adapter raises RuntimeError when CLI exits with non-zero code."""
    skills = scan_vendor_directory(vendor_dir)
    skill = skills[0]
    out_dir = tmp_path / "output"

    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"Error: something failed"))

    with patch("aqualib.skills.loader.asyncio.create_subprocess_exec", return_value=mock_process):
        with pytest.raises(RuntimeError, match="Vendor CLI exited with code 1"):
            await skill.execute({}, out_dir)


@pytest.mark.asyncio
async def test_cli_skill_reads_json_output(vendor_dir: Path, tmp_path: Path):
    """When the CLI writes a JSON output file, the adapter should parse it."""
    skills = scan_vendor_directory(vendor_dir)
    skill = skills[0]
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)

    expected_output = {"alignment_score": 0.95, "matches": 42}

    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"ok", b""))

    async def _fake_exec(*args, **kwargs):
        # Simulate the CLI writing the output file
        output_file = out_dir / "vendor_output.json"
        output_file.write_text(json.dumps(expected_output))
        return mock_process

    with patch("aqualib.skills.loader.asyncio.create_subprocess_exec", side_effect=_fake_exec):
        result = await skill.execute({}, out_dir)

    assert result["output"] == expected_output
    assert result["status"] == "completed"


def test_cli_skill_entry_point_resolution(tmp_path: Path):
    """Verify entry point resolution prefers cli.py."""
    d = tmp_path / "vendor_lib"
    d.mkdir()
    (d / "cli.py").write_text("# entry")

    from aqualib.skills.skill_base import SkillMeta

    skill = VendorCliSkill(
        skill_meta=SkillMeta(name="test", description="test", source=SkillSource.VENDOR),
        skill_dir=d,
        vendor_root=d,
    )
    assert skill._resolve_entry_point() == d / "cli.py"
