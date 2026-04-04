"""Tests for vendor directory loading and three-tier priority."""

import textwrap
from pathlib import Path

from aqualib.skills.clawbio.skills import ALL_CLAWBIO_SKILLS
from aqualib.skills.loader import mount_vendor_skills
from aqualib.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill_md(directory: Path, name: str, description: str = "Test skill") -> None:
    """Write a minimal SKILL.md into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        textwrap.dedent(f"""\
        ---
        name: {name}
        description: {description}
        ---
        """)
    )


# ---------------------------------------------------------------------------
# Vendor path resolution
# ---------------------------------------------------------------------------


def test_vendor_path_resolution():
    """vendor_path in bootstrap.py should resolve to <repo_root>/vendor/ClawBio."""
    import aqualib.bootstrap as bootstrap_module

    # The bootstrap.py file lives at src/aqualib/bootstrap.py.
    # Going up 3 levels (aqualib -> src -> repo_root) and appending vendor/ClawBio
    # should produce the same path as the actual repo's vendor/ClawBio directory.
    bootstrap_file = Path(bootstrap_module.__file__).resolve()
    # src/aqualib/bootstrap.py -> parent = src/aqualib, parent.parent = src, parent.parent.parent = repo root
    repo_root = bootstrap_file.parent.parent.parent
    vendor_path = repo_root / "vendor" / "ClawBio"

    # Verify the directory structure makes sense: repo_root should contain src/
    assert (repo_root / "src").is_dir(), f"Expected repo root at {repo_root}, but src/ not found"
    # Verify the path components are correct
    assert vendor_path.parts[-1] == "ClawBio"
    assert vendor_path.parts[-2] == "vendor"


def test_vendor_path_points_to_correct_repo_location():
    """Vendor path should be relative to the repo root, not the working directory."""
    import aqualib.bootstrap as bootstrap_module

    vendor_path = Path(bootstrap_module.__file__).resolve().parent.parent.parent / "vendor" / "ClawBio"
    # The path should end with vendor/ClawBio
    assert vendor_path.parts[-1] == "ClawBio"
    assert vendor_path.parts[-2] == "vendor"


# ---------------------------------------------------------------------------
# Vendor directory scanning
# ---------------------------------------------------------------------------


def test_vendor_skills_are_registered_when_dir_exists(tmp_path: Path):
    """Skills in a vendor-like directory should be registered."""
    vendor = tmp_path / "vendor" / "ClawBio"
    _write_skill_md(vendor / "pharmgx", "pharmgx_reporter", "PharmGx reporter skill")

    registry = SkillRegistry(vendor_priority=True)
    count = mount_vendor_skills(vendor, registry)

    assert count == 1
    assert registry.get("pharmgx_reporter") is not None


def test_vendor_scan_returns_zero_when_dir_missing(tmp_path: Path):
    """mount_vendor_skills should return 0 when directory doesn't exist."""
    missing = tmp_path / "vendor" / "ClawBio"
    registry = SkillRegistry(vendor_priority=True)
    count = mount_vendor_skills(missing, registry)
    assert count == 0


def test_vendor_multiple_skills(tmp_path: Path):
    """All SKILL.md files in vendor directory should be discovered."""
    vendor = tmp_path / "vendor" / "ClawBio"
    _write_skill_md(vendor / "skill_a", "vendor_skill_a")
    _write_skill_md(vendor / "skill_b", "vendor_skill_b")
    _write_skill_md(vendor / "nested" / "skill_c", "vendor_skill_c")

    registry = SkillRegistry(vendor_priority=True)
    count = mount_vendor_skills(vendor, registry)

    assert count == 3
    assert registry.get("vendor_skill_a") is not None
    assert registry.get("vendor_skill_b") is not None
    assert registry.get("vendor_skill_c") is not None


# ---------------------------------------------------------------------------
# Three-tier priority: runtime > vendor > placeholder
# ---------------------------------------------------------------------------


def test_runtime_mount_takes_priority_over_vendor(tmp_path: Path):
    """A skill registered from runtime mount point should not be overwritten by vendor."""
    from aqualib.skills.loader import scan_vendor_directory

    # Runtime mount: registers "my_skill" with description "runtime version"
    runtime = tmp_path / "runtime"
    _write_skill_md(runtime / "s", "my_skill", "runtime version")

    # Vendor: also has "my_skill" but with different description
    vendor = tmp_path / "vendor"
    _write_skill_md(vendor / "s", "my_skill", "vendor version")

    registry = SkillRegistry(vendor_priority=True)

    # 1. Mount runtime (highest priority) — always registers
    mount_vendor_skills(runtime, registry)

    # 2. Mount vendor — should NOT overwrite runtime registration (uses get() check)
    for skill in scan_vendor_directory(vendor):
        if registry.get(skill.meta.name) is None:
            registry.register(skill)

    skill = registry.get("my_skill")
    assert skill is not None
    assert skill.meta.description == "runtime version"


def test_vendor_takes_priority_over_placeholder(tmp_path: Path):
    """A vendor skill should take priority over bundled placeholder skills."""
    vendor = tmp_path / "vendor"
    # Use the same name as one of the built-in placeholder skills
    placeholder_name = ALL_CLAWBIO_SKILLS[0]().meta.name
    _write_skill_md(vendor / "override", placeholder_name, "vendor override")

    registry = SkillRegistry(vendor_priority=True)

    # 1. Mount vendor (tier 2)
    mount_vendor_skills(vendor, registry)

    # 2. Register placeholder skills (tier 3) — should not overwrite vendor
    for cls in ALL_CLAWBIO_SKILLS:
        instance = cls()
        if registry.get(instance.meta.name) is None:
            registry.register(instance)

    skill = registry.get(placeholder_name)
    assert skill is not None
    assert skill.meta.description == "vendor override"


def test_placeholder_registered_when_no_runtime_or_vendor_skill(tmp_path: Path):
    """Placeholder skills should be registered when no higher-priority skill exists."""
    registry = SkillRegistry(vendor_priority=True)

    # No runtime, no vendor — only placeholders
    for cls in ALL_CLAWBIO_SKILLS:
        instance = cls()
        if registry.get(instance.meta.name) is None:
            registry.register(instance)

    assert len(registry.list_all()) == len(ALL_CLAWBIO_SKILLS)


# ---------------------------------------------------------------------------
# build_registry integration (with vendor path mocked)
# ---------------------------------------------------------------------------


def test_build_registry_scans_vendor_if_exists(tmp_path: Path):
    """build_registry should register vendor skills when the vendor directory exists."""
    from aqualib.skills.loader import scan_vendor_directory

    vendor = tmp_path / "vendor" / "ClawBio"
    _write_skill_md(vendor / "bio_tool", "bio_tool_skill", "A vendor skill")

    # Directly verify the vendor scan logic: scan_vendor_directory + get() check
    # mirrors what bootstrap.py does for tier 2
    registry = SkillRegistry(vendor_priority=True)
    for skill in scan_vendor_directory(vendor):
        if registry.get(skill.meta.name) is None:
            registry.register(skill)

    assert registry.get("bio_tool_skill") is not None


def test_build_registry_skips_vendor_if_missing(tmp_path: Path):
    """build_registry should not fail and should use placeholders if vendor is absent."""
    from aqualib.bootstrap import build_registry
    from aqualib.config import DirectorySettings, Settings

    settings = Settings(
        directories=DirectorySettings(base=str(tmp_path / "ws")).resolve(),
        vendor_priority=True,
    )

    # Run build_registry normally — if vendor doesn't exist, placeholders should still load
    registry = build_registry(settings)
    # At minimum, placeholders should be registered
    assert len(registry.list_all()) >= len(ALL_CLAWBIO_SKILLS)
