"""Unit tests for the skill registry."""

from aqualib.core.message import SkillSource
from aqualib.skills.clawbio.skills import ALL_CLAWBIO_SKILLS
from aqualib.skills.registry import SkillRegistry
from aqualib.skills.skill_base import BaseSkill, SkillMeta


class _DummyGenericSkill(BaseSkill):
    meta = SkillMeta(
        name="generic_tool",
        description="A generic alignment tool",
        source=SkillSource.GENERIC,
        tags=["alignment", "generic"],
    )

    async def execute(self, params, output_dir):
        return {"ok": True}


def _make_registry(*, clawbio_priority: bool = True) -> SkillRegistry:
    reg = SkillRegistry(clawbio_priority=clawbio_priority)
    for cls in ALL_CLAWBIO_SKILLS:
        reg.register(cls())
    reg.register(_DummyGenericSkill())
    return reg


def test_register_and_list():
    reg = _make_registry()
    assert len(reg.list_all()) == len(ALL_CLAWBIO_SKILLS) + 1
    assert len(reg.list_clawbio()) == len(ALL_CLAWBIO_SKILLS)
    assert len(reg.list_generic()) == 1


def test_get_by_name():
    reg = _make_registry()
    s = reg.get("clawbio_sequence_alignment")
    assert s is not None
    assert s.meta.source == SkillSource.CLAWBIO


def test_resolve_clawbio_first():
    reg = _make_registry(clawbio_priority=True)
    candidates = reg.resolve("alignment")
    # Clawbio alignment skill should come before the generic one
    clawbio_idx = next(
        i for i, s in enumerate(candidates) if s.meta.source == SkillSource.CLAWBIO and "alignment" in s.meta.name
    )
    generic_idx = next(i for i, s in enumerate(candidates) if s.meta.source == SkillSource.GENERIC)
    assert clawbio_idx < generic_idx


def test_resolve_without_priority():
    reg = _make_registry(clawbio_priority=False)
    candidates = reg.resolve("alignment")
    # Without priority, all are in the same bucket – just ensure we get results
    assert len(candidates) == len(ALL_CLAWBIO_SKILLS) + 1


def test_to_descriptions():
    reg = _make_registry()
    descs = reg.to_descriptions()
    assert len(descs) == len(ALL_CLAWBIO_SKILLS) + 1
    assert all("name" in d and "source" in d for d in descs)
