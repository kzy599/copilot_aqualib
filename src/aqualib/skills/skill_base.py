"""Abstract base for every skill (Clawbio or generic)."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aqualib.core.message import SkillSource


class SkillMeta(BaseModel):
    """Declarative metadata attached to every skill."""

    name: str
    description: str
    source: SkillSource = SkillSource.GENERIC
    tags: list[str] = Field(default_factory=list)
    version: str = "0.1.0"
    parameters_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-Schema style dict describing accepted parameters.",
    )


class BaseSkill(abc.ABC):
    """All skills – including Clawbio skills – implement this interface."""

    meta: SkillMeta  # subclass MUST set this

    @abc.abstractmethod
    async def execute(self, params: dict[str, Any], output_dir: Path) -> Any:
        """Run the skill, writing artefacts to *output_dir*.

        Returns an arbitrary result object that will be serialised into the
        audit trail.
        """
        ...

    def describe(self) -> str:
        """Human-friendly one-liner for progressive disclosure."""
        return f"[{self.meta.source.value}] {self.meta.name}: {self.meta.description}"
