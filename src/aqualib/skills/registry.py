"""Global skill registry with Clawbio-first resolution."""

from __future__ import annotations

import logging
from typing import Optional

from aqualib.core.message import SkillSource
from aqualib.skills.skill_base import BaseSkill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Singleton-ish registry that knows every registered skill.

    Resolution order when ``clawbio_priority`` is ``True``:
      1. Clawbio skills matching the query
      2. Generic / external skills matching the query
    """

    def __init__(self, *, clawbio_priority: bool = True) -> None:
        self._skills: dict[str, BaseSkill] = {}
        self.clawbio_priority = clawbio_priority

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: BaseSkill) -> None:
        key = skill.meta.name.lower()
        if key in self._skills:
            logger.warning("Skill %s is being re-registered – overwriting.", key)
        self._skills[key] = skill
        logger.info("Registered skill: %s [%s]", skill.meta.name, skill.meta.source.value)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name.lower())

    def list_all(self) -> list[BaseSkill]:
        return list(self._skills.values())

    def list_clawbio(self) -> list[BaseSkill]:
        return [s for s in self._skills.values() if s.meta.source == SkillSource.CLAWBIO]

    def list_generic(self) -> list[BaseSkill]:
        return [s for s in self._skills.values() if s.meta.source != SkillSource.CLAWBIO]

    # ------------------------------------------------------------------
    # Resolution (used by the executor to pick the best skill)
    # ------------------------------------------------------------------

    def resolve(self, query: str, tags: Optional[list[str]] = None) -> list[BaseSkill]:
        """Return candidate skills sorted by priority.

        Matching is intentionally simple – the *Searcher* agent will refine
        this via RAG retrieval; the registry only does keyword / tag matching.
        """
        query_lower = query.lower()
        tags_lower = {t.lower() for t in (tags or [])}

        def _score(skill: BaseSkill) -> tuple[int, int]:
            # priority bucket: 0 = clawbio, 1 = other
            bucket = 0 if skill.meta.source == SkillSource.CLAWBIO and self.clawbio_priority else 1
            # relevance: simple keyword overlap
            text = f"{skill.meta.name} {skill.meta.description} {' '.join(skill.meta.tags)}".lower()
            overlap = sum(1 for word in query_lower.split() if word in text)
            tag_match = len(tags_lower & {t.lower() for t in skill.meta.tags})
            return (bucket, -(overlap + tag_match))

        candidates = sorted(self._skills.values(), key=_score)
        return candidates

    # ------------------------------------------------------------------
    # Serialisation helpers  (for RAG indexing)
    # ------------------------------------------------------------------

    def to_descriptions(self) -> list[dict[str, str]]:
        """Return a list of dicts suitable for RAG ingestion."""
        return [
            {
                "name": s.meta.name,
                "source": s.meta.source.value,
                "description": s.meta.description,
                "tags": ", ".join(s.meta.tags),
                "parameters": str(s.meta.parameters_schema),
            }
            for s in self._skills.values()
        ]
