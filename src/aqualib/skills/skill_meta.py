"""Re-export of SkillMeta for the Copilot SDK path.

Provides both the legacy Pydantic-based SkillMeta (from skill_base)
and the lightweight scanner-based SkillMeta (from scanner).
"""

from __future__ import annotations

# Lightweight dataclass used by the SDK tool adapter
from aqualib.skills.scanner import SkillMeta  # noqa: F401

# Legacy Pydantic model used by the registry-based pipeline
from aqualib.skills.skill_base import SkillMeta as PydanticSkillMeta  # noqa: F401

__all__ = ["PydanticSkillMeta", "SkillMeta"]
