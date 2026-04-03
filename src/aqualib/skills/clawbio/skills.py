"""Example Clawbio skills – placeholders for real implementations.

Each skill follows the ``BaseSkill`` contract and writes its artefacts
into the provided ``output_dir`` so the reviewer can inspect them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aqualib.core.message import SkillSource
from aqualib.skills.skill_base import BaseSkill, SkillMeta


class SequenceAlignmentSkill(BaseSkill):
    """Run a Clawbio sequence-alignment pipeline."""

    meta = SkillMeta(
        name="clawbio_sequence_alignment",
        description="Perform pairwise or multiple sequence alignment using Clawbio algorithms.",
        source=SkillSource.CLAWBIO,
        tags=["alignment", "sequence", "bioinformatics", "clawbio"],
        parameters_schema={
            "sequences": {"type": "array", "items": {"type": "string"}},
            "algorithm": {"type": "string", "default": "needleman-wunsch"},
        },
    )

    async def execute(self, params: dict[str, Any], output_dir: Path) -> Any:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "skill": self.meta.name,
            "algorithm": params.get("algorithm", "needleman-wunsch"),
            "input_count": len(params.get("sequences", [])),
            "status": "completed",
            "alignment_score": 0.95,  # placeholder
        }
        (output_dir / "alignment_result.json").write_text(json.dumps(result, indent=2))
        return result


class StructurePredictionSkill(BaseSkill):
    """Predict protein structure via Clawbio models."""

    meta = SkillMeta(
        name="clawbio_structure_prediction",
        description="Predict 3D protein structure from amino-acid sequence using Clawbio ML models.",
        source=SkillSource.CLAWBIO,
        tags=["structure", "protein", "prediction", "ml", "clawbio"],
        parameters_schema={
            "sequence": {"type": "string"},
            "model_version": {"type": "string", "default": "v2"},
        },
    )

    async def execute(self, params: dict[str, Any], output_dir: Path) -> Any:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "skill": self.meta.name,
            "model_version": params.get("model_version", "v2"),
            "confidence": 0.87,
            "status": "completed",
        }
        (output_dir / "structure_prediction.json").write_text(json.dumps(result, indent=2))
        return result


class GeneExpressionAnalysisSkill(BaseSkill):
    """Analyse gene expression data using Clawbio statistical methods."""

    meta = SkillMeta(
        name="clawbio_gene_expression",
        description="Differential gene-expression analysis with Clawbio normalisation and statistical testing.",
        source=SkillSource.CLAWBIO,
        tags=["gene", "expression", "RNA", "statistics", "clawbio"],
        parameters_schema={
            "dataset_path": {"type": "string"},
            "conditions": {"type": "array", "items": {"type": "string"}},
        },
    )

    async def execute(self, params: dict[str, Any], output_dir: Path) -> Any:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "skill": self.meta.name,
            "dataset": params.get("dataset_path", ""),
            "conditions": params.get("conditions", []),
            "deg_count": 142,
            "status": "completed",
        }
        (output_dir / "gene_expression_result.json").write_text(json.dumps(result, indent=2))
        return result


# Convenience list for auto-registration
ALL_CLAWBIO_SKILLS: list[type[BaseSkill]] = [
    SequenceAlignmentSkill,
    StructurePredictionSkill,
    GeneExpressionAnalysisSkill,
]
