"""Unit tests for Clawbio skills."""

from pathlib import Path

import pytest

from aqualib.core.message import SkillSource
from aqualib.skills.clawbio.skills import (
    GeneExpressionAnalysisSkill,
    SequenceAlignmentSkill,
    StructurePredictionSkill,
)


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"


@pytest.mark.asyncio
async def test_sequence_alignment(out_dir: Path):
    skill = SequenceAlignmentSkill()
    assert skill.meta.source == SkillSource.CLAWBIO
    result = await skill.execute(
        {"sequences": ["ATCG", "ATCG"], "algorithm": "needleman-wunsch"},
        out_dir,
    )
    assert result["status"] == "completed"
    assert (out_dir / "alignment_result.json").exists()


@pytest.mark.asyncio
async def test_structure_prediction(out_dir: Path):
    skill = StructurePredictionSkill()
    result = await skill.execute({"sequence": "MVKL", "model_version": "v2"}, out_dir)
    assert result["status"] == "completed"
    assert (out_dir / "structure_prediction.json").exists()


@pytest.mark.asyncio
async def test_gene_expression(out_dir: Path):
    skill = GeneExpressionAnalysisSkill()
    result = await skill.execute(
        {"dataset_path": "/data/expr.csv", "conditions": ["control", "treatment"]},
        out_dir,
    )
    assert result["status"] == "completed"
    assert result["deg_count"] == 142
    assert (out_dir / "gene_expression_result.json").exists()
