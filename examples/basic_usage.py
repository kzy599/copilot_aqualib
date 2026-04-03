#!/usr/bin/env python
"""Example: run the AquaLib pipeline programmatically."""

import asyncio

from aqualib.bootstrap import build_orchestrator
from aqualib.config import Settings, DirectorySettings, LLMSettings


async def main():
    # 1. Configure
    settings = Settings(
        directories=DirectorySettings(base="./my_workspace").resolve(),
        llm=LLMSettings(model="gpt-4o"),
        clawbio_priority=True,
    )

    # 2. Build the orchestrator (wires all agents + RAG)
    orch = await build_orchestrator(settings, skip_rag_index=True)

    # 3. Run a task
    task = await orch.run("Align the protein sequences MVKLF and MVKLT")

    # 4. Inspect results
    print(f"Task ID:  {task.task_id}")
    print(f"Status:   {task.status.value}")
    print(f"Approved: {task.review_passed}")
    for inv in task.skill_invocations:
        print(f"  Skill: {inv.skill_name} ({inv.source.value}) -> {inv.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
