#!/usr/bin/env python
"""Example: run the AquaLib pipeline programmatically using the Copilot SDK."""

import asyncio

from aqualib.config import CopilotSettings, DirectorySettings, Settings
from aqualib.sdk.client import AquaLibClient
from aqualib.sdk.session_manager import SessionManager
from aqualib.workspace.manager import WorkspaceManager


async def main():
    # 1. Configure
    settings = Settings(
        directories=DirectorySettings(base="./my_workspace").resolve(),
        copilot=CopilotSettings(auth="github", model="gpt-4o"),
        vendor_priority=True,
    )

    # 2. Create client and session manager
    aqua_client = AquaLibClient(settings)
    client = await aqua_client.start()

    try:
        workspace = WorkspaceManager(settings)
        if workspace.load_project() is None:
            workspace.create_project(name="my_workspace")

        sm = SessionManager(client, settings, workspace)
        session = await sm.get_or_create_session()

        # 3. Run a task and collect results
        done = asyncio.Event()
        result_messages: list[str] = []

        def on_event(event) -> None:
            event_type = getattr(event, "type", None)
            type_val = event_type.value if hasattr(event_type, "value") else str(event_type)
            if type_val == "assistant.message":
                content = getattr(getattr(event, "data", {}), "content", "") or ""
                result_messages.append(content)
            elif type_val == "session.idle":
                done.set()

        session.on(on_event)
        await session.send("Align the protein sequences MVKLF and MVKLT")
        await done.wait()

        # 4. Inspect results
        print(f"Received {len(result_messages)} response(s)")
        for msg in result_messages:
            print(f"  Response: {msg[:200]}")

        workspace.update_project_after_task(
            "Align the protein sequences MVKLF and MVKLT",
            result_messages,
        )

    finally:
        await aqua_client.stop()


if __name__ == "__main__":
    asyncio.run(main())
