"""AquaLibClient – CopilotClient lifecycle management.

Wraps the GitHub Copilot SDK ``CopilotClient`` (and ``SubprocessConfig``) so
that the rest of AquaLib only needs to interact with :class:`AquaLibClient`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aqualib.config import Settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AquaLibClient:
    """Manages the lifecycle of a ``CopilotClient`` instance.

    Usage::

        client = AquaLibClient(settings)
        copilot_client = await client.start()
        try:
            ...
        finally:
            await client.stop()
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    async def start(self) -> "Any":
        """Start the Copilot CLI subprocess and return the ``CopilotClient``."""
        try:
            from copilot import CopilotClient, SubprocessConfig  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'github-copilot-sdk' package is required. "
                "Install it with: pip install github-copilot-sdk"
            ) from exc

        config = self._build_config(SubprocessConfig)
        self._client = CopilotClient(config)
        await self._client.start()
        logger.info("CopilotClient started (auth=%s)", self.settings.copilot.auth)
        return self._client

    async def stop(self) -> None:
        """Stop the Copilot CLI subprocess."""
        if self._client is not None:
            await self._client.stop()
            self._client = None
            logger.info("CopilotClient stopped.")

    def _build_config(self, SubprocessConfig: type) -> "Any":  # type: ignore[type-arg]
        """Construct a ``SubprocessConfig`` from the current settings."""
        s = self.settings.copilot
        return SubprocessConfig(
            cli_path=s.cli_path,
            use_stdio=s.use_stdio,
            log_level="debug" if self.settings.verbose else "info",
            github_token=s.github_token or None,
            use_logged_in_user=(s.auth == "github"),
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "Any":
        return await self.start()

    async def __aexit__(self, *args: object) -> None:
        await self.stop()
