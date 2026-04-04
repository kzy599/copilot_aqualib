"""AquaLib REST API server (FastAPI) — **experimental**.

This module is functional but considered experimental. The CLI (``aqualib``)
is the primary and fully supported interface.

Start with:
    pip install aqualib[api]
    uvicorn aqualib.api:app

.. note::
    The REST API has not been migrated to the Copilot SDK pipeline yet.
    Use the CLI: ``aqualib run '...'``
"""

from __future__ import annotations


def serve() -> None:
    """Start the AquaLib API server (not yet migrated to Copilot SDK)."""
    raise NotImplementedError(
        "The REST API has not been migrated to the Copilot SDK pipeline yet. "
        "Use the CLI: aqualib run '...'"
    )
