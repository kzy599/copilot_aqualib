"""Centralised logging configuration for AquaLib."""

from __future__ import annotations

import logging
import sys


def setup_logging(*, verbose: bool = False) -> None:
    """Configure the root logger for console output.

    Call once at startup (CLI / API entrypoint).
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger("aqualib")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
