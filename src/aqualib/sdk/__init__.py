"""Copilot SDK integration layer for AquaLib.

Provides the bridge between AquaLib's workspace management, SKILL.md-driven
vendor skills, and the GitHub Copilot SDK session/tools/hooks system.
"""

from __future__ import annotations

from aqualib.sdk.client import AquaLibClient
from aqualib.sdk.session_manager import SessionManager

__all__ = ["AquaLibClient", "SessionManager"]
