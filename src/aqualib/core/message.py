"""Structured message & audit-record types shared by all agents."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    SEARCHER = "searcher"
    ORCHESTRATOR = "orchestrator"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"
    APPROVED = "approved"
    REJECTED = "rejected"


class SkillSource(str, Enum):
    """Where a skill originates – used by the reviewer to enforce priority."""
    VENDOR = "vendor"
    GENERIC = "generic"
    EXTERNAL = "external"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single message exchanged between agents or with the user."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: Role
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Skill invocation record  (written to the audit trail)
# ---------------------------------------------------------------------------

class SkillInvocation(BaseModel):
    """Immutable record of a single skill call – stored under results/<task>/skills/."""

    invocation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    skill_name: str
    source: SkillSource
    parameters: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    output_dir: Optional[str] = None  # relative path under results/<task>/skills/
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Task envelope  – the top-level unit of work
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """Represents one user request flowing through the agent pipeline."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    user_query: str
    status: TaskStatus = TaskStatus.PENDING
    messages: list[Message] = Field(default_factory=list)
    skill_invocations: list[SkillInvocation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None

    # Reviewer notes
    review_passed: Optional[bool] = None
    review_notes: str = ""
    vendor_priority_satisfied: Optional[bool] = None

    def add_message(self, role: Role, content: str, **meta: Any) -> Message:
        msg = Message(role=role, content=content, metadata=meta)
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)
        return msg


# ---------------------------------------------------------------------------
# Audit report  (standard file format for the reviewer)
# ---------------------------------------------------------------------------

class AuditReport(BaseModel):
    """The standard review document produced for *every* task.

    Written as JSON + rendered Markdown so that both models and humans
    can consume it.
    """

    task_id: str
    user_query: str
    status: TaskStatus
    executor_summary: str = ""
    reviewer_verdict: str = ""
    clawbio_priority_check: str = ""  # Deprecated: kept for backward compat with existing reports
    vendor_priority_check: str = ""
    skill_invocations: list[SkillInvocation] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_markdown(self) -> str:
        """Render the audit report as human-readable Markdown."""
        # Use vendor_priority_check if set, fall back to clawbio_priority_check for compat
        priority_check = self.vendor_priority_check or self.clawbio_priority_check
        lines = [
            f"# Audit Report – Task `{self.task_id}`",
            f"**Generated:** {self.generated_at.isoformat()}",
            f"**Status:** {self.status.value}",
            "",
            "## User Query",
            self.user_query,
            "",
            "## Executor Summary",
            self.executor_summary or "_No summary provided._",
            "",
            "## Reviewer Verdict",
            self.reviewer_verdict or "_Pending review._",
            "",
            "## Vendor Priority Check",
            priority_check or "_Not evaluated._",
            "",
            "## Skill Invocations",
        ]
        if not self.skill_invocations:
            lines.append("_No skills invoked._")
        for inv in self.skill_invocations:
            flag = "✅" if inv.success else "❌"
            lines.append(f"- {flag} **{inv.skill_name}** (`{inv.source.value}`) → `{inv.output_dir or 'N/A'}`")
        lines += [
            "",
            "## Conversation Log",
        ]
        for msg in self.messages:
            lines.append(f"- **{msg.role.value}** ({msg.timestamp.isoformat()}): {msg.content[:200]}")
        return "\n".join(lines)
