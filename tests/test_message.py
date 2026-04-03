"""Unit tests for the core message / task / audit models."""

from aqualib.core.message import (
    AuditReport,
    Message,
    Role,
    SkillInvocation,
    SkillSource,
    Task,
    TaskStatus,
)


def test_message_creation():
    msg = Message(role=Role.USER, content="hello")
    assert msg.role == Role.USER
    assert msg.content == "hello"
    assert len(msg.id) == 12


def test_task_add_message():
    task = Task(user_query="align sequences")
    msg = task.add_message(Role.EXECUTOR, "Working on it")
    assert len(task.messages) == 1
    assert msg.role == Role.EXECUTOR
    assert task.updated_at is not None


def test_skill_invocation_defaults():
    inv = SkillInvocation(skill_name="test_skill", source=SkillSource.CLAWBIO)
    assert inv.success is False
    assert inv.error is None
    assert len(inv.invocation_id) == 12


def test_audit_report_to_markdown():
    report = AuditReport(
        task_id="abc123",
        user_query="test query",
        status=TaskStatus.APPROVED,
        executor_summary="All good",
        reviewer_verdict="Approved",
        clawbio_priority_check="Satisfied",
        skill_invocations=[
            SkillInvocation(
                skill_name="clawbio_align",
                source=SkillSource.CLAWBIO,
                success=True,
                output_dir="abc123/skills/inv1",
            ),
        ],
    )
    md = report.to_markdown()
    assert "# Audit Report" in md
    assert "abc123" in md
    assert "clawbio_align" in md
    assert "✅" in md


def test_task_status_values():
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.APPROVED.value == "approved"
    assert TaskStatus.NEEDS_REVISION.value == "needs_revision"
