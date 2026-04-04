## 修复 AquaLib v0.2.0 已知 Bug 和不足

请修复 `kzy599/copilot_aqualib` 仓库中以下 6 个已确认的问题。每个问题都包含了根因分析和期望的修复方案。

**核心原则**：所有修复必须在 Copilot SDK 的架构边界内完成。SDK 负责 ReAct 循环、上下文 compact、session 持久化/恢复。AquaLib 只通过 SDK 公开接口（hooks、@define_tool、custom_agents、system_message）注入策略。不要在 AquaLib 侧重复实现 SDK 已提供的能力。

---

### Bug 1：`on_session_end` hook 写入的 Executor 记忆内容为空

**文件**：`src/aqualib/sdk/hooks.py` L171-186

**问题**：`on_session_end` hook 的 `input_data` 由 Copilot SDK 传入，SDK 不会在 `input_data` 中提供 `query`、`skills_used`、`summary` 这三个字段。导致写入 executor 记忆的条目全部为空字符串/空列表。

**根因**：SDK 的 `on_session_end` 回调设计用于清理资源，不传递业务上下文。AquaLib 错误地假设了 SDK 会透传这些字段。

**修复方案**：将 executor 记忆的写入从 `on_session_end` hook 移到 `cli.py` 的 `_run()` 函数中，在 `ws.update_session_after_task()` 调用之前。此时 `query` 和 `result_messages` 都是 CLI 层已知的上下文。同时从 hooks.py 的 `_make_session_end_hook` 中删除写入 executor memory 的代码（只保留 `workspace.finalize_task()`）。

**注意**：这不是绕过 SDK hooks，而是把"业务状态持久化"放到正确的层——CLI 编排层拥有完整的请求上下文（query），而 SDK hooks 只应做与 SDK 生命周期相关的事（资源清理、审计日志）。

在 `cli.py` 中，在 `ws.update_session_after_task(actual_slug, query, result_messages)` 之前添加：
```python
# Write executor memory — CLI layer has the query context that SDK hooks don't
ws.append_agent_memory_entry(actual_slug, "executor", {
    "query": query,
    "skills_used": [],  # populated by Bug 5 fix below
    "output_preview": (result_messages[-1][:200] if result_messages else ""),
})
```

同步更新 `tests/test_agent_memory.py` 中的 `TestSessionEndHookMemory` 测试类：
- 验证 `on_session_end` hook 不再写入 executor memory（只调用 `finalize_task()`）
- 新增一个独立测试验证 CLI 层的 executor memory 写入逻辑

---

### Bug 2：`_is_rag_available()` 检测逻辑过于宽松

**文件**：`src/aqualib/skills/tool_adapter.py` L308-320

**问题**：当用户只配置了 `OPENAI_API_KEY`（用于 Copilot SDK BYOK 模式）但不想启用 RAG 时，`_is_rag_available()` 仍会返回 True（因为 `settings.llm.api_key` 非空），导致意外通过 `@define_tool` 注册 `rag_search` 工具到 SDK session，并在 LLM 首次调用时尝试构建向量索引。

**根因**：检测逻辑把 LLM API key 当作 RAG 可用的信号，但两者职责不同——LLM key 驱动 Copilot SDK 的 ReAct 循环，RAG key 驱动 LlamaIndex 的 embedding 构建。

**修复方案**：收紧检测逻辑——只有当 `rag.enabled == True` **或** `rag.api_key` 被用户单独设置（不同于 llm.api_key 回退值）时才返回 True：

```python
def _is_rag_available(settings: "Settings") -> bool:
    try:
        import llama_index.core  # noqa: F401
    except ImportError:
        return False
    rag = settings.rag
    if rag.enabled:
        return True
    # Only activate if user explicitly set a RAG-specific API key
    if rag.api_key and rag.api_key != settings.llm.api_key:
        return True
    return False
```

同步更新 `tests/test_rag_auto.py` 中的 `TestIsRagAvailable`，新增测试用例验证：当 `rag.api_key` 为空但 `llm.api_key` 非空时，`_is_rag_available()` 返回 False。

---

### Bug 3：`aqualib status` 命令不显示 Session 信息

**文件**：`src/aqualib/cli.py` L418-479

**问题**：`status` 命令只显示项目级别信息（task_count、data files、skills），完全没有展示 session 列表和活跃 session。用户无法通过 `status` 命令了解当前有哪些 SDK session 可供 `resume_session()` 恢复。

**修复方案**：在 `status` 命令的输出中，在 "Skills" 行之后、"Recent tasks" 之前，增加 session 信息展示。这里读取的是 AquaLib 自己管理的 `sessions/*/session.json` 元数据（slug → SDK session_id 的映射），不涉及 SDK 内部的 session 状态：

```python
# Session information (AquaLib workspace metadata, not SDK internal state)
all_sessions = ws.list_sessions()
active = ws.get_active_session()
if all_sessions:
    active_slug = active["slug"] if active else "none"
    rprint(f"   [bold]Sessions:[/bold] {len(all_sessions)} total, active: {active_slug}")
    for s in all_sessions[:3]:  # show top 3 most recent
        indicator = "▶ " if s["slug"] == active_slug else "  "
        rprint(
            f"     {indicator}{s.get('name', s['slug'])} "
            f"({s.get('task_count', 0)} tasks, {s.get('updated_at', '')[:10]})"
        )
    if len(all_sessions) > 3:
        rprint(f"     ... and {len(all_sessions) - 3} more (use 'aqualib sessions' to see all)")
```

---

### Bug 4：Vendor trace 没有写入 session 级目录

**文件**：`src/aqualib/skills/tool_adapter.py` L217-224 和 `src/aqualib/workspace/manager.py` L128-148

**问题**：`save_sdk_vendor_trace()` 始终写入全局 `results/vendor_traces/`，但 `WorkspaceManager` 已经提供了 `session_vendor_traces_dir(slug)` 方法。不同 session 的 trace 混在一起，无法区分。

**根因**：vendor tool 通过 `@define_tool` 注册到 SDK session 后，由 SDK ReAct 循环驱动调用。tool 函数内部通过闭包访问 `workspace` 对象来写 trace，但没有 session 上下文。

**修复方案**：通过闭包将 `session_slug` 传入 vendor tool 函数。这不改变 SDK 的 tool 签名（SDK 只关心 `@define_tool` 的 name/description/params），AquaLib 只是在闭包环境中多捕获一个变量：

1. 给 `build_tools_from_skills()` 增加可选的 `session_slug: str | None = None` 参数
2. 在 `_create_vendor_tool()` 中通过闭包捕获 `session_slug`
3. 给 `save_sdk_vendor_trace()` 增加可选的 `session_slug` 参数，当提供时同时写入 session 级目录
4. 在 `SessionManager._create_sdk_session()` 中调用 `build_tools_from_skills()` 时传入 `slug`

`save_sdk_vendor_trace()` 的实现——全局 trace 保持不变（向后兼容），session 级 trace 作为补充：

```python
def save_sdk_vendor_trace(self, skill_name: str, trace: dict, session_slug: str | None = None) -> Path:
    # ... existing code writes to global vendor_traces/ (unchanged) ...
    
    # Additionally write to session-specific directory if slug provided
    if session_slug:
        session_trace_dir = self.session_vendor_traces_dir(session_slug)
        session_trace_path = session_trace_dir / filename
        session_trace_path.write_text(json.dumps(trace_data, indent=2))
    
    return trace_path
```

---

### Bug 5：`update_session_after_task()` 中 `skills_used` 始终为空列表

**文件**：`src/aqualib/workspace/manager.py` L607-637

**问题**：`update_session_after_task()` 写入 `context_log.jsonl` 时，`skills_used` 字段硬编码为 `[]`。这导致 `aqualib status` 的 Skills 统计永远为空，`on_session_start` hook 注入的历史记录也不显示技能使用情况。

**根因**：vendor tool 的调用由 SDK ReAct 循环驱动，调用结果已经通过 `on_post_tool_use` hook 写入了 `context_log.jsonl`（每条包含 `"event": "post_tool_use", "tool": "vendor_xxx"` 字段）。但 `update_session_after_task()` 没有利用这些已有的审计记录。

**修复方案（利用 hook 已写入的审计数据，不额外造收集器）**：

1. 给 `update_session_after_task()` 增加一个 `skills_used: list[str] | None = None` 参数
2. 在 `cli.py` 中调用前，从 `context_log.jsonl` 回溯提取本次 task 使用的 vendor tool 名称——这些数据由 SDK 的 `on_post_tool_use` hook 回调写入，是 SDK 生态的一部分

在 `cli.py` 的 `_run()` 函数中，在 `ws.update_session_after_task()` 之前：

```python
# Extract skills_used from hook audit trail (written by on_post_tool_use hook)
# This reads data already recorded by the SDK hook pipeline, not a parallel collector
recent_entries = ws.load_context_log()
task_skills: list[str] = []
for entry in reversed(recent_entries):
    # Stop at the user_prompt entry that marks the start of this task
    if entry.get("event") == "user_prompt" and entry.get("query") == query:
        break
    if entry.get("event") == "post_tool_use":
        tool_name = entry.get("tool", "")
        if tool_name.startswith("vendor_") and tool_name not in task_skills:
            task_skills.append(tool_name)
task_skills.reverse()  # chronological order

ws.update_session_after_task(actual_slug, query, result_messages, skills_used=task_skills)
```

同时更新 `manager.py` 中 `update_session_after_task()` 的签名和 context_log 写入：

```python
def update_session_after_task(
    self, slug: str, query: str, messages: list, skills_used: list[str] | None = None
) -> None:
    # ... existing code ...
    # In the context_log entry:
    self.append_context_log({
        "session_slug": slug,
        "task_id": uuid.uuid4().hex[:8],
        "query": query,
        "status": "completed",
        "skills_used": skills_used or [],  # was hardcoded []
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
```

同时更新 Bug 1 中的 executor memory 写入，把 `task_skills` 传入：
```python
ws.append_agent_memory_entry(actual_slug, "executor", {
    "query": query,
    "skills_used": task_skills,
    "output_preview": (result_messages[-1][:200] if result_messages else ""),
})
```

---

### Bug 6：`cli.py` 的 `run` 命令在项目不存在时只打印警告但不退出

**文件**：`src/aqualib/cli.py` L82-86

**问题**：
```python
project = ws.load_project()
if project is None:
    rprint("[yellow]⚠️ No project found. Run 'aqualib init' first...[/yellow]")
    # 没有 return 或 raise typer.Exit(1)！代码继续执行
```

后续 `ws.create_session()` 会调用 `ws.update_project()`，而 `update_project()` 在 project 为 None 时返回 None 不写入，但 session 目录仍会被创建。更糟的是，`update_session_after_task()` 尝试更新 project.json 时会静默跳过，导致 task_count 不记录。最终 SDK 的 `create_session()` 会成功创建会话，但 AquaLib 工作空间元数据处于不一致状态。

**修复方案**：在警告后添加退出，确保 SDK session 不会在无效的工作空间状态下创建：
```python
if project is None:
    rprint("[yellow]⚠️ No project found. Run 'aqualib init' first to set up your workspace.[/yellow]")
    raise typer.Exit(1)
```

---

### 额外要求

- 所有修改必须通过现有测试（`python -m pytest tests/ -v`）
- 新增的测试用例要覆盖上述每个修复点
- 不要修改 `pyproject.toml` 的依赖项
- **不要修改 Copilot SDK 的调用接口**（`create_session`、`resume_session`、`@define_tool` 签名、hook 回调签名等）
- **不要在 AquaLib 侧重复实现 SDK 已提供的能力**（ReAct 循环、上下文 compact、session 持久化等）
- 保持向后兼容：旧版工作空间（没有 `sessions/` 目录的）不应报错
- 所有对 SDK 事件/数据的消费都应通过 SDK 公开接口（hooks、事件回调），不要直接读取 SDK 内部的磁盘状态（`~/.copilot/session-state/`）
