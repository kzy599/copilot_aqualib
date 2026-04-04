现在我有了完整的 Copilot SDK 文档和 AquaLib 现有代码。下面是详细的重构 prompt。

---

# AquaLib v2 重构 Prompt：基于 GitHub Copilot SDK 的完整重写

## 📋 Problem Statement

将 `kzy599/copilot_aqualib` 从当前"手搓 raw OpenAI API"架构，**完整迁移到基于 GitHub Copilot Python SDK (`github-copilot-sdk`) 的架构**。保留 AquaLib 原有的工作空间隔离、vendor skill 优先级、SKILL.md 驱动的外载技能框架和审计日志系统，但底层引擎全部换成 Copilot SDK 提供的 session/tools/hooks/custom-agents/skills 机制。

---

## 🏗️ 架构总览

```
┌──────────────────────────────────────────────────────────┐
│                     aqualib CLI (Typer)                   │
│  aqualib init / run / status / tasks / report / skills   │
├──────────────────────────────────────────────────────────┤
│                  AquaLib Core Layer                       │
│  ┌────────────┐  ┌────────────┐  ┌───────────────────┐  │
│  │  Config     │  │  Workspace │  │  Vendor Skill     │  │
│  │  (YAML +   │  │  Manager   │  │  Scanner          │  │
│  │   BYOK)    │  │            │  │  (SKILL.md →      │  │
│  │            │  │            │  │   SDK Tools)       │  │
│  └────────────┘  └────────────┘  └───────────────────┘  │
├──────────────────────────────────────────────────────────┤
│              Copilot SDK Integration Layer                │
│  ┌──────────────────────────────────────────────────┐    │
│  │  CopilotClient                                    │    │
│  │  ├─ Session (with custom_agents + tools + hooks) │    │
│  │  │   ├─ executor agent  (custom_agent, infer)    │    │
│  │  │   ├─ reviewer agent  (custom_agent, infer)    │    │
│  │  │   └─ vendor tools    (from SKILL.md scan)     │    │
│  │  │                                                │    │
│  │  │  Hooks:                                        │    │
│  │  │   ├─ onSessionStart  → load project context   │    │
│  │  │   ├─ onPreToolUse    → vendor priority check  │    │
│  │  │   ├─ onPostToolUse   → audit trail            │    │
│  │  │   ├─ onErrorOccurred → retry / skip / abort   │    │
│  │  │   └─ onSessionEnd    → save workspace state   │    │
│  │  │                                                │    │
│  │  │  Skills (native SDK skill_directories):        │    │
│  │  │   └─ workspace/skills/vendor/ → SKILL.md      │    │
│  │  └────────────────────────────────────────────────┘    │
│  └──────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────┤
│  Copilot CLI (JSON-RPC server, managed by SDK)           │
│  ├─ 内置 ReAct 循环 (tool_calls → execute → observe)    │
│  ├─ 内置工具 (read_file, edit_file, grep, glob, bash)   │
│  ├─ Sub-agent orchestration                              │
│  └─ Infinite sessions + context compaction               │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 目标目录结构

```
copilot_aqualib/
├── pyproject.toml                    # 依赖改为 github-copilot-sdk
├── README.md
├── aqualib.yaml.example              # 配置模板
│
├── src/aqualib/
│   ├── __init__.py
│   ├── cli.py                        # Typer CLI (保留 init/run/status/tasks/report/skills)
│   ├── config.py                     # Settings (扩展支持 BYOK provider)
│   │
│   ├── sdk/                          # ★ 新增：Copilot SDK 集成层
│   │   ├── __init__.py
│   │   ├── client.py                 # CopilotClient 生命周期管理
│   │   ├── session_manager.py        # Session 创建/恢复/切换/清理
│   │   ├── agents.py                 # custom_agents 定义 (executor + reviewer)
│   │   ├── tools.py                  # vendor SKILL.md → @define_tool 适配器
│   │   ├── hooks.py                  # 6 个 hook 实现
│   │   └── system_prompt.py          # SystemMessage 定制
│   │
│   ├── workspace/                    # 保留：工作空间物理文件管理
│   │   ├── __init__.py
│   │   └── manager.py               # WorkspaceManager (改造)
│   │
│   ├── skills/                       # 保留：SKILL.md 扫描 + 注册表
│   │   ├── __init__.py
│   │   ├── scanner.py                # scan_vendor_directory (保留)
│   │   ├── skill_meta.py             # SkillMeta 数据模型 (保留)
│   │   └── tool_adapter.py           # ★ 新增：SkillMeta → SDK Tool 转换器
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging.py
│
├── vendor/                           # 保留：vendor submodule 挂载点
│   └── ClawBio/
│
└── tests/
```

---

## 📝 逐模块详细规格

### 1. `config.py` — 配置系统（扩展 BYOK）

保留现有的 `aqualib.yaml` + 环境变量解析，**新增** Copilot SDK 专属配置：

```yaml
# aqualib.yaml
copilot:
  # Copilot CLI 认证方式 (三选一)
  auth: "github"                      # "github" | "token" | "byok"
  github_token: ""                    # auth=token 时使用，也读 GH_TOKEN 环境变量
  
  # BYOK 模式：用户自己提供模型 API
  provider:
    type: "openai"                    # "openai" | "azure" | "anthropic"
    base_url: ""                      # e.g. "http://localhost:11434/v1" for Ollama
    api_key: ""                       # 也读 AQUALIB_PROVIDER_API_KEY 环境变量
    azure:
      api_version: "2024-10-21"
  
  model: "gpt-4o"                     # session 默认模型
  reasoning_effort: null              # "low" | "medium" | "high" | "xhigh" | null
  streaming: false                    # 是否启用流式
  
  cli_path: null                      # 自定义 Copilot CLI 路径，也读 COPILOT_CLI_PATH
  use_stdio: true                     # stdio 或 TCP 传输

# 保留原有配置
workspace:
  base: ./aqualib_workspace

vendor_priority: true

# RAG 独立配置 (用于 workspace data/ 索引)
rag:
  embed_api_key: ""
  embed_base_url: null
  embed_model: "text-embedding-3-small"
  chunk_size: 512
  similarity_top_k: 5
```

Pydantic 模型新增：

```python
class CopilotSettings(BaseModel):
    auth: Literal["github", "token", "byok"] = "github"
    github_token: str = ""
    provider: ProviderConfig | None = None
    model: str = "gpt-4o"
    reasoning_effort: str | None = None
    streaming: bool = False
    cli_path: str | None = None
    use_stdio: bool = True

class ProviderConfig(BaseModel):
    type: Literal["openai", "azure", "anthropic"] = "openai"
    base_url: str = ""
    api_key: str = ""
    azure: AzureConfig | None = None

class Settings(BaseModel):
    copilot: CopilotSettings = Field(default_factory=CopilotSettings)
    workspace: DirectorySettings = Field(default_factory=DirectorySettings)
    vendor_priority: bool = True
    rag: RAGSettings = Field(default_factory=RAGSettings)
```

环境变量解析优先级：

| 配置项 | yaml | 环境变量 | 默认 |
|--------|------|---------|------|
| `copilot.github_token` | yaml | `GH_TOKEN` / `GITHUB_TOKEN` | `""` |
| `copilot.provider.api_key` | yaml | `AQUALIB_PROVIDER_API_KEY` | `""` |
| `copilot.provider.base_url` | yaml | `AQUALIB_PROVIDER_BASE_URL` | `""` |
| `copilot.cli_path` | yaml | `COPILOT_CLI_PATH` | `None` |
| `rag.embed_api_key` | yaml | `AQUALIB_RAG_API_KEY` | falls back to provider.api_key |

---

### 2. `sdk/client.py` — CopilotClient 生命周期

```python
from copilot import CopilotClient, SubprocessConfig, ExternalServerConfig

class AquaLibClient:
    """管理 CopilotClient 的创建和生命周期。"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: CopilotClient | None = None
    
    async def start(self) -> CopilotClient:
        config = self._build_config()
        self._client = CopilotClient(config)
        await self._client.start()
        return self._client
    
    async def stop(self):
        if self._client:
            await self._client.stop()
    
    def _build_config(self) -> SubprocessConfig:
        s = self.settings.copilot
        return SubprocessConfig(
            cli_path=s.cli_path,
            use_stdio=s.use_stdio,
            log_level="debug" if self.settings.verbose else "info",
            github_token=s.github_token or None,
            use_logged_in_user=(s.auth == "github"),
        )
```

---

### 3. `sdk/session_manager.py` — Session 管理 + 项目级持久化

**核心设计**：每个 AquaLib project 对应一个 Copilot session_id，通过 `project.json` 中记录的 `session_id` 实现跨命令的 session resume。

```python
class SessionManager:
    """管理 session 的创建、恢复、切换。
    
    session_id 规则: "aqualib-{project_name_slug}-{created_timestamp}"
    存储在 workspace/project.json 的 "session_id" 字段中。
    """
    
    def __init__(self, client: CopilotClient, settings: Settings, workspace: WorkspaceManager):
        self.client = client
        self.settings = settings
        self.workspace = workspace
    
    async def get_or_create_session(self) -> CopilotSession:
        """如果 project.json 有 session_id 且 session 存在则 resume，否则 create。"""
        project = self.workspace.load_project()
        existing_id = project.get("session_id") if project else None
        
        if existing_id:
            try:
                return await self._resume_session(existing_id)
            except Exception:
                pass  # session 不存在了，重建
        
        return await self._create_session()
    
    async def _create_session(self) -> CopilotSession:
        session_id = self._generate_session_id()
        
        session = await self.client.create_session(
            session_id=session_id,
            model=self.settings.copilot.model,
            reasoning_effort=self.settings.copilot.reasoning_effort,
            streaming=self.settings.copilot.streaming,
            provider=self._build_provider(),        # BYOK 支持
            
            # ★ SDK 原生 skill_directories
            skill_directories=self._collect_skill_dirs(),
            
            # ★ 自定义 Agents (executor + reviewer)
            custom_agents=build_custom_agents(self.settings),
            
            # ★ 自定义 Tools (vendor SKILL.md → SDK tools)
            tools=self._build_vendor_tools(),
            
            # ★ System prompt 定制
            system_message=build_system_message(self.settings, self.workspace),
            
            # ★ Hooks
            hooks=build_hooks(self.settings, self.workspace),
            
            # ★ 错误处理
            on_error_occurred=on_error_handler,
            
            # ★ Permission 控制
            on_permission_request=build_permission_handler(self.settings),
            
            # ★ Infinite sessions (长任务上下文管理)
            infinite_sessions={
                "enabled": True,
                "background_compaction_threshold": 0.80,
                "buffer_exhaustion_threshold": 0.95,
            },
        )
        
        # 记录 session_id 到 project.json
        self.workspace.update_project({"session_id": session_id})
        return session
    
    async def _resume_session(self, session_id: str) -> CopilotSession:
        return await self.client.resume_session(
            session_id,
            on_permission_request=build_permission_handler(self.settings),
            provider=self._build_provider(),  # BYOK 必须重新提供
            tools=self._build_vendor_tools(),
            skill_directories=self._collect_skill_dirs(),
            custom_agents=build_custom_agents(self.settings),
        )
    
    def _collect_skill_dirs(self) -> list[str]:
        """收集所有 skill 目录，供 SDK 原生 skill_directories 使用。
        
        三层优先级:
        1. workspace/skills/vendor/ (per-project)
        2. repo vendor/ (global submodules)
        3. 内置 skills/
        """
        dirs = []
        ws_vendor = self.workspace.dirs.skills_vendor
        if ws_vendor.exists():
            dirs.append(str(ws_vendor))
        
        repo_vendor = Path(__file__).resolve().parent.parent.parent.parent / "vendor"
        if repo_vendor.is_dir():
            for lib_dir in sorted(repo_vendor.iterdir()):
                if lib_dir.is_dir() and (lib_dir / "SKILL.md").exists():
                    dirs.append(str(lib_dir))
        
        return dirs
    
    def _build_provider(self) -> dict | None:
        """构建 BYOK provider 配置。"""
        if self.settings.copilot.auth != "byok" or not self.settings.copilot.provider:
            return None
        p = self.settings.copilot.provider
        config = {"type": p.type, "base_url": p.base_url}
        if p.api_key:
            config["api_key"] = p.api_key
        if p.azure:
            config["azure"] = {"api_version": p.azure.api_version}
        return config
    
    def _build_vendor_tools(self) -> list:
        """扫描 SKILL.md 文件，转换为 SDK @define_tool 工具。"""
        from aqualib.skills.tool_adapter import build_tools_from_skills
        return build_tools_from_skills(self.settings, self.workspace)
```

---

### 4. `sdk/agents.py` — Custom Agents（Executor + Reviewer）

利用 Copilot SDK 的 `custom_agents` 实现**多 Agent 协作**。CLI 内部 ReAct 循环自动处理 sub-agent 委派。

```python
def build_custom_agents(settings: Settings) -> list[dict]:
    """定义 executor 和 reviewer 两个 custom agents。
    
    SDK 的 custom_agents 机制：
    - 运行时根据用户意图自动选择合适的 agent
    - 每个 agent 有独立的 prompt 和工具范围
    - sub-agent 执行完后结果自动集成到 parent session
    """
    agents = [
        {
            "name": "executor",
            "display_name": "Executor Agent",
            "description": (
                "Carries out the user's scientific research task by invoking vendor skills "
                "and built-in tools. Always prefers vendor skills when available. "
                "Handles sequence alignment, drug interaction analysis, and data processing."
            ),
            "tools": None,  # 所有工具都可用
            "prompt": _EXECUTOR_PROMPT.format(
                vendor_priority="ALWAYS" if settings.vendor_priority else "when appropriate"
            ),
            "infer": True,  # 运行时自动选择
        },
        {
            "name": "reviewer",
            "display_name": "Reviewer Agent", 
            "description": (
                "Audits the executor's work for correctness and vendor priority compliance. "
                "Called after task execution to validate results."
            ),
            "tools": ["grep", "glob", "view", "read_file"],  # 只读工具
            "prompt": _REVIEWER_PROMPT,
            "infer": False,  # 只由 parent agent 显式委派，不自动推断
        },
    ]
    return agents

_EXECUTOR_PROMPT = """\
You are the **Executor** agent of the AquaLib framework.

Rules:
1. {vendor_priority} prefer vendor skills (tools prefixed with `vendor_`) over \
   built-in tools when there is any possibility of using them.
2. Before invoking a skill, read its SKILL.md for parameter details.
3. If a vendor skill fails, analyze the error and retry with corrected parameters \
   before falling back to built-in tools.
4. Write all outputs to the workspace results directory.
5. After completing all tasks, explicitly delegate to the reviewer agent by saying: \
   "Delegating to reviewer for audit."
"""

_REVIEWER_PROMPT = """\
You are the **Reviewer** agent of the AquaLib framework.

Your responsibilities:
1. Verify the executor's outputs for correctness and completeness.
2. **Vendor Priority Enforcement**: Check if a vendor skill could have been used \
   instead of a built-in tool. If yes, flag it as a violation.
3. Check that all output files exist and contain valid data.
4. Return your verdict in this exact format:
   VERDICT: approved | needs_revision
   VENDOR_PRIORITY: satisfied | violated - [reason]
   SUGGESTIONS: [list]
"""
```

---

### 5. `sdk/tools.py` — Vendor SKILL.md → SDK Tools 适配

**关键**：将 AquaLib 的 SKILL.md 驱动的 vendor skill 转换为 Copilot SDK 的 `@define_tool` 工具。CLI 的内置 ReAct 循环会自主决定何时调用。

```python
from pydantic import BaseModel, Field
from copilot import define_tool

def build_tools_from_skills(settings: Settings, workspace: WorkspaceManager) -> list:
    """扫描所有 SKILL.md，为每个 vendor skill 创建一个 SDK tool。"""
    from aqualib.skills.scanner import scan_all_skill_dirs
    
    skill_metas = scan_all_skill_dirs(settings, workspace)
    tools = []
    
    for meta in skill_metas:
        tool = _create_vendor_tool(meta, workspace)
        tools.append(tool)
    
    # 额外的 workspace 工具
    tools.append(_create_workspace_search_tool(workspace))
    tools.append(_create_read_skill_doc_tool(workspace, skill_metas))
    
    return tools

def _create_vendor_tool(meta: SkillMeta, workspace: WorkspaceManager):
    """为单个 vendor skill 创建 SDK tool（带 subprocess 执行）。"""
    
    class VendorSkillParams(BaseModel):
        parameters: dict = Field(description="Parameters to pass to the vendor skill CLI")
    
    @define_tool(
        name=f"vendor_{meta.name}",
        description=f"[VENDOR] {meta.description}. Tags: {', '.join(meta.tags)}",
    )
    async def vendor_skill_tool(params: VendorSkillParams) -> str:
        import asyncio, json, tempfile
        
        entry = _resolve_entry_point(meta.skill_dir)
        output_dir = workspace.next_invocation_dir()
        input_file = output_dir / "input.json"
        output_file = output_dir / "output.json"
        input_file.write_text(json.dumps(params.parameters, indent=2))
        
        proc = await asyncio.create_subprocess_exec(
            "python", str(entry), "run", str(input_file),
            "--output", str(output_file), "--skill", meta.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(meta.vendor_root),
        )
        stdout, stderr = await proc.communicate()
        
        # 保存 trace
        workspace.save_vendor_trace(meta.name, {
            "returncode": proc.returncode,
            "stdout": stdout.decode()[:2000],
            "stderr": stderr.decode()[:2000],
        })
        
        if proc.returncode != 0:
            return f"ERROR: Vendor skill '{meta.name}' failed (exit {proc.returncode}): {stderr.decode()[:500]}"
        
        if output_file.exists():
            return output_file.read_text()[:4000]
        return stdout.decode()[:4000]
    
    return vendor_skill_tool

def _create_workspace_search_tool(workspace: WorkspaceManager):
    """渐进式披露工具：搜索 workspace data/ 中的文件。"""
    
    class SearchParams(BaseModel):
        query: str = Field(description="Search keywords to find in workspace data files")
        max_results: int = Field(default=5, description="Maximum results to return")
    
    @define_tool(
        name="workspace_search",
        description="Search through project data files (CSV, FASTA, JSON, etc.) in the workspace. "
                    "Use this to find relevant data before invoking a vendor skill.",
    )
    async def workspace_search(params: SearchParams) -> str:
        hits = workspace.scan_data_files(params.query, max_results=params.max_results)
        if not hits:
            return "No matching files found in workspace data/."
        return json.dumps(hits, indent=2)
    
    return workspace_search

def _create_read_skill_doc_tool(workspace: WorkspaceManager, skill_metas: list):
    """渐进式披露工具：深入读取某个 skill 的完整 SKILL.md 文档。
    
    这是渐进式披露的关键：
    Level 1: custom_agents 的 description 里已经有一行摘要
    Level 2: CLI 内置 ReAct 循环遇到需要时调用此工具，读取完整文档
    Level 3: 读取 skill 同目录下的 README.md / AGENTS.md
    """
    
    class ReadSkillParams(BaseModel):
        skill_name: str = Field(description="Name of the vendor skill to read documentation for")
        include_readme: bool = Field(default=False, description="Also read README.md if present")
    
    @define_tool(
        name="read_skill_doc",
        description="Read the full SKILL.md documentation for a vendor skill. "
                    "Use this to understand parameters, constraints, and usage examples "
                    "BEFORE invoking a vendor skill.",
        skip_permission=True,  # 只读，无需权限确认
    )
    async def read_skill_doc(params: ReadSkillParams) -> str:
        for meta in skill_metas:
            if meta.name == params.skill_name:
                result = f"# SKILL.md for {meta.name}\n\n"
                skill_md = meta.skill_dir / "SKILL.md"
                if skill_md.exists():
                    result += skill_md.read_text(encoding="utf-8")
                
                if params.include_readme:
                    readme = meta.skill_dir / "README.md"
                    if readme.exists():
                        result += f"\n\n# README.md\n\n{readme.read_text(encoding='utf-8')}"
                    agents_md = meta.skill_dir / "AGENTS.md"
                    if agents_md.exists():
                        result += f"\n\n# AGENTS.md\n\n{agents_md.read_text(encoding='utf-8')}"
                
                return result[:8000]
        
        return f"Skill '{params.skill_name}' not found."
    
    return read_skill_doc
```

---

### 6. `sdk/hooks.py` — 6 个 Hook 实现

```python
def build_hooks(settings: Settings, workspace: WorkspaceManager) -> dict:
    """构建完整的 hook 集合。"""
    return {
        "on_session_start": _make_session_start_hook(workspace),
        "on_user_prompt_submitted": _make_prompt_hook(workspace),
        "on_pre_tool_use": _make_pre_tool_hook(settings, workspace),
        "on_post_tool_use": _make_post_tool_hook(workspace),
        "on_session_end": _make_session_end_hook(workspace),
        "on_error_occurred": _make_error_hook(workspace),
    }

def _make_session_start_hook(workspace):
    async def on_session_start(input_data, invocation):
        """注入项目上下文到 session。"""
        project = workspace.load_project()
        context_parts = []
        
        if project:
            context_parts.append(f"Project: {project.get('name', 'unknown')}")
            if project.get("summary"):
                context_parts.append(f"History: {project['summary']}")
        
        # 最近 5 条任务记录 (物理文件 context_log.jsonl 的回忆)
        entries = workspace.load_context_log()
        if entries:
            context_parts.append("Recent tasks:")
            for e in entries[-5:]:
                icon = "✅" if e.get("status") == "approved" else "⚠️"
                context_parts.append(
                    f"  {icon} \"{e.get('query', '')}\" → {e.get('status')} "
                    f"(skills: {', '.join(e.get('skills_used', []))})"
                )
        
        # 可用 vendor skills 概览（渐进式 Level 1：名称 + 一行描述）
        from aqualib.skills.scanner import scan_all_skill_dirs
        skills = scan_all_skill_dirs(workspace.settings, workspace)
        if skills:
            context_parts.append(f"\nAvailable vendor skills ({len(skills)}):")
            for s in skills:
                context_parts.append(f"  - vendor_{s.name}: {s.description[:80]}")
            context_parts.append(
                "\nUse 'read_skill_doc' tool for full documentation before invoking."
            )
        
        return {
            "additionalContext": "\n".join(context_parts),
        } if context_parts else None
    
    return on_session_start

def _make_pre_tool_hook(settings, workspace):
    async def on_pre_tool_use(input_data, invocation):
        """Vendor 优先级检查 + 审计记录。
        
        如果用户请求的任务有对应的 vendor skill，但 agent 选择了内置工具，
        返回 additionalContext 提醒 agent 优先使用 vendor。
        """
        tool_name = input_data["toolName"]
        
        # 审计日志
        workspace.append_audit_entry({
            "event": "pre_tool_use",
            "tool": tool_name,
            "args_preview": str(input_data.get("toolArgs", {}))[:200],
        })
        
        # Vendor 优先级提醒
        if settings.vendor_priority and not tool_name.startswith("vendor_"):
            vendor_skills = [t for t in input_data.get("availableTools", []) 
                          if t.startswith("vendor_")]
            if vendor_skills:
                return {
                    "permissionDecision": "allow",
                    "additionalContext": (
                        f"⚠️ VENDOR PRIORITY REMINDER: You are about to use '{tool_name}' "
                        f"but vendor skills are available: {', '.join(vendor_skills)}. "
                        f"Prefer vendor skills when applicable."
                    ),
                }
        
        return {"permissionDecision": "allow"}
    
    return on_pre_tool_use

def _make_post_tool_hook(workspace):
    async def on_post_tool_use(input_data, invocation):
        """记录工具执行结果到审计日志。"""
        workspace.append_audit_entry({
            "event": "post_tool_use",
            "tool": input_data["toolName"],
            "success": not input_data.get("toolError"),
            "result_preview": str(input_data.get("toolResult", ""))[:300],
        })
        return None
    
    return on_post_tool_use

def _make_error_hook(workspace):
    async def on_error_occurred(input_data, invocation):
        """错误处理策略：vendor skill 失败时自动重试。"""
        error_context = input_data.get("errorContext", "")
        error_msg = input_data.get("error", "")
        
        workspace.append_audit_entry({
            "event": "error",
            "context": error_context,
            "error": error_msg[:500],
        })
        
        # Vendor skill 错误：自动重试一次
        if "vendor_" in error_context:
            return {"errorHandling": "retry"}
        
        # 其他错误：跳过当前工具，让 agent 换个方案
        return {"errorHandling": "skip"}
    
    return on_error_occurred

def _make_session_end_hook(workspace):
    async def on_session_end(input_data, invocation):
        """保存最终状态到物理文件。"""
        workspace.finalize_task()
        return None
    
    return on_session_end

def _make_prompt_hook(workspace):
    async def on_user_prompt_submitted(input_data, invocation):
        """记录用户 prompt 到 context_log。"""
        workspace.append_context_log({
            "query": input_data.get("prompt", ""),
            "timestamp": input_data.get("timestamp"),
        })
        return None  # 不修改 prompt
    
    return on_user_prompt_submitted
```

---

### 7. `sdk/system_prompt.py` — SystemMessage 定制

使用 SDK 的 `customize` mode 精确控制 system prompt 各 section：

```python
def build_system_message(settings: Settings, workspace: WorkspaceManager) -> dict:
    """构建 system message，使用 SDK 的 customize mode。"""
    return {
        "mode": "customize",
        "sections": {
            "identity": {
                "action": "replace",
                "content": (
                    "You are AquaLib, a multi-agent scientific research assistant. "
                    "You coordinate between an executor agent (task execution) and a "
                    "reviewer agent (quality audit). You have access to specialized "
                    "vendor skills for scientific workflows."
                ),
            },
            "guidelines": {
                "action": "append",
                "content": _AQUALIB_GUIDELINES.format(
                    vendor_priority="ALWAYS" if settings.vendor_priority else "when appropriate"
                ),
            },
        },
        # Additional instructions appended after all sections
        "content": _build_additional_context(workspace),
    }

_AQUALIB_GUIDELINES = """
## AquaLib Framework Rules

1. **Vendor Priority**: {vendor_priority} prefer vendor tools (prefixed `vendor_`) over 
   built-in tools when there is any possibility of using them.

2. **Progressive Disclosure**: 
   - First check available vendor skills via the tool list
   - Use `read_skill_doc` to read full SKILL.md before invoking a vendor skill
   - Use `workspace_search` to find relevant data in the project

3. **Executor → Reviewer Pipeline**:
   - After completing a task, delegate to the reviewer agent for audit
   - If the reviewer says "needs_revision", address the feedback and re-run

4. **Workspace Discipline**:
   - All outputs go to the workspace results directory
   - Never modify files in data/ (read-only source data)
   - Vendor skill invocations are traced in vendor_traces/
"""
```

---

### 8. `cli.py` — CLI 重构

```python
@app.command()
def run(query: str, ...):
    """Run a task using the Copilot SDK agent pipeline."""
    settings = _get_settings(base_dir, verbose)
    
    async def _run():
        from aqualib.sdk.client import AquaLibClient
        from aqualib.sdk.session_manager import SessionManager
        
        aqua_client = AquaLibClient(settings)
        client = await aqua_client.start()
        
        try:
            ws = WorkspaceManager(settings)
            sm = SessionManager(client, settings, ws)
            session = await sm.get_or_create_session()
            
            # 监听事件
            done = asyncio.Event()
            result_messages = []
            
            def on_event(event):
                if event.type.value == "assistant.message":
                    result_messages.append(event.data.content)
                elif event.type.value == "subagent.started":
                    rprint(f"  [dim]▶ {event.data.agent_display_name} started[/dim]")
                elif event.type.value == "subagent.completed":
                    rprint(f"  [dim]✅ {event.data.agent_display_name} completed[/dim]")
                elif event.type.value == "session.idle":
                    done.set()
            
            session.on(on_event)
            
            await session.send(query)
            await done.wait()
            
            # 更新 project 状态
            ws.update_project_after_task(query, result_messages)
            
            return result_messages
        finally:
            await aqua_client.stop()
    
    results = asyncio.run(_run())
    _display_results(results, settings)
```

---

### 9. `workspace/manager.py` — 保留 + 适配

保留现有的 WorkspaceManager，但移除 agent 内部的 task 状态管理（交给 SDK session），只保留：
- 目录结构创建 (`create_dirs`)
- `project.json` 管理 (增加 `session_id` 字段)
- `context_log.jsonl` 追加
- `vendor_traces/` 写入
- `data/` 文件扫描 (`scan_data_files`)
- 审计日志追加 (`append_audit_entry`)

---

### 10. `pyproject.toml` — 依赖更新

```toml
[project]
name = "aqualib"
version = "0.2.0"
requires-python = ">=3.11"

dependencies = [
    "github-copilot-sdk>=0.1.0",    # ★ 核心依赖
    "pydantic>=2.0",
    "typer>=0.9.0",
    "rich>=13.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
rag = [
    "llama-index-core>=0.11.0",
    "llama-index-embeddings-openai>=0.2.0",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.3.0",
]
```

注意：`openai` 和 `llama-index-llms-openai` 不再是核心依赖（LLM 调用全部通过 SDK），RAG embedding 作为可选依赖。

---

## 🔄 SDK 能力利用清单

| Copilot SDK 能力 | AquaLib 中如何利用 | 替代了旧版的什么 |
|---|---|---|
| `CopilotClient` + `SubprocessConfig` | `sdk/client.py` | 旧 `openai.AsyncOpenAI` 直接调用 |
| `create_session(provider=...)` BYOK | `config.py` + `session_manager.py` | 旧 `llm.base_url` + `llm.api_key` |
| `custom_agents` | executor + reviewer 两个 sub-agent | 旧 `ExecutorAgent` + `ReviewerAgent` 类 |
| Sub-agent auto-delegation | CLI 自动委派 executor/reviewer | 旧 `Orchestrator` 手动管道 |
| **CLI 内置 ReAct 循环** | tool_calls → execute → observe 自动循环 | 旧单程 `_plan()` → `_invoke_skill()` |
| `@define_tool` | vendor SKILL.md → SDK 自定义工具 | 旧 `VendorCliSkill.execute()` |
| `skill_directories` | 直接传入 skill 目录，SDK 自动加载 SKILL.md | 旧 `scan_vendor_directory()` |
| `hooks.on_pre_tool_use` | vendor 优先级检查 | 旧 `ReviewerAgent._review()` |
| `hooks.on_post_tool_use` | 审计日志 | 旧 `workspace.save_vendor_trace()` |
| `hooks.on_error_occurred` | `"retry"` / `"skip"` 策略 | 旧 `except: pass` 无重试 |
| `hooks.on_session_start` | 注入项目上下文 + skill 概览 | 旧 `Orchestrator._build_project_context()` |
| `session.resume_session` | 跨命令 session 恢复 | 旧无此能力（每次 `aqualib run` 独立） |
| `infinite_sessions` | 长对话自动上下文压缩 | 旧无此能力 |
| `streaming=True` | 流式输出 | 旧无此能力 |
| `SystemMessage(mode="customize")` | 精确控制 prompt sections | 旧硬编码 `_SYSTEM_PROMPT` 字符串 |
| Session lifecycle events | `subagent.started/completed` 实时反馈 | 旧无可观测性 |
| `session.workspace_path` | SDK 管理的 session 文件 | 旧自管 `aqualib_workspace/` |
| `on_permission_request` | 工具执行权限控制 | 旧无权限机制 |
| `on_user_input_request` | Agent 可以反问用户 | 旧无交互能力 |

---

## 🧪 测试要求

1. **单元测试**：mock `CopilotClient`，测试 `SessionManager` 创建/恢复逻辑
2. **单元测试**：测试 `build_tools_from_skills()` 能正确扫描 SKILL.md 生成 SDK tool
3. **单元测试**：测试每个 hook 的输入/输出格式
4. **单元测试**：测试 `config.py` 的 BYOK provider 解析 + 环境变量优先级
5. **集成测试**：`aqualib init` → `aqualib run` 完整流程（需要 CLI 可用）
6. **保留现有测试**：workspace/manager、skills/scanner、config 的测试应继续通过

---

## 📌 关键约束

1. **所有 LLM 调用必须通过 Copilot SDK**，不得直接使用 `openai.AsyncOpenAI`
2. **SKILL.md 驱动**的 vendor skill 标准保持不变——scanner 读 Markdown，tool_adapter 转 SDK tool
3. **物理文件系统**作为持久化层（`project.json`、`context_log.jsonl`、`vendor_traces/`）——这是 AquaLib 的核心设计
4. **一个目录 = 一个项目**的原则不变，`aqualib.yaml` 为 per-project 配置
5. **Session ID 存在 `project.json`**，实现跨 `aqualib run` 调用的 session 恢复
6. **Vendor 优先级**通过 `on_pre_tool_use` hook 实时强制执行，不再依赖 Reviewer 事后审查
7. **渐进式披露**通过 `read_skill_doc` 工具 + `on_session_start` context 注入实现

---

## 🗑️ 应该删除的旧文件

| 文件 | 原因 |
|------|------|
| `src/aqualib/core/agent_base.py` | 被 SDK session 替代 |
| `src/aqualib/core/executor.py` | 被 `sdk/agents.py` custom_agent 替代 |
| `src/aqualib/core/reviewer.py` | 被 `sdk/agents.py` custom_agent 替代 |
| `src/aqualib/core/searcher.py` | 被 SDK 内置 ReAct + `read_skill_doc` + `workspace_search` 工具替代 |
| `src/aqualib/core/orchestrator.py` | 被 SDK sub-agent delegation 替代 |
| `src/aqualib/core/message.py` | Task/Message 模型被 SDK session events 替代 |
| `src/aqualib/rag/` | RAG retriever 和 indexer 作为可选模块保留，但不再是核心路径 |
| `src/aqualib/skills/registry.py` | 被 SDK 原生 tool 注册替代 |
| `src/aqualib/skills/loader.py` | 被 `skills/tool_adapter.py` 替代 |
| `src/aqualib/bootstrap.py` | 被 `sdk/client.py` + `sdk/session_manager.py` 替代 |