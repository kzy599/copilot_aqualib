# AquaLib

Multi-agent scientific research framework built on the GitHub Copilot SDK.
Executor + Reviewer pipeline with vendor skill priority, workspace-isolated audit trails, and interactive chat mode.

## Features

- **Copilot SDK integration** — agent orchestration, tool dispatch, and infinite session management
- **Two-agent pipeline** — Executor (skill dispatch & execution) + Reviewer (quality assurance & audit)
- **Interactive chat mode** — multi-turn REPL with session persistence and slash commands
- **Vendor skill priority** — vendor skills discovered via `SKILL.md` are automatically preferred
- **Multi-session workspaces** — named sessions with independent agent memory and vendor traces
- **Project-aware state** — one directory = one project, with full history recall across sessions
- **Extensible skill registry** — mount external skill libraries at runtime via `SKILL.md` contracts
- **RAG retrieval** — optional LlamaIndex-based document and skill description indexing

## Installation

```bash
git clone --recursive https://github.com/kzy599/copilot_aqualib.git
cd copilot_aqualib
pip install -e ".[dev]"
```

`--recursive` pulls vendor submodules (e.g. `vendor/ClawBio`). Only run these commands once — the `aqualib` CLI is then available globally in your Python environment.

## Authentication & API Configuration

### Mode A: GitHub Subscription (default)

```bash
gh auth login
```

Keep `copilot.auth: "github"` in `aqualib.yaml` (the default). No API key needed.

### Mode B: GitHub Token

```bash
export GH_TOKEN="ghp-your-token"
```

Or set in `aqualib.yaml`:

```yaml
copilot:
  auth: "token"
  github_token: "ghp-your-token"
```

### Mode C: BYOK (Bring Your Own Key)

Use any OpenAI-compatible API — OpenRouter, OpenAI direct, local Ollama, etc.

**OpenRouter:**

```yaml
copilot:
  auth: "byok"
  model: "anthropic/claude-sonnet-4"
  provider:
    type: "openai"
    base_url: "https://openrouter.ai/api/v1"
    api_key: "sk-or-v1-xxx"
```

**OpenAI direct:**

```yaml
copilot:
  auth: "byok"
  model: "gpt-4o"
  provider:
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key: "sk-xxx"
```

**Local Ollama:**

```yaml
copilot:
  auth: "byok"
  model: "llama3"
  provider:
    type: "openai"
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
```

API key can also be set via environment variable:

```bash
export AQUALIB_PROVIDER_API_KEY="sk-or-v1-xxx"
```

## Quick Start

```bash
# 1. Initialise a project
mkdir ~/my_gwas_study && cd ~/my_gwas_study
aqualib init --name "罗非鱼GWAS" --description "Growth trait GWAS analysis"

# 2. Add data
cp ~/data/snp_data.vcf aqualib_workspace/data/

# 3. Interactive chat (recommended)
aqualib chat

# 4. Or single-shot execution
aqualib run "Analyze the SNP data for growth-related QTLs"

# 5. Check project status
aqualib status
```

## Chat Mode

`aqualib chat` starts an interactive REPL session with the AquaLib agent. The Copilot SDK session stays alive across turns — no process restart between questions.

### Basic usage

```bash
aqualib chat
```

### Session management

```bash
# Resume a specific session by slug or prefix
aqualib chat --session protein-alignment

# Force a new session
aqualib chat --new-session --session-name "gwas-round-2"
```

### Slash commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/status` | Display current project status |
| `/skills` | List all vendor skills |
| `/session` | Show current session info (slug, task count, timestamps) |
| `/history` | Show last 5 conversation entries in this session |
| `exit` / `quit` | Exit the chat |

### Example session

```text
🐙 AquaLib Chat
📂 Project: 罗非鱼GWAS
🔗 Session: gwas-round-1-a1b2c3d4 (0 tasks)

Type your message, or use /help for commands.
Type 'exit' to quit.
─────────────────────────────────────────

🧑 > What SNP markers are in the VCF file?
  ▶ executor started
[green]Found 12,847 SNP markers across 24 linkage groups...[/green]
  ✅ executor completed

🧑 > Which markers are associated with growth traits?
  ▶ executor started
  ▶ reviewer started
  ✅ reviewer completed
  ✅ executor completed

🧑 > /skills
(table of vendor skills)

🧑 > exit
👋 Chat ended. Session state saved.
```

### Exit behaviour

> **No background processes.** When you type `exit` or press `Ctrl+D`, the CopilotClient subprocess and all child processes terminate immediately. There is no daemon, no server, nothing left running. Your workspace state is fully persisted to disk.

`Ctrl+C` also triggers a graceful shutdown.

## Session Management

Each session maintains independent agent memory:

- **Executor memory** (`executor.json`) — tasks, skills used, output previews
- **Reviewer memory** (`reviewer.json`) — verdicts, violations, suggestions

When resuming a session, these memories are injected into agent system messages for continuity.

```bash
# List all sessions
aqualib sessions

# Resume in chat mode
aqualib chat --session <slug-prefix>

# Resume in single-shot mode
aqualib run "Follow-up query" --session <slug-prefix>
```

Session slug format: `{name}-{8-char-uuid}` (e.g. `alignment-a1b2c3d4`).

| Scope | Tracks | Stored in |
|-------|--------|-----------|
| **Project** | All tasks, global summary | `project.json`, `context_log.jsonl` |
| **Session** | Tasks within one line of work, agent memory | `sessions/<slug>/` |

## Running Multiple Projects in Parallel

AquaLib is a foreground process — exit the chat and the process ends. To run multiple projects simultaneously or keep sessions alive after disconnecting from SSH, use **tmux**.

### tmux workflow

```bash
# Install tmux
sudo apt install tmux   # Ubuntu/Debian
brew install tmux        # macOS

# Project 1
tmux new -s project_a
cd ~/project_a && aqualib chat
# Ctrl+B D to detach (process continues running)

# Project 2
tmux new -s project_b
cd ~/project_b && aqualib chat
# Ctrl+B D to detach

# Reconnect anytime
tmux attach -t project_a
tmux attach -t project_b

# List all tmux sessions
tmux ls
```

### tmux quick reference

| Keys | Action |
|------|--------|
| `Ctrl+B D` | Detach from session (process keeps running) |
| `Ctrl+B C` | Create new window |
| `Ctrl+B N` | Next window |
| `Ctrl+B P` | Previous window |
| `Ctrl+B [` | Scroll mode (q to exit) |
| `Ctrl+B &` | Kill current window |

> **tmux keeps your session alive.** SSH disconnects, terminal closes, even server reboots (if tmux is restarted) won't kill your AquaLib chat process. Simply `tmux attach` to reconnect.

## CLI Reference

All commands accept `--base-dir` (`-d`) and `--verbose` (`-v`).

| Command | Description |
|---------|-------------|
| `aqualib init` | Initialise workspace and `project.json` |
| `aqualib chat` | Interactive multi-turn chat mode |
| `aqualib run "<query>"` | Single-shot task execution |
| `aqualib status` | Project overview (tasks, data, sessions) |
| `aqualib sessions` | List all sessions |
| `aqualib skills` | List registered vendor skills |
| `aqualib tasks` | List completed tasks |
| `aqualib report <id>` | Display audit report for a task |

### Key flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--session` / `-s` | `chat`, `run` | Resume a session by slug or prefix |
| `--new-session` | `chat`, `run` | Force create a new session |
| `--session-name` | `chat`, `run` | Name for the new session |
| `--limit` / `-l` | `status` | Number of recent tasks to show |
| `--format` / `-f` | `report` | Output format: `markdown` or `json` |

## Architecture

```text
┌─────────────────────────────────────────────┐
│              Copilot SDK (Parent Agent)      │
│  Receives user input → formulates plan      │
│  → calls write_plan → delegates to agents   │
├──────────────┬──────────────────────────────┤
│              │                              │
│   ┌──────────▼──────────┐  ┌───────────────▼──────────┐
│   │     Executor        │  │       Reviewer            │
│   │  (custom sub-agent) │  │   (custom sub-agent)      │
│   │                     │  │                            │
│   │  Reads plan.md      │  │  Reads plan.md, verifies   │
│   │  Tools:             │  │  executor followed plan,   │
│   │  • vendor_<name>    │  │  checks VENDOR_PRIORITY,   │
│   │  • workspace_search │  │  writes verdict + audit.   │
│   │  • read_skill_doc   │  │                            │
│   │  • rag_search       │  │                            │
│   └─────────────────────┘  └────────────────────────────┘
│                                                         │
│  SDK built-in tools: file editing, terminal, search     │
│  AquaLib tools: write_plan, workspace_search, etc.      │
└─────────────────────────────────────────────────────────┘
```

The Parent Agent follows a **Plan-First workflow**: when a task involves tool execution, it first presents an execution plan (Goal, Data, Steps, Expected Output), then calls `write_plan` to persist the plan to `sessions/<slug>/plan.md` before delegating to sub-agents. The Executor and Reviewer both read `plan.md` at the start of their work. For pure knowledge questions, the plan is skipped entirely.

AquaLib registers custom tools (`vendor_*`, `workspace_search`, `read_skill_doc`, `write_plan`, `rag_search`) and hooks (`on_pre_tool_use`, `on_post_tool_use`, `on_error`) alongside the SDK's built-in capabilities.

## Vendor Skill Ecosystem

### Three-tier priority

| Priority | Source | Description |
|----------|--------|-------------|
| **1 (highest)** | `aqualib_workspace/skills/vendor/` | Per-project custom skills |
| **2** | `vendor/*/` | Repo-shipped submodule libraries |
| **3 (lowest)** | Built-in placeholders | Framework examples |

Higher-priority skills are never overwritten by lower-priority registrations.

### SKILL.md format

```markdown
---
name: my_custom_skill
description: Performs custom analysis on input data
version: 1.0.0
tags: analysis, custom
parameters: {"input_file": "string", "threshold": "float"}
---

# My Custom Skill

Detailed documentation here.
```

### Execution flow

When the agent invokes a vendor skill:

1. Create isolated invocation directory (`work/inv_NNNN/`)
2. Write parameters to `input.json`
3. Run vendor CLI: `python <entry_point> run <input.json> --output <output.json> --skill <name>`
4. Apply **43,200-second timeout** (12 hours) — process killed on timeout
5. Capture stdout/stderr, save trace to `vendor_traces/`
6. Return `output.json` content to the agent

Vendor skills with errors are retried up to **2 times** before being skipped.

## Configuration

`aqualib.yaml` (generated by `aqualib init`):

```yaml
copilot:
  auth: github               # "github" | "token" | "byok"
  # github_token: ""         # defaults to GH_TOKEN / GITHUB_TOKEN
  # provider:                # only for "byok"
  #   type: openai           # "openai" | "azure" | "anthropic"
  #   base_url: ""
  #   api_key: ""            # or AQUALIB_PROVIDER_API_KEY env var
  model: gpt-4o

rag:
  enabled: false
  api_key: ""                # AQUALIB_RAG_API_KEY → falls back to llm.api_key
  base_url: null             # AQUALIB_RAG_BASE_URL → falls back to llm.base_url
  chunk_size: 512
  chunk_overlap: 64
  similarity_top_k: 5
  embed_model: text-embedding-3-small

vendor_priority: true

directories:
  base: ./aqualib_workspace
```

### Credential resolution order

| Credential | 1st (highest) | 2nd | 3rd (lowest) |
|---|---|---|---|
| `copilot.github_token` | `aqualib.yaml` | `GH_TOKEN` / `GITHUB_TOKEN` | _(empty)_ |
| `copilot.provider.api_key` | `aqualib.yaml` | `AQUALIB_PROVIDER_API_KEY` | _(empty)_ |
| `rag.api_key` | `aqualib.yaml` | `AQUALIB_RAG_API_KEY` | falls back to `llm.api_key` |
| `rag.base_url` | `aqualib.yaml` | `AQUALIB_RAG_BASE_URL` | falls back to `llm.base_url` |

## Development

```bash
python -m pytest tests/ -v
ruff check .
```
