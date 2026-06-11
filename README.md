# Attack Inject

Currently supports prompt and tool I/O insertion and replacement through file-based agent harness configs.

# LangGraph Agent Scaffolding

A configurable LangGraph agent scaffold. Agent behavior is organized as file-based harness directories under `agents/`, with separate prompt, tools, skills, planner, middleware, and memory files. 

## Quick Start
```bash
pip install -r requirements.txt
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/travel/agent.yaml
```
Support both online models, i.e., ChatGPT, Gemini... and local models (via vLLM)

## Config Overview
- `llm`: provider and model parameters (supports OpenAI/Anthropic and vLLM via OpenAI-compatible endpoint)
- `agent`: system prompt
- `tools`: tool list (imported via `module:attr`)
- `graph`: graph structure and loop control (`single_agent` or `langchain_react`)

Note: For LangChain v1.x, legacy agent classes are provided by `langchain-classic`.

## Harness Components

The scaffold now supports lightweight harness components while keeping existing YAML files compatible:

- `skills`: reusable task methods loaded from `skills/<name>/skill.yaml` and `SKILL.md`, then appended to the agent instructions.
- `planner`: optional static task checklist stored in state and trace, and injected during execution as current plan context.
- `middleware`: runtime hooks for model/tool boundaries. The built-in middleware tracks plan progress, records recent tool errors, and blocks unsafe or empty `write_text_file` calls before execution.

Example:
```yaml
skills:
  enabled:
    - paper_summary

planner:
  enabled: true
  steps:
    - Search papers using configured parameters.
    - Summarize returned papers only.
    - Save the final report.

middleware:
  enabled: true
```

Trace files include `harness` and `plan` sections so runs can be audited without changing task-specific tools.


### File-Based Agent Harness

Agents can also be organized as file-based harness directories, similar to AHE-style harness workspaces:

```text
agents/
  assistant/
  deep_research/
  email/
  paper_summary/
  tool_qa/
  travel/
  web/
  web_update_readme/

Each directory contains:
  agent.yaml
  environment.yaml
  systemprompt.md
  tools.yaml
  skills.yaml
  planner.yaml
  middleware.yaml
  memory.md
```

Use `agent.yaml` as the entrypoint. When `harness: .` is present, the loader merges sibling files into the runtime config:

- `systemprompt.md` -> `agent.system_prompt`
- `task.md` -> `agent.task` when present
- `environment.yaml` -> scenario defaults such as `trip`, `paper`, `email`, `web`, or `research`
- `tools.yaml` -> `tools`
- `skills.yaml` -> `skills`
- `planner.yaml` -> `planner`
- `middleware.yaml` -> `middleware`
- `memory.md` -> an inline `agent_memory` skill

Example:

```bash
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/paper_summary/agent.yaml
```

Single-file YAML configs can still be passed explicitly, but runtime tools no longer infer or depend on the old per-agent directories.

### Legacy Single-File Configs

The old single-file agent directories and historical run outputs have been moved to:

```text
legacy/old_agent_configs/
```

They are kept for reference and rollback only. New runs should use `agents/<name>/agent.yaml`.

### Skill Package Layout

Agent-level `skills.yaml` files are indexes that enable reusable skill packages:

```yaml
base_dir: ../../skills
enabled:
  - deep_research
```

Each reusable skill lives under `skills/<name>/`:

```text
skills/<name>/
  skill.yaml  # machine-readable metadata: name, description, requires_tools, priority
  SKILL.md    # human-readable skill instructions injected into the agent context
```

Keep concrete tool-use strategy in `SKILL.md`; keep task goals and acceptance criteria in `agent.yaml`.

### Job Outputs

Each run writes trace files and generated artifacts under a timestamped job directory:

```text
jobs/<YYYYMMDDTHHMMSSZ>_<agent-name>/
```

Relative `monitoring.output_path` values and `write_text_file` outputs are resolved inside that job directory. The trace records both the `job_dir` and the original `config_path`.

