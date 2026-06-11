# Attack Inject

Currently support prompt and tool I/O insertion and replacement, example is in `travel.yaml`.

# LangGraph Agent Scaffolding

A configurable LangGraph agent scaffold. You can modify `*.yaml` to define different agent behavior, models, tools, and graph settings, or add YAML files to implement other agents. In summary, unified agent scaffolding with different agent settings from `*.yaml`. 

## Quick Start
```bash
pip install -r requirements.txt
python src/agent_scaffold/main.py --config travel.yaml
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
- `tools.yaml` -> `tools`
- `skills.yaml` -> `skills`
- `planner.yaml` -> `planner`
- `middleware.yaml` -> `middleware`
- `memory.md` -> an inline `agent_memory` skill

Example:

```bash
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/paper_summary/agent.yaml
```

The older single-file YAML configs remain supported for compatibility.
