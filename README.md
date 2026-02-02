# LangGraph Agent Scaffolding

A configurable LangGraph agent scaffold. You can modify `*.yaml` to define different agent behavior, models, tools, and graph settings, or add YAML files to implement other agents.

## Layout
- `config.yaml` example configuration
- `src/agent_scaffold/` core code

## Quick Start
```bash
pip install -r requirements.txt
python src/agent_scaffold/main.py --config travel.yaml
```

## Config Overview
- `llm`: provider and model parameters (supports OpenAI/Anthropic and vLLM via OpenAI-compatible endpoint)
- `agent`: system prompt
- `tools`: tool list (imported via `module:attr`)
- `graph`: graph structure and loop control (`single_agent` or `langchain_react`)

Note: For LangChain v1.x, legacy agent classes are provided by `langchain-classic`.

