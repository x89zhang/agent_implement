# LangGraph Agent Scaffolding

A configurable LangGraph agent scaffold. You can modify `config.yaml` to define different agent behavior, models, tools, and graph settings.

## Layout
- `config.yaml` example configuration
- `src/agent_scaffold/` core code

## Quick Start
```bash
pip install -r requirements.txt
python -m agent_scaffold.main --config config.yaml --input "Hi, what can you do?"
```

## Config Overview
- `llm`: provider and model parameters (supports OpenAI/Anthropic and vLLM via OpenAI-compatible endpoint)
- `agent`: system prompt
- `tools`: tool list (imported via `module:attr`)
- `graph`: graph structure and loop control (`single_agent` or `langchain_react`)

Note: For LangChain v1.x, legacy agent classes are provided by `langchain-classic`.

See inline comments in `config.yaml` for more details.
