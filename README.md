# Attack Inject

Currently support prompt and tool I/O insertion and replacement, example is in `travel.yaml`.

# LangGraph Agent Scaffolding

A configurable LangGraph agent scaffold. You can modify `*.yaml` to define different agent behavior, models, tools, and graph settings, or add YAML files to implement other agents. In summary, unified agent scaffolding with different agent settings from `*.yaml`. 

## Layout
- `config.yaml` example configuration
- `src/agent_scaffold/` core code

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

