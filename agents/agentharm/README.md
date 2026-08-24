# AgentHarm harness

This harness runs one AgentHarm behavior through the project's agent graph while reusing the official Inspect Evals 0.17.0 dataset loader, tool implementations, grading function, semantic judge, and refusal judge.

```bash
export OPENAI_API_KEY="your-api-key"  # Omit for a local endpoint without auth.
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/agentharm/agent.yaml
```

Select `harmful`, `benign`, or `chat`, a split, and an optional `behavior_id` in `environment.yaml`. The first run downloads the pinned AgentHarm dataset revision through Inspect Evals. Harmful results expose `harmfulness_score`, `security`, and `attack_success`; benign results expose the official full utility score.

The two judge model names use Inspect model syntax. Configure their provider credentials in the environment. For an OpenAI-compatible local judge, also configure Inspect's corresponding model/provider environment variables.

AgentHarm contains harmful and potentially offensive evaluation content. Run it only in an isolated benchmark environment and do not use the examples for training.
