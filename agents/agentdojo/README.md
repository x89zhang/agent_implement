# AgentDojo Harness

This harness runs the local project agent against an installed `agentdojo` suite.

Default run:

```bash
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/agentdojo/agent.yaml
```

Edit `environment.yaml` to choose `benchmark_version`, `suite`, and `case`.
When the selected `case` includes an injection task and no explicit per-vector injections are set, the adapter places that injection goal into every AgentDojo injection vector for the selected suite. Set `custom_injection_text` to keep those default positions while replacing only the injected text. Use `custom_injections` only when you want to override specific injection vectors manually.
