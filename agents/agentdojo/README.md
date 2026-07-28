# AgentDojo Harness

This harness runs the local project agent against an installed `agentdojo` suite.

Default run:

```bash
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/agentdojo/agent.yaml
```

Edit `environment.yaml` to choose `benchmark_version`, `suite`, and `case`.
When the selected `case` includes an injection task, the adapter places that injection goal into every AgentDojo injection vector for the selected suite by default. Set `attack_template` to wrap the goal with an AgentDojo attack template without changing those locations. Set `custom_injection_text` to replace only the goal inside the template. `injection_vectors` is only an optional location filter. Use `custom_injections` only when you want fully custom per-vector payloads; it takes precedence over the other injection settings.
