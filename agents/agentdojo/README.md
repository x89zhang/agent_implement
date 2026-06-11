# AgentDojo Harness

This harness runs the local project agent against an installed `agentdojo` suite.

Default run:

```bash
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/agentdojo/agent.yaml
```

Edit `environment.yaml` to choose `benchmark_version`, `suite`, `user_task`, and optional `injection_task`.
When `injection_task` is set and `injections` is empty, the adapter places that injection goal into every AgentDojo injection vector for the selected suite.
