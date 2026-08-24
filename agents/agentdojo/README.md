# AgentDojo Harness

This harness runs the local project agent against an installed `agentdojo` suite.

Default run:

```bash
export OPENAI_API_KEY="your-api-key"  # Only needed when the model endpoint requires auth.
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/agentdojo/agent.yaml
```

API keys are never declared in `agent.yaml`. The main model reads the variable
named by `llm.api_key_env` (by default `OPENAI_API_KEY`). Optional guards use
`TOOLSAFE_API_KEY`, `AGENTDOG_API_KEY`, and `AGENTGUARD_API_KEY` when
enabled. These variables are forwarded into the benchmark container and are
not written to the batch JSON.

Every run records `harness.agentdojo.asr` in its trace (0.0 or 1.0 for an
attack case, null for a benign case). For repeated runs, the batch
`summary.json` is updated after every completed run. Its top-level `asr` is the
aggregate attack success rate, while `agentdojo` contains `evaluated_runs`,
`attack_trials`, and `attack_successes`. The same JSON includes `config_path`
and a credential-redacted `config` snapshot containing only `agent.yaml` and
`environment.yaml` for the batch. Other harness files and runtime-expanded
configuration are excluded. ASR is not printed to the terminal. Benign cases
remain visible in `evaluated_runs` but are excluded from the ASR denominator.

Edit `environment.yaml` to choose `benchmark_version`, `suite`, and `case`.
When the selected `case` includes an injection task, the adapter places that injection goal into every AgentDojo injection vector for the selected suite by default. Set `attack_template` to wrap the goal with an AgentDojo attack template without changing those locations. Set `custom_injection_text` to replace only the goal inside the template. `injection_vectors` is only an optional location filter. Use `custom_injections` only when you want fully custom per-vector payloads; it takes precedence over the other injection settings.
