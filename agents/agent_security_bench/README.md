# Agent Security Bench harness

This harness maps one official ASB task/attacker-tool pair into the project's normal agent graph. The adapter uses the official ASB task, normal-tool, attacker-tool, prompt-template, and tool-call scoring semantics from revision `1f561dccf92d55302368fa67679b4ba9d9c8fdc4`.

Run the configured case with:

```bash
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/agent_security_bench/agent.yaml
```

Choose the scenario with `agent_name`, `task_index`, and `attacker_tool` in `environment.yaml`. `utility` means that all official normal tools for the scenario were called; `attack_success` means that the selected attacker tool was called. The `memory_attack` and PoT modes reproduce their attack context in this project's single-agent prompt rather than starting ASB's AIOS scheduler or Chroma memory service.

For a host run without Docker, clone the official ASB repository and set `data_dir` or `ASB_DATA_DIR` to its `data/` directory.
