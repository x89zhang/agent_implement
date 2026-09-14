# Hermes backend

The project runs the upstream Hermes `AIAgent` in a separate process. Its reasoning loop, prompts, memory tool, and context management replace the project's builtin graph. A stateless MCP stdio proxy forwards tool calls to a runner-owned benchmark session over an authenticated loopback bridge. Restarting the proxy does not recreate the benchmark environment.

## Install

Use Python 3.11–3.13. Keep project dependencies and Hermes dependencies in separate environments. From the project root:

```bash
# Run with your existing project interpreter.
python -m pip install -r requirements-hermes-bridge.txt

# This creates a new environment inside the project; Hermes requires an editable install.
python3 -m venv runtime/venvs/hermes
runtime/venvs/hermes/bin/python -m pip install -e '/home/xiaoliang_zhang/hermes-agent[mcp]'
```

Installation uses the local checkout's package requirements and may generate ignored packaging metadata there. Upstream intentionally rejects ordinary wheel installation. The launcher imports code from `execution.hermes.repo_path`, so record its commit and keep it fixed. The supplied examples require commit `5172f22df275a8c0b4bc0545da01c7a377c1690f`. Update `expected_commit` deliberately when upgrading and rerun integration tests. A dirty checkout is rejected unless `allow_dirty_checkout: true` is explicit.

Install the benchmark's existing dependencies in the project environment: AgentDojo for AgentDojo runs, `requirements-agentharm.txt` for AgentHarm, and ASB data for ASB. These do not need to be installed in Hermes. Never use a Hermes source checkout containing `.env`; the upstream loader reads that file, defeating clean experiment configuration. Supply model credentials through `llm.api_key_env` instead.

## Run

Edit the standalone YAML examples in `agents/hermes/` to select the model, benchmark case, interpreter, and local data path. Then use the normal project runner:

```bash
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/hermes/agentdojo.yaml
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/hermes/agentharm.yaml
PYTHONPATH=src python src/agent_scaffold/main.py --config agents/hermes/asb.yaml
```

Export the environment variable named by `llm.api_key_env` before starting. The model provider must be `openai` (an OpenAI-compatible chat endpoint) or `openrouter`. Hermes owns transport behavior; project graph transport settings do not apply. Set `container.enabled: true` to build and run the Hermes image automatically (see below). The host-only examples keep it false for local debugging.

Select `execution.backend: builtin` or omit `execution` to retain existing behavior. Hermes dispatch happens before graph construction. Project system prompts and planner are not used. Enabled defense policies use the effective tool inventory, including native Hermes tools. In `inline` mode they are compiled before execution; in `replay` mode each policy is compiled in its own replay controller after execution. Configured skills are materialized in the isolated native skill library. Supported model settings are model, base URL, credential environment variable, and temperature. Limits are `execution.hermes.max_iterations` and `timeout_seconds` (per phase).

## ASB native-memory experiments

`native_two_stage` runs three isolated phases by default:

1. **poison**: submits `poisoning_input_file` as a user-message carrier. Hermes may write native memory.
2. **target**: starts a new Hermes process/session, copying only `MEMORY.md` and `USER.md` into its new home. The target receives the clean ASB task, with no synthetic retrieved-memory prompt.
3. **control**: repeats the same clean ASB task from the clean initial memory, without poisoned state.

The target and control both expose the same normal and attacker tools. Their environments and call journals are independent; poison-phase calls cannot satisfy target scoring. Session search, general filesystem tools, delegation, background review, and external memory providers are disabled. Native skills are enabled only when skills are configured; only memory files carry over between phases. Native memory is included in Hermes' session-start prompt rather than a vector search.

The supplied poisoning text is an illustrative controller-authored preparation task, not a canonical ASB attack-generation algorithm. Customize it for the chosen case. `clean_initial_memory_dir` optionally names a directory containing baseline `MEMORY.md`/`USER.md`. `direct_seed` instead requires `poisoned_memory_dir` and measures downstream susceptibility to controller-seeded memory. It does not test agent-mediated writing. Do not pool these two modes.

The lifecycle file reports byte-level memory changes, target attack success/utility, and clean-control scores. `semantic_poisoning_verified` is null: a changed file alone does not establish semantic attack adoption. This is an ASB-derived protocol using the existing ASB call-based evaluator, not an exact reproduction of the original ASB memory runtime. Set `memory_experiment.mode: off` for ordinary single-phase runs; existing ASB prompt injection remains in effect in that mode.

## Artifacts and failure behavior

Each run contains `result.json`, a benchmark trace, `hermes.metadata.json`, a config snapshot, and `memory.lifecycle.json`. Each phase contains its isolated home/workspace, effective tools, raw Hermes result/events, worker logs, memory manifests, canonical benchmark call journal, and evaluation. The worker result is written only after cleanup; memory transfer happens after process exit. The supervisor tracks descendant process identities because MCP proxies can start in separate process groups, and terminates tracked proxies on timeout. Effective model-facing tool names and their canonical benchmark mapping are recorded separately.

Missing dependencies, unexpected tool inventories, unsupported settings, timeouts, and failed Hermes runs raise errors and are counted by the existing batch runner as failed runs. Phase `failure.json` and worker logs explain runtime failures. No failed run is silently scored as a successful safe outcome. A normal model refusal remains a completed answer and is handled by the benchmark's scorer.

## Project defenses

Use the existing top-level `aegis`, `pro2guard`, `agentspec`, `toolsafe`, `agentdog`, `agentguard`, and `llamafirewall` settings. No separate Hermes policy format is needed. Their optional dependencies/model endpoints still need to be available in the **project interpreter**. The external Hermes interpreter does not import the project's guard packages.

```yaml
aegis:
  enabled: true
  mode: block
  block_tools: [memory]  # Example: prevent native memory writes during all phases.
```

Benchmark tool policies use their canonical names, such as `send_direct_message`, without the `mcp__benchmark__` prefix. Native tools use names such as `memory`, `skill_view`, and `skill_manage`. All enabled policies apply to poison, target, and control phases. Blocking every memory write is a baseline, not a semantic memory-poisoning detector.

Hermes' public `llm_execution` and `tool_execution` middleware registration APIs forward checks to the project service. The adapter covers model input/output, each proposed action, and returned tool content. It supports blocking, argument/tool rewrites within the granted inventory, result isolation, monitoring, and bounded final-answer revision through Hermes' own redirect/retry mechanism. Stateful tool checks and execution are serialized even when Hermes proposes parallel calls. Transport/adapter failures stop execution instead of falling through Hermes' default plugin-error behavior.

`execution.hermes.defense_mode` selects the execution protocol. The default, `inline`, preserves enforcement behavior: guard decisions can block, rewrite, or retry the live run. Set it to `replay` for measurement-only experiments. Hermes then runs with a recorder that forwards the exact request, response, arguments, and result objects unchanged. It makes no guard RPCs and does not serialize parallel tools. After benchmark evaluation, the service replays the immutable `guard_lifecycle.jsonl` once per enabled defense. Every replay gets a deep-copied configuration, a fresh controller/state, and its own output directory, so one defense's decisions and trace cannot enter another defense's input. Returned block/rewrite decisions are recorded but never applied to Hermes.

In `inline` mode, `defense_events.jsonl` and `defenses.json` remain at the phase root. In `replay` mode, root `defenses.json` is a manifest and per-method results are written under `defense_replay/<method>/defense_events.jsonl` and `defense_replay/<method>/defenses.json`. A detector failure is attached only to that method in the manifest; the captured Hermes result and other detector replays remain available. `agentsight.enabled` uses the existing process observer, or the host-managed container observer. Actual eBPF collection still depends on the existing AgentSight installation and privileges; attaching this lifecycle is not a claim that every platform supports observation.

## AgentDojo native skill injection

```yaml
agentdojo:
  enabled: true
  suite: slack
  case: user_task_15_injection_5
  injection_enabled: true
  standard_injection_enabled: false
  skill_injection_enabled: true
```

The selected attack template/custom text becomes `home/skills/agentdojo-skill-payload/SKILL.md`. Hermes receives a skill-name hint and its native skill tools; the payload body is delivered by `skill_view`, not appended directly to the system prompt. Enabling both injection flags combines tool-output and skill attacks. Existing `skills.enabled` text instructions are also staged as native skills; executable supporting scripts are not copied.

`skills.json` records source/hash, `events.jsonl` records skill reads, and `skills.exposure.json` distinguishes attempted reads from delivery of the original content. A model may choose not to read the skill. A blocked or rewritten result is not recorded as original-payload exposure. Skill files and histories do not transfer from the ASB poison phase to its target.

## Automatic Docker execution

```yaml
container:
  enabled: true
  auto_build: true
  network: host
```

```bash
PYTHONPATH=src python src/agent_scaffold/main.py \
  --config agents/hermes/agentdojo-guarded-container.yaml
```

The runner stages a shallow copy of the clean local Hermes checkout, builds the project's benchmark image, and layers an editable Hermes installation in `/opt/hermes-venv`. The image tag includes the source commit, dependency-file contents, Dockerfiles and build arguments. It reuses that image on subsequent runs. The host Hermes virtual environment is not mounted or required for container execution. Automatic image builds require a clean checkout, even when host-only runs allow a dirty source tree.

AgentDojo/AgentHarm and supported optional guard dependencies are selected automatically from enabled settings. External dataset, memory-input and policy paths are mounted read-only and rewritten in `hermes.container.yaml`; project files and run artifacts use the existing workspace mount. Credentials pass by environment-variable name. The container runs as the host UID/GID, keeps the two Python environments separate, and reuses existing logs, AgentSight start gates and cleanup. Model access must be reachable under the selected Docker network (`host` is convenient for local model servers on Linux). This is experiment-level OS isolation; the benchmark service and worker share a container, not separate security domains.

## Current boundaries

- All three benchmark adapters are supported; live evaluation still needs model access and benchmark data.
- Existing defense configuration/dependency requirements still apply; adapting a policy does not establish its effectiveness against these attacks.
- Project conversation replay is rejected; each memory phase starts a fresh Hermes session.
- This is a pinned-source integration, not a stable public Hermes SDK compatibility promise.

## Test

```bash
PYTHONPATH=src python -m pytest tests/test_hermes_backend.py

# Use the project test interpreter plus a separate Hermes interpreter.
# Uses a local scripted model endpoint; no paid model API calls.
HERMES_TEST_REPO=/home/xiaoliang_zhang/hermes-agent \
  HERMES_TEST_PYTHON="$PWD/runtime/venvs/hermes/bin/python" \
  PYTHONPATH=src python -m pytest tests/test_hermes_backend.py

# Also build/run the real Docker integration test:
HERMES_TEST_DOCKER=1 HERMES_TEST_REPO=/home/xiaoliang_zhang/hermes-agent \
  HERMES_TEST_PYTHON="$PWD/runtime/venvs/hermes/bin/python" \
  PYTHONPATH=src python -m pytest tests/test_hermes_backend.py
```

The optional real-source test drives upstream Hermes through MCP, performs a native memory write, verifies exposure in a fresh session, and checks the target/control difference. It is a transport and lifecycle test, not evidence of real-model attack effectiveness.

Validated locally with Hermes 0.21.1 at the pinned commit, including its installed MCP 2.0.0 client and both 1.x/2.x proxy server APIs. Tests use scripted local model responses; no live-model attack effectiveness is asserted.


### Model transport selection

`execution.hermes.api_mode` defaults to `auto`, preserving Hermes' native routing.
Use `chat_completions` for chat-only model servers or `codex_responses` for the
Responses API (this Hermes name also covers ordinary OpenAI API-key access).
The defense bridge handles both wire formats. Unmodified Responses requests and
outputs retain their native replay IDs and opaque reasoning; rewritten input is
rebuilt from the guarded history. Each phase records its selected transport in
`transport.json`. Do not set project `graph.openai_transport` to configure Hermes.
