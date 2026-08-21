# Optional LlamaFirewall middleware

The scaffold can use Meta's LlamaFirewall as an optional, role-aware safety
middleware. It is disabled by default and is independent of Pro2Guard, ToolSafe,
AgentDoG, and AgentGuard.

## Installation

LlamaFirewall is deliberately not part of the base requirements because it
installs Torch, Transformers, CodeShield, and model-related dependencies.

For a local run:

```bash
python -m pip install -r requirements-llamafirewall.txt
llamafirewall configure
```

The integration is pinned to `llamafirewall==1.0.3`, the current PyPI release.
The configuration helper checks or downloads the required Hugging Face models.

For the scaffold's default container flow, enable the optional image layer:

```yaml
container:
  enabled: true
  build_args:
    INSTALL_LLAMA_FIREWALL: "true"
```

The container runner propagates `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
`TOGETHER_API_KEY`, and `TOKENIZERS_PARALLELISM` when they are present in
the host environment.

## Configuration

Use LlamaFirewall's native default role mapping:

```yaml
llamafirewall:
  enabled: true
  mode: enforce
  max_revisions: 1
  fail_closed: false
```

Use one of LlamaFirewall's predefined use cases:

```yaml
llamafirewall:
  enabled: true
  use_case: chatbot  # chatbot or coding_assistant
  mode: enforce
```

Or compose scanners explicitly:

```yaml
llamafirewall:
  enabled: true
  mode: enforce
  scanners:
    user: [prompt_guard]
    assistant: [agent_alignment, code_shield]
    tool: [prompt_guard, code_shield]
    system: []
  max_revisions: 1
  fail_closed: false
```

Supported built-in scanner names follow LlamaFirewall's `ScannerType` values:
`prompt_guard`, `code_shield`, `agent_alignment`, `hidden_ascii`,
`pii_detection`, and `regex`. Registered custom scanner names can also be
used. Supplying `scanners` replaces LlamaFirewall's default mapping; omitted
roles have no scanners.

Start with `mode: monitor` to collect native decisions without changing agent
behavior:

```yaml
llamafirewall:
  enabled: true
  mode: monitor
  scanners:
    user: [prompt_guard]
    assistant: [agent_alignment]
```

Every scan is appended to `state["llamafirewall_events"]` with its phase,
role, native decision, reason, score, status, and integration mode.

## Lifecycle and decision semantics

The adapter preserves LlamaFirewall's role and trace model:

1. Initial system and user messages are scanned before the first model call.
2. Every assistant response is scanned before it is accepted. A proposed tool
   call is represented both as `tool_calls` and as a rendered selected action,
   so AlignmentCheck can evaluate it against the prior trace.
3. Tool output is scanned as a `ToolMessage`.
4. `scan_replay_build_trace` maintains a per-run LlamaFirewall trace. As in
   LlamaFirewall itself, only messages with an `ALLOW` decision enter that
   trusted trace.

In `enforce` mode:

- `ALLOW` continues normally.
- `BLOCK` on model output removes any proposed tool call and asks the model
  for a safe revision, up to `max_revisions`. Blocked initial input does not
  reach the model. Blocked tool output is replaced before the model sees it.
- `HUMAN_IN_THE_LOOP_REQUIRED` remains distinct from `BLOCK`: the current
  action is withheld and the run terminates with the native decision recorded.
  The scaffold does not currently have a resumable human-approval queue.
- Scanner/package failures obey `fail_closed`. The recommended initial setting
  is `false` while deploying in monitor mode, then an explicit policy choice
  before enforcement.

In `monitor` mode, all three native decisions are recorded, but the adapter
does not change model output or tool results.

## Operational considerations

- Prompt Guard downloads/loads a local classifier on first use.
- CodeShield statically scans generated code and adds CPU latency.
- AlignmentCheck evaluates the accumulated agent trace through an LLM and
  currently requires `TOGETHER_API_KEY`; it is more expensive than Prompt
  Guard and is best assigned to assistant actions rather than every role.
- A tool-output scanner runs after the tool has executed. It can prevent an
  untrusted result from steering later model steps, but cannot undo an external
  side effect. AlignmentCheck on the assistant's selected action is the
  LlamaFirewall layer that can intervene before dispatch.
