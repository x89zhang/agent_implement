# AgentDoG integration

The scaffold can run AgentDoG 1.5 as an optional trajectory-level safety
diagnostic. It is disabled by default. Unlike tool-scoped guards, AgentDoG
reviews the accumulated agent trajectory: user requests, agent decisions, tool
calls, environment results, and the candidate action or final response.

The integration preserves AgentDoG's two-stage method:

1. classify the agent's actual or imminent behavior as `safe` or `unsafe`;
2. for unsafe trajectories, diagnose Failure Mode, Risk Consequence, and Risk
   Source using the AgentDoG 1.5 taxonomy.

Malicious user input or an injected tool result is not itself enough to make a
trajectory unsafe. The judgment concerns how the agent handled it.

## Configuration

```yaml
agentdog:
  enabled: true

  # diagnose: record the complete diagnosis without changing execution
  # revise: request a safer candidate, then gate if max_revisions is exhausted
  # gate: withhold an unsafe pending action or final response immediately
  mode: diagnose

  # unified emits the three-dimensional taxonomy; coarse emits safe/unsafe plus
  # the official four-question analysis.
  task: unified

  # pre_reply is the method-faithful default. pre_action is optional and reviews
  # an imminent tool call together with the accumulated trajectory.
  checkpoints: [pre_reply]

  provider: openai_compatible
  model: AgentDoG1.5-Unified-Qwen3.5-4B
  base_url: http://127.0.0.1:8000/v1
  base_url_env: AGENTDOG_BASE_URL
  api_key_env: AGENTDOG_API_KEY

  timeout_seconds: 60
  temperature: 0.0
  max_tokens: 1024

  # 0 preserves the complete trajectory. A positive value explicitly enables
  # oldest-step truncation while always retaining the latest candidate.
  max_trajectory_chars: 0

  max_revisions: 2
  fail_closed: false
  include_raw_response: true
  replacement_message: >-
    The response was withheld because AgentDoG diagnosed unsafe behavior in the
    accumulated agent trajectory.
```

`agentdog: true` is accepted as shorthand and uses the defaults. In that form,
set `AGENTDOG_BASE_URL`; the expanded mapping can instead supply `base_url`
directly. The endpoint must expose an OpenAI-compatible
`POST /v1/chat/completions` API.

Use a checkpoint that matches `task`:

- `unified`: an AgentDoG 1.5 Unified checkpoint, which returns safety plus the
  three diagnostic dimensions;
- `coarse`: an AgentDoG 1.5 coarse-grained moderation checkpoint, which returns
  the four-part rationale and binary judgment.

## Checkpoints and modes

| Checkpoint | `diagnose` | `revise` | `gate` |
| --- | --- | --- | --- |
| `pre_reply` | Record diagnosis and release the original candidate | Regenerate only the final response, without running tools again; re-evaluate every revision | Replace an unsafe candidate with `replacement_message` |
| `pre_action` | Record diagnosis and execute the proposed tool | Return the diagnosis as a non-executed tool result so the agent can replan | Stop the run before the unsafe tool executes |

`pre_reply` is the default because it matches AgentDoG's accumulated-trajectory
guardrail design. It cannot undo side effects from tools that already ran. Add
`pre_action` when risky tool effects must be reviewed before execution.

In `revise` mode, every new candidate is independently re-evaluated. Once
`max_revisions` unsafe revisions have been requested, the next unsafe candidate
is gated. For the LangChain ReAct backend, final-response revision uses the base
chat model directly and is explicitly prohibited from invoking tools, so the
completed trajectory is not replayed.

## Trajectory conversion

The adapter converts the scaffold's native trace into AgentDoG's conversation
format:

```text
=== Agent Profile ===
<system prompt and available tool definitions>

=== Conversation History ===
[USER]: ...
[AGENT]:
[THOUGHT]: ...
[ACTION]: {"name": "...", "arguments": {...}}
[ENVIRONMENT]: ...
[AGENT]:
[CONTENT]: <candidate final response>
```

Both graph backends are supported:

- `single_agent` and other native graph modes use `trace_messages` directly;
- `langchain_react` normalizes in-flight intermediate steps before PRE_REPLY,
  including tool name, arguments, thought/log text, result, and failed tool
  observations. Those steps are available to AgentDoG even though the normal
  trace is finalized only after the ReAct executor returns.

By default, no trajectory steps are discarded. If `max_trajectory_chars` is
positive, older conversation steps are removed with an explicit omission marker.
The adapter raises an evaluation error instead of silently dropping the latest
candidate or the agent profile.

## Result and trace schema

Every evaluation adds a standalone `agentdog` trace event and updates
`harness.agentdog`. A representative result is:

```json
{
  "judgment": "unsafe",
  "safe": false,
  "analysis": "The agent trusted an injected tool result ...",
  "failure_modes": ["Failure to Validate Tool Outputs"],
  "risk_consequences": ["Privacy & Confidentiality Harm"],
  "risk_sources": ["Indirect Prompt Injection"],
  "checkpoint": "pre_reply",
  "task": "unified",
  "mode": "diagnose",
  "action": "diagnose",
  "model": "AgentDoG1.5-Unified-Qwen3.5-4B",
  "latency_ms": 842,
  "trajectory_steps": 12,
  "trajectory_chars": 14520,
  "truncated": false,
  "revision_count": 0,
  "error": "",
  "usage": {
    "prompt_tokens": 3610,
    "completion_tokens": 180,
    "total_tokens": 3790
  }
}
```

Pre-action results are also attached to the corresponding tool trace metadata as
`agentdog`. `harness.agentdog.latest` contains the most recent result,
`event_count` counts evaluations, and `usage` accumulates judge-model tokens
separately from the task agent's model statistics.

The parser enforces the official output contract. A malformed response is an
evaluation error rather than an implicit safe verdict.

## Failure behavior

- With `fail_closed: false`, endpoint, timeout, formatting, and context-limit
  errors are recorded as `judgment: error`; execution continues.
- With `fail_closed: true`, `revise` and `gate` modes withhold the affected
  candidate when evaluation fails.
- `diagnose` remains observational even when `fail_closed` is true. This avoids a
  configuration named `diagnose` unexpectedly changing agent behavior.

## Middleware interaction

AgentDoG is registered as an independent optional middleware. Tool-scoped guards
run first. AgentDoG then evaluates the accumulated trajectory and candidate;
AgentGuard's output policy runs afterward. AgentDoG does not reuse Pro2Guard's
DTMC probability, threshold, or block semantics.

Enabling AgentDoG sends prompts, tool arguments, tool results, and candidate
responses to the configured judge endpoint. These data can contain credentials,
private documents, or injected content. Use a trusted endpoint and protect trace
files. Set `include_raw_response: false` when the judge's raw rationale should not
be retained, although the parsed analysis and labels remain in the trace.
