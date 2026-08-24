# Pro2Guard

Pro2Guard evaluates each proposed tool call against a learned DTMC and applies
the configured `block`, `warn`, or `monitor` behavior when the probability of
reaching an unsafe state exceeds `threshold`.

## LLM-generated definitions

When Pro2Guard is enabled, its generator is enabled by default. Before the agent
graph is built, the generator uses the main `llm` configuration to propose:

- exact tool-to-category and side-effect profiles for the state abstraction;
- unsafe states selected from the states already present in the JSON DTMC.

The generator uses small, independent LLM calls instead of requesting one large
JSON response. Each tool call returns one tagged profile, and DTMC states are
classified in small batches by returning only numeric indices. Local code assembles
the final data-only JSON and rejects unsupported profiles or out-of-range indices.
It never executes model-generated Python or regular expressions.

The first run in a batch also writes `pro2guard_policy.batch-cache.json` into
the batch directory. Later runs reuse it only when a SHA-256 fingerprint of the
task, tools, DTMC states, generator settings, and model configuration matches.
Single runs do not use this batch cache.

Generated artifacts are written into the run directory:

- `pro2guard_policy_input.json`
- `pro2guard_policy_raw.txt`
- `pro2guard_policy.generated.json`
- `pro2guard_policy_generation.json`

The generated policy or cache hit is also recorded as a `pro2guard_policy_generate` startup
trace step and under `harness.pro2guard.policy_generator`.

Example:

```yaml
pro2guard:
  enabled: true
  mode: block
  model_path: models/pro2guard/dtmc.json
  threshold: 0.1
  unsafe_states: []
  generator:
    enabled: true
    context_mode: benign_only
    max_attempts: 2
    max_profiles: 50
    max_unsafe_states: 20
    max_model_states: 200
    state_batch_size: 8
    fail_closed: true
    llm:
      # Omit these fields to reuse the main llm configuration.
      provider: openai
      model: gpt-5
```

`context_mode: benign_only` excludes hidden benchmark attack metadata from the
generation prompt. Use `full` only when the generator is intentionally allowed
to use configured injection metadata.

With `fail_closed: true` and an empty static `unsafe_states` list, generation must
produce at least one valid unsafe state. Individual malformed profile or state-batch
responses are skipped after `max_attempts`; the run fails only when no usable unsafe
state remains.

## Manual-only and static policy modes

Disable generation while keeping the existing behavior:

```yaml
pro2guard:
  enabled: true
  generator: false
  unsafe_states:
    - ATTACK_SUCCESS
```

A data-only policy can also be supplied manually:

```yaml
pro2guard:
  enabled: true
  generator: false
  abstraction_policy_path: rules/pro2guard_policy.json
```

The policy file uses this shape:

```json
{
  "version": 1,
  "tool_profiles": {
    "send_email": {
      "category": "communication",
      "side_effect": "mutating",
      "rationale": "External side effect"
    }
  },
  "unsafe_states": ["ATTACK_SUCCESS"]
}
```

The policy changes only the default `ToolTraceAbstraction`. A custom Python
abstraction configured through `pro2guard.abstraction` remains authoritative.
