# ToolSafe

ToolSafe is an optional, step-level tool-call guard. Before each tool executes,
it sends the original user request, recent interaction history, proposed action,
and configured tool descriptions to a TS-Guard-compatible model. In `replan` mode, a risky call is withheld and returned to the agent as
security feedback so the agent can reason again and choose a safer action.

ToolSafe is disabled by default. Add this block to an agent YAML file:

```yaml
toolsafe:
  enabled: true
  mode: replan                # replan / block / warn / monitor
  threshold: 0.5
  provider: openai_compatible
  model: TS-Guard
  base_url: http://localhost:8001/v1
  api_key_env: TOOLSAFE_API_KEY
  timeout_seconds: 30
  max_history_steps: 20
  max_replans: 3
  fail_closed: false
```

Set `TOOLSAFE_API_KEY` in the environment when the endpoint requires
authentication. `base_url` may be either the server root, its `/v1` endpoint, or
the full `/chat/completions` URL.

Modes:

- `replan`: prevent only the current risky call, return the TS-Guard analysis as
  an Observation, and let the agent reason again. After `max_replans` risky
  replans, ToolSafe escalates to a terminal block.
- `block`: prevent the risky call and terminate the current agent run immediately.
- `warn`: execute the call but append the warning to its result before the next
  model turn.
- `monitor`: execute the call and only record the decision in the trace.

If the endpoint or response parser fails, `fail_closed: true` treats the check as
unsafe and applies the selected mode; `false` allows it and records the error. Trace tool steps and trace messages expose
the complete decision under the `toolsafe` key. API keys are not written there.

The guard accepts the ToolSafe score set `0.0`, `0.5`, and `1.0`, including the
paper's `<Judgment>...</Judgment>` response form and a JSON `risk_score` form.
