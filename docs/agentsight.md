# AgentSight integration

The scaffold can run AgentSight as an optional, run-scoped observer. It is disabled by default and operates independently of tool middleware such as Aegis and Pro2Guard.

## Configuration

```yaml
agentsight:
  enabled: true
  # full captures TLS/LLM payloads plus process, file, and resource events.
  # system disables TLS payload capture and records system activity only.
  capture: full
  binary: agentsight
  privilege: auto       # auto, sudo, or none
  required: false       # fail the agent run if capture or export fails
  db_path: agentsight.db
  snapshot_path: agentsight_snapshot.json
  log_path: agentsight.log
  web_server: false
  server_port: 7395
  startup_timeout_seconds: 10
  warmup_seconds: 1
  shutdown_timeout_seconds: 10
```

`agentsight: true` is accepted as shorthand and uses full capture. Every configured artifact path is relative to the job directory; absolute paths and `..` traversal are rejected.

`privilege: auto` uses non-interactive sudo when it is already authorized and otherwise launches AgentSight directly, allowing installations that grant the binary suitable eBPF capabilities. `sudo` requires `sudo -n` to work. `none` always launches the binary directly. The integration never opens an interactive sudo prompt.

## Runtime behavior

For a host run, the observer attaches to the scaffold process before the graph starts and follows its process activity. It stops after graph execution, exports a static snapshot, and stores the final status under `result.harness.agentsight` and the normal trace payload.

For a Docker run, AgentSight remains on the host. The runtime:

1. starts a uniquely named, unprivileged Agent container in detached mode;
2. holds the in-container scaffold at a file-based start gate;
3. resolves the container's host PID and starts AgentSight with `docker://<name>`;
4. releases the gate only after the observer has survived its configured warm-up;
5. waits for the Agent, stops capture, exports the snapshot, and applies the existing container cleanup policy.

The Agent container does not receive `--privileged`, host PID access, or eBPF capabilities. A failed startup is also cleaned up even when `container.remove` is false, so a gated orphan is not left running.

## Job artifacts

An enabled run writes:

- `agentsight.db`: the captured SQLite session;
- `agentsight_snapshot.json`: a portable dashboard/report export;
- `agentsight.log`: capture and export diagnostics;
- `agentsight_status.json`: machine-readable lifecycle state.

When sudo is used, the integration restores database and snapshot ownership to the invoking user after export.

With `required: false`, an unavailable binary, insufficient privileges, capture failure, or export failure is represented as `failed` or `degraded` in the result while the Agent run continues. With `required: true`, the same condition fails the run.

## Privacy

`capture: full` can store prompts, responses, tool payloads, file paths, HTTP headers, and network destinations. Treat the database and snapshot as sensitive. Select `capture: system` when payload capture is not acceptable; this runs AgentSight with SSL capture disabled while retaining process and resource observation.
