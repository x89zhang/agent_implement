#!/usr/bin/env bash
# Start/stop a fresh upstream AEGIS Gateway for one evaluation batch.
#
#   scripts/aegis_gateway.sh start [name] [port]   # fresh DB, waits for /health
#   scripts/aegis_gateway.sh stop  [name]
#
# The Gateway keeps behavioral profiles, pending checks and usage counters in
# its SQLite DB, so verdicts would otherwise depend on earlier batches. The DB
# lives on a tmpfs and is discarded on stop. Build the image from paper/Aegis
# with `docker compose build gateway` (tagged aegis-gateway:latest below) or
# set AEGIS_GATEWAY_IMAGE.
#
# Two deployment settings keep a batch from being cut off mid-run (the
# research baseline retries 429s, aegis_rules_http.py:61-67): RATE_LIMIT_MAX
# is raised (default 100 checks/agent/min, config.ts:29), and the "default"
# org is put on upstream's own enterprise plan so the Free plan's 1000
# checks/month quota (services/billing.ts) does not return 429s. Neither
# changes classification, policy, DSL or anomaly verdicts.
set -euo pipefail

command=${1:-start}
name=${2:-aegis-gateway-eval}
port=${3:-8080}
image=${AEGIS_GATEWAY_IMAGE:-aegis-gateway:latest}

case "$command" in
  start)
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" -p "127.0.0.1:${port}:8080" \
      --tmpfs /data:uid=1000,gid=1000 \
      -e NODE_ENV=production -e PORT=8080 -e DB_PATH=/data/agentguard.db \
      -e RATE_LIMIT_MAX="${AEGIS_RATE_LIMIT_MAX:-1000000}" \
      "$image" >/dev/null
    for _ in $(seq 1 60); do
      if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        docker exec "$name" node -e "
          const Database = require('/app/node_modules/better-sqlite3');
          const db = new Database(process.env.DB_PATH);
          db.prepare(\"INSERT INTO org_subscriptions (org_id, plan, status) VALUES ('default', 'enterprise', 'active') ON CONFLICT(org_id) DO UPDATE SET plan='enterprise', status='active'\").run();
        "
        echo "AEGIS Gateway $name ready at http://127.0.0.1:${port} (image $image, fresh DB)"
        docker image inspect "$image" --format 'image id: {{.Id}}'
        exit 0
      fi
      sleep 1
    done
    docker logs "$name" >&2 || true
    echo "AEGIS Gateway did not become healthy" >&2
    exit 1
    ;;
  stop)
    docker rm -f "$name" >/dev/null
    ;;
  *)
    echo "usage: $0 start|stop [name] [port]" >&2
    exit 2
    ;;
esac
