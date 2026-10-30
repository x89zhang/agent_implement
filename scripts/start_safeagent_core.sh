#!/usr/bin/env bash
# Start the upstream SafeAgent Core MCP server for the safeagent guard.
#
#   SAFEGUARD_ENDPOINT_URL=https://openrouter.ai/api/v1 \
#   SAFEGUARD_ENDPOINT_API_KEY=... \
#   scripts/start_safeagent_core.sh [port]          # default 8010
#
# The core is upstream code at a pinned commit, unmodified; it is checked out
# under runtime/ with its own venv. Upstream's __main__ hard-codes port 8000
# (safeagent_core.py), which JANUS uses here, so the same FastMCP server
# object is started on another port. Its LLMs (config/config.yaml):
# openai/gpt-oss-safeguard-20b (encoder, override) and openai/gpt-oss-120b
# (world model, aggregator), all through one OpenAI-compatible endpoint; the
# "openai/" ids and provider routing fields follow OpenRouter.
# The guard's YAML must point at http://127.0.0.1:<port>/mcp (safeagent.mcp_url).
set -euo pipefail

port=${1:-8010}
revision=${SAFEAGENT_CORE_REVISION:-ec7873a}
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/runtime/safeagent_core"
venv="${repo_root}/runtime/venvs/safeagent-core"
python_bin=${PYTHON_BIN:-python3}

for name in SAFEGUARD_ENDPOINT_URL SAFEGUARD_ENDPOINT_API_KEY; do
  if [[ -z "${!name:-}" ]]; then
    echo "$name is required (SafeAgent Core's LLM endpoint)" >&2
    exit 2
  fi
done

if [[ ! -d "${source_dir}/.git" ]]; then
  git clone --quiet https://github.com/SafeAgent-Development/safeagent_core.git "${source_dir}"
fi
git -C "${source_dir}" fetch --quiet origin
git -C "${source_dir}" checkout --quiet "${revision}"
echo "SafeAgent Core at $(git -C "${source_dir}" rev-parse --short HEAD)"

if [[ ! -x "${venv}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv}"
  # Upstream ships no requirements file; these are its imports.
  "${venv}/bin/pip" install --quiet fastmcp langgraph langchain-core langchain-openai numpy pydantic pyyaml
fi

cd "${source_dir}"
exec "${venv}/bin/python" -c "
import sys
sys.path.insert(0, '.')
from safeagent_core import mcp
mcp.run(transport='streamable-http', host='127.0.0.1', port=int(sys.argv[1]))
" "${port}"
