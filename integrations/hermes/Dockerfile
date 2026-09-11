ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-hermes-bridge.txt /tmp/requirements-hermes-bridge.txt
RUN pip install --no-cache-dir -r /tmp/requirements-hermes-bridge.txt
COPY hermes-source /opt/hermes-agent
RUN python -m venv /opt/hermes-venv \
    && /opt/hermes-venv/bin/pip install --no-cache-dir -e '/opt/hermes-agent[mcp]'
RUN git config --system --add safe.directory /opt/hermes-agent
