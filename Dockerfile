FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ARG INSTALL_LLAMA_FIREWALL=false
COPY requirements-llamafirewall.txt /tmp/requirements-llamafirewall.txt
RUN if [ "$INSTALL_LLAMA_FIREWALL" = "true" ]; then \
      pip install --no-cache-dir -r /tmp/requirements-llamafirewall.txt; \
    fi

ARG INSTALL_SAFEAGENT=false
COPY requirements-safeagent.txt /tmp/requirements-safeagent.txt
RUN if [ "$INSTALL_SAFEAGENT" = "true" ]; then \
      pip install --no-cache-dir -r /tmp/requirements-safeagent.txt; \
    fi

ARG INSTALL_AGENTSPEC=false
COPY requirements-agentspec.txt /tmp/requirements-agentspec.txt
RUN if [ "$INSTALL_AGENTSPEC" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends git ca-certificates \
      && rm -rf /var/lib/apt/lists/* \
      && pip install --no-cache-dir -r /tmp/requirements-agentspec.txt; \
    fi

ARG INSTALL_PROGENT=false
COPY requirements-progent.txt /tmp/requirements-progent.txt
RUN if [ "$INSTALL_PROGENT" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends git ca-certificates \
      && rm -rf /var/lib/apt/lists/* \
      && pip install --no-cache-dir -r /tmp/requirements-progent.txt; \
    fi

ARG INSTALL_CLAWSENTRY=false
COPY requirements-clawsentry.txt /tmp/requirements-clawsentry.txt
RUN if [ "$INSTALL_CLAWSENTRY" = "true" ]; then \
      python -m venv /opt/clawsentry-venv \
      && /opt/clawsentry-venv/bin/pip install --no-cache-dir -r /tmp/requirements-clawsentry.txt; \
    fi

# ADR Detection is a research artifact; install only its detector runtime in
# an isolated environment, never its full benchmark dependency set.
ARG INSTALL_ADR=false
ARG ADR_REVISION=1c8ecd631e5ed19afc4d61ec1c309a94445476f8
COPY scripts/install_adr_source.py /tmp/install_adr_source.py
COPY requirements-adr.txt /tmp/requirements-adr.txt
RUN if [ "$INSTALL_ADR" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends nodejs npm ca-certificates \
      && rm -rf /var/lib/apt/lists/* \
      && python /tmp/install_adr_source.py "$ADR_REVISION" /opt/adr/Detection \
      && mkdir -p /opt/adr/Detection/ads_reasoning_workspace_agentdojo /opt/adr/Detection/ads_reasoning_workspace \
      && chmod 1777 /opt/adr/Detection/ads_reasoning_workspace_agentdojo /opt/adr/Detection/ads_reasoning_workspace \
      && python -m venv /opt/adr-venv \
      && /opt/adr-venv/bin/pip install --no-cache-dir -r /tmp/requirements-adr.txt \
      && npm install -g @anthropic-ai/claude-code \
      && /opt/adr-venv/bin/python -c "import openai, mcp, yaml; import sys; sys.path.insert(0, '/opt/adr/Detection'); from guardrail.adr_agent.adr_baseline import ADRBaseline" \
      && claude --version; \
    fi

# Optional benchmark extras. Set --build-arg INSTALL_AGENTDOJO=true if the
# `agentdojo` package is available from your configured Python package index,
# or use container.image to point at a custom image that already includes it.
ARG INSTALL_AGENTDOJO=false
RUN if [ "$INSTALL_AGENTDOJO" = "true" ]; then \
      pip install --no-cache-dir agentdojo; \
    fi

ARG INSTALL_AGENT_SECURITY_BENCH=false
COPY scripts/install_asb_data.py /tmp/install_asb_data.py
COPY scripts/install_asb_source.py /tmp/install_asb_source.py
COPY requirements-asb-bridge.txt /tmp/requirements-asb-bridge.txt
ARG ASB_REVISION=eac7bcf38c116f42b46e6d480e56e599b99e73c2
RUN if [ "$INSTALL_AGENT_SECURITY_BENCH" = "true" ]; then \
      python /tmp/install_asb_source.py "$ASB_REVISION" /opt/agent-security-bench \
      && python -m venv /opt/asb-venv \
      && /opt/asb-venv/bin/pip install --no-cache-dir -r /tmp/requirements-asb-bridge.txt; \
    fi

ARG INSTALL_PRIVACYLENS_LIVE=false
COPY scripts/install_privacylens_live_source.py /tmp/install_privacylens_live_source.py
COPY scripts/install_privacylens_evaluator_source.py /tmp/install_privacylens_evaluator_source.py
ARG PRIVACYLENS_LIVE_REVISION=994ac15db6fff8a5131bbf5a26e84e352e676796
ARG PRIVACYLENS_EVALUATOR_REVISION=9c2ee07b080dc54ed4924af11d9751e81753c94d
RUN if [ "$INSTALL_PRIVACYLENS_LIVE" = "true" ]; then \
      python /tmp/install_privacylens_live_source.py "$PRIVACYLENS_LIVE_REVISION" /opt/privacylens-live \
      && python /tmp/install_privacylens_evaluator_source.py "$PRIVACYLENS_EVALUATOR_REVISION" /opt/privacylens-evaluator \
      && python -m venv /opt/privacylens-live-venv; \
    fi

ARG INSTALL_AGENTHARM=false
COPY requirements-agentharm.txt /tmp/requirements-agentharm.txt
RUN if [ "$INSTALL_AGENTHARM" = "true" ]; then \
      pip install --no-cache-dir -r /tmp/requirements-agentharm.txt; \
    fi

CMD ["python", "src/agent_scaffold/main.py"]
