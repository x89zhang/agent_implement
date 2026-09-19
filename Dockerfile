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

ARG INSTALL_AGENTSPEC=false
COPY requirements-agentspec.txt /tmp/requirements-agentspec.txt
RUN if [ "$INSTALL_AGENTSPEC" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends git ca-certificates \
      && rm -rf /var/lib/apt/lists/* \
      && pip install --no-cache-dir -r /tmp/requirements-agentspec.txt; \
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
RUN if [ "$INSTALL_AGENT_SECURITY_BENCH" = "true" ]; then \
      python /tmp/install_asb_data.py /opt/agent-security-bench/data; \
    fi

ARG INSTALL_PRIVACYLENS_LIVE=false
COPY scripts/install_privacylens_live_data.py /tmp/install_privacylens_live_data.py
RUN if [ "$INSTALL_PRIVACYLENS_LIVE" = "true" ]; then \
      python /tmp/install_privacylens_live_data.py /opt/privacylens-live/data; \
    fi

ARG INSTALL_AGENTHARM=false
COPY requirements-agentharm.txt /tmp/requirements-agentharm.txt
RUN if [ "$INSTALL_AGENTHARM" = "true" ]; then \
      pip install --no-cache-dir -r /tmp/requirements-agentharm.txt; \
    fi

CMD ["python", "src/agent_scaffold/main.py"]
