FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Optional benchmark extras. Set --build-arg INSTALL_AGENTDOJO=true if the
# `agentdojo` package is available from your configured Python package index,
# or use container.image to point at a custom image that already includes it.
ARG INSTALL_AGENTDOJO=false
RUN if [ "$INSTALL_AGENTDOJO" = "true" ]; then \
      pip install --no-cache-dir agentdojo; \
    fi

CMD ["python", "src/agent_scaffold/main.py"]
