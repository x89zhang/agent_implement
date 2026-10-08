"""AlignmentCheck on a self-hosted OpenAI-compatible judge (``llamafirewall.factory``).

Upstream's ``AlignmentCheckScanner`` takes no endpoint arguments and always
calls Llama-4-Maverick on Together AI (``custom_check_scanner.py``). This
factory registers a subclass that keeps upstream's prompt, output schema,
trace rendering and decision mapping, and only replaces the ``LLMClient``
endpoint and model. The endpoint must support structured outputs
(``beta.chat.completions.parse``); vLLM does through ``response_format``.

Using a judge other than the paper's model is a protocol deviation; record
the served model with the results.
"""

from __future__ import annotations

import os
import re
from typing import Any

_SCANNER_PREFIX = "agent_alignment_local"


def alignment_check(
    base_url: str,
    model: str,
    api_key_env: str = "",
    temperature: float = 0.0,
) -> Any:
    """Return a LlamaFirewall that runs AlignmentCheck on ``model`` at ``base_url``."""
    from llamafirewall import LlamaFirewall, Role, register_llamafirewall_scanner
    from llamafirewall.scanners.base_scanner import Scanner
    from llamafirewall.scanners.experimental import alignmentcheck_scanner as upstream
    from llamafirewall.utils.base_llm import LLMClient

    if not base_url or not model:
        raise ValueError("llamafirewall.factory_kwargs needs base_url and model")
    # vLLM accepts any key unless started with --api-key.
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    name = f"{_SCANNER_PREFIX}_{re.sub(r'[^A-Za-z0-9]+', '_', model).strip('_').lower()}"

    class LocalAlignmentCheckScanner(upstream.AlignmentCheckScanner):
        def __init__(self, scanner_name: str = "AlignmentCheck Scanner") -> None:
            # The same state AlignmentCheckScanner/CustomCheckScanner.__init__
            # set, without their Together client, which requires
            # TOGETHER_API_KEY at construction.
            Scanner.__init__(self, scanner_name=scanner_name, block_threshold=0.0)
            self.system_prompt = upstream.SYSTEM_PROMPT
            self.output_schema = upstream.AlignmentCheckOutputSchema
            self.temperature = temperature
            self.llm = LLMClient(
                model_name=model, api_base_url=base_url, api_key=api_key or "EMPTY"
            )
            self.require_full_trace = True

    register_llamafirewall_scanner(name)(LocalAlignmentCheckScanner)
    return LlamaFirewall(scanners={Role.ASSISTANT: [name]})
