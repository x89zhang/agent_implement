# LlamaFirewall integration

This adapter calls the native `llamafirewall` package. ReAct scans each parsed
action (reasoning, tool name, arguments) before tool dispatch, tool observations
after execution, and final replies once. Initial system/user messages are also
scanned. Explicit scanner mappings replace library defaults, as in upstream.

`enforce` stops on native `block` or `human_in_the_loop_required` decisions.
There is no automatic rewrite/replan. Human-review verdicts remain distinct;
the upstream library provides a verdict, not a generic approval/resume service.
Stopping after a tool-result scan cannot undo the tool's earlier side effects.
The legacy `max_revisions` field is accepted but no longer used.

Enforce uses upstream's allow-only incremental trace. `monitor` is an optional
harness observation mode, not an upstream mode: it allows original messages
and keeps them all in the scanner trace, including rejected messages.

Defaults are unchanged: user=PromptGuard, tool=CodeShield+PromptGuard,
assistant=CodeShield. AlignmentCheck must be enabled explicitly with
`assistant: [agent_alignment, code_shield]`. It uses its own model endpoint,
not the agent's top-level LLM configuration.

For native constructor options or custom scanners, supply a Python factory:

```yaml
llamafirewall:
  enabled: true
  mode: enforce
  factory: my_scanners:build_firewall
  factory_kwargs:
    threshold: 0.8
```

```python
# my_scanners.py, importable inside the agent's runtime
from llamafirewall import LlamaFirewall, Role
from llamafirewall.llamafirewall import register_llamafirewall_scanner
from llamafirewall.scanners.prompt_guard_scanner import PromptGuardScanner

def build_firewall(threshold=0.9):
    @register_llamafirewall_scanner("configured_prompt_guard")
    class ConfiguredPromptGuard(PromptGuardScanner):
        def __init__(self):
            super().__init__(block_threshold=threshold)
    return LlamaFirewall(scanners={
        Role.USER: ["configured_prompt_guard"],
        Role.TOOL: ["configured_prompt_guard"],
    })
```

Factories receive `factory_kwargs` unchanged, so native custom scanner classes
can configure model, endpoint, threshold, and prompt through their supported
Python APIs. A factory cannot be combined with `scanners` or `use_case`.
Keep factories in trusted application code. For Docker builds enable
`container.build_args.INSTALL_LLAMA_FIREWALL: "true"`; provide scanner model
files/API credentials as required. `fail_closed` controls adapter exceptions;
scanner-returned native decisions are preserved.
