from __future__ import annotations

"""Adapters: expose dummy_tools.* functions as Qwen-Agent tools.

Each dummy tool is wrapped into a `BaseTool` subclass and auto-registered via
`@register_tool`.  Qwen-Agent's Assistant can then invoke them seamlessly.
"""

import json
from typing import Any

from qwen_agent.tools.base import BaseTool, register_tool  # type: ignore

from tools import retail_tools as _d


# Helper to dynamically create wrappers ------------------------------------------------------------

def _make_tool_cls(name: str, spec: dict):  # noqa: D401 – simple factory
    """Return a BaseTool subclass that proxies to the corresponding dummy function."""

    _desc: str = spec.get("description", name)
    params_schema: dict = spec.get("parameters", {})

    # Convert OpenAI-style schema to Qwen-Agent list-of-dicts format
    _params: list[dict[str, Any]] = []
    for k, meta in params_schema.items():
        _params.append(
            {
                "name": k,
                "type": meta.get("type", "string"),
                "description": meta.get("description", ""),
                "required": meta.get("required", False),
            }
        )

    func = _d.RUNTIME_FUNCTIONS[name]

    @register_tool(name)
    class _Tool(BaseTool):
        description = _desc
        parameters = _params

        def call(self, params: str, **kwargs):  # type: ignore[override]
            data = json.loads(params) if isinstance(params, str) else params
            try:
                result = func(**data)
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
            return json.dumps(result, ensure_ascii=False)

    _Tool.__name__ = f"QwenTool_{name}"
    return _Tool


# Generate classes for every dummy tool -----------------------------------------------------------
for _tool_name, _spec in _d.TOOLS_SCHEMA.items():
    _make_tool_cls(_tool_name, _spec)

# The created classes are automatically registered via decorator. No exports needed. 