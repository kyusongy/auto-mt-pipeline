#!/usr/bin/env python3
"""Qwen planning + AgentCortex execution hybrid agent.

This class wraps the existing `QwenTestAgent` (planning only) and pipes all
function calls through AgentCortex services so the execution results are
returned to Qwen and the conversation can continue.

It exposes the same `respond(history, tools_schema)` interface as
`QwenTestAgent` so it can be slotted into `TrajectoryCollector` without any
other modifications.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Local application imports
from config import LLMConfig, ASSISTANT_AGENT_OPTIONS, is_agentcortex_enabled
from core.trajectory.pipeline import QwenTestAgent  # reuse existing wrapper
from core.agentcortex.service_clients import AgentCortexServiceClients
from agent_types.execution import ToolExecutingRequest
from agent_types.common import Plan, ToolCalling

__all__ = ["QwenAgentCortexHybrid"]


class QwenAgentCortexHybrid:
    """Hybrid agent that uses Qwen for planning and AgentCortex for execution."""

    def __init__(
        self,
        llm_cfg: LLMConfig,
        generation_opts: Dict[str, Any] | None = None,
        tool_names: Optional[List[str]] = None,
    ) -> None:
        self.qwen_agent = QwenTestAgent(
            llm_cfg, generation_opts or ASSISTANT_AGENT_OPTIONS, tool_names
        )
        self.agentcortex_clients = AgentCortexServiceClients()

        if not is_agentcortex_enabled() or not self.agentcortex_clients.is_available():
            raise RuntimeError("AgentCortex services are not available or disabled")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def respond(self, history: List[dict], tools_schema: List[dict]) -> List[dict]:
        """Generate assistant response, executing any tool calls via AgentCortex."""

        # 1) Let Qwen plan
        qwen_response = self.qwen_agent.respond(history, tools_schema)

        processed_messages: List[dict] = []
        executed_any_call = False

        for msg in qwen_response:
            if msg.get("role") == "assistant" and "function_call" in msg:
                executed_any_call = True
                processed_messages.append(msg)  # keep original call record

                # Convert to Plan and execute via AgentCortex
                fc = msg["function_call"]
                name = fc["name"]
                try:
                    args = json.loads(fc.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                plan = Plan(tool_callings=[ToolCalling(name=name, arguments=args)], content="")
                req = ToolExecutingRequest(plan=plan)
                try:
                    rsp = self.agentcortex_clients.execute_tools(req)
                    obs = rsp.observation
                    if obs.status and obs.status[0].error is None:
                        result_content = json.dumps(obs.status[0].result, ensure_ascii=False)
                    else:
                        err = obs.status[0].error.message if obs.status else "Unknown error"
                        result_content = f"Error: {err}"
                except Exception as e:
                    result_content = f"Error: {e}"

                # Append function result for Qwen follow-up turn
                processed_messages.append(
                    {
                        "role": "function",
                        "name": name,
                        "content": result_content,
                    }
                )
            else:
                processed_messages.append(msg)

        # 2) If any tools executed, ask Qwen for a follow-up text response
        if executed_any_call:
            follow_msgs = self.qwen_agent.respond(history + processed_messages, tools_schema)
            for m in follow_msgs:
                # avoid duplicating the function call again
                if m.get("role") == "assistant" and "function_call" not in m:
                    processed_messages.append(m)

        return processed_messages
