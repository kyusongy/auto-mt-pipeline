#!/usr/bin/env python3

"""PlanLLM: Planning agent following agentcortex-lsa patterns.

This module provides a Plan LLM that generates Plan objects with tool_callings
and content using the real AgentCortex planning service.
"""

from typing import Optional

from agent_types.planning import PlanningRequest, PlanningResponse
from config import LLMConfig, GenerationOptions
from config.agentcortex_config import is_agentcortex_enabled


class PlanLLM:
    """Plan LLM that generates Plan objects following agentcortex-lsa planning patterns."""
    
    def __init__(self, llm_config: LLMConfig, generation_options: Optional[GenerationOptions] = None):
        """Initialize Plan LLM.
        
        Args:
            llm_config: LLM configuration  
            generation_options: Generation options for the LLM
        """
        self.llm_config = llm_config
        self.generation_options = generation_options or GenerationOptions(
            temperature=0.1,
            max_tokens=2048
        )
        
        # Require AgentCortex services
        if not is_agentcortex_enabled():
            raise RuntimeError(
                "AgentCortex services must be enabled to use PlanLLM. "
                "Set AGENTCORTEX_ENABLED=true and configure service URLs."
            )
        
        from .service_clients import service_clients
        self.service_clients = service_clients
        print("🧠 PlanLLM: Using AgentCortex planning service")
    
    def plan(self, request: PlanningRequest) -> PlanningResponse:
        """Generate planning response using AgentCortex planning service.
        
        Args:
            request: PlanningRequest with task, tools, observations, etc.
            
        Returns:
            PlanningResponse with Plan containing tool_callings and content
        """
        try:
            print(f"🧠 PlanLLM calling AgentCortex planning service for task: {request.task}")
            return self.service_clients.plan(request)
        except Exception as e:
            print(f"❌ AgentCortex planning service failed: {e}")
            raise RuntimeError(f"AgentCortex planning service failed: {e}") 