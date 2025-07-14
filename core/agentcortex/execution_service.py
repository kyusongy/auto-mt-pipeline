#!/usr/bin/env python3

"""ExecutionService: AgentCortex-based tool execution following agentcortex-lsa patterns.

This service uses the real AgentCortex execution service from agentcortex-lsa,
providing authentic Lenovo service execution for data generation.
"""

from typing import List, Dict, Any, Optional

from agent_types.common import Tool
from agent_types.execution import ToolExecutingRequest, ToolExecutingResponse


class ExecutionService:
    """Tool execution service using AgentCortex services, following agentcortex-lsa patterns."""
    
    def __init__(self, executor_url: str = None, tools_schema: Optional[Dict[str, Any]] = None):
        """Initialize execution service.
        
        If tools_schema is provided, it will be used directly. Otherwise, tools
        will be loaded from the AgentCortex execution service.
        
        Args:
            executor_url: Ignored - only kept for backward compatibility
            tools_schema: Optional dictionary of tool schemas to use directly
        """
        from config.agentcortex_config import is_agentcortex_enabled
        
        # Require AgentCortex services
        if not is_agentcortex_enabled():
            raise RuntimeError(
                "AgentCortex services must be enabled to use ExecutionService. "
                "Set AGENTCORTEX_ENABLED=true and configure service URLs."
            )
        
        from .service_clients import service_clients
        self.service_clients = service_clients
        print("🔧 ExecutionService: Using AgentCortex execution service")
        
        # If tool schemas are passed directly, use them; otherwise, load from service
        if tools_schema:
            self.tools = self._load_tools_from_schema(tools_schema)
        else:
            self.tools = self._load_tools_from_service()
        
    def _load_tools_from_schema(self, tools_schema: Dict[str, Any]) -> List[Tool]:
        """Load tools from a provided schema dictionary."""
        tools = []
        for tool_name, schema in tools_schema.items():
            # Create a Tool object from the schema
            tool_data = {
                "name": tool_name,
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {})
            }
            tools.append(Tool.model_validate(tool_data))
        print(f"✅ ExecutionService loaded {len(tools)} tools from provided schema")
        return tools

    def _load_tools_from_service(self) -> List[Tool]:
        """Load tools from AgentCortex execution service, following agentcortex workflow.py pattern."""
        try:
            # Get tools from AgentCortex execution service
            tools_data = self.service_clients.list_tools()
            agentcortex_tools = tools_data.get("tools", [])
            
            tools = []
            for tool_data in agentcortex_tools:
                # Skip tools that don't need planning (like in agentcortex)
                NO_PLAN_TOOLS = {"solution_click_event", "product_params_compare"}
                if tool_data.get("name") in NO_PLAN_TOOLS:
                    continue
                    
                # Convert to agentcortex Tool format
                tool = Tool.model_validate(tool_data)
                tools.append(tool)
                
            print(f"✅ ExecutionService loaded {len(tools)} tools from AgentCortex")
            return tools
            
        except Exception as e:
            print(f"❌ Failed to load tools from AgentCortex: {e}")
            raise RuntimeError(f"Could not load tools from AgentCortex execution service: {e}")
    

    
    def execute_tools(self, request: ToolExecutingRequest) -> ToolExecutingResponse:
        """Execute tools in plan using AgentCortex execution service.
        
        Args:
            request: ToolExecutingRequest with plan, default_args, session_memory
            
        Returns:
            ToolExecutingResponse with Observation containing execution results
        """
        try:
            print(f"🔧 ExecutionService calling AgentCortex execution service")
            return self.service_clients.execute_tools(request)
        except Exception as e:
            print(f"❌ AgentCortex execution service failed: {e}")
            raise RuntimeError(f"AgentCortex execution service failed: {e}")
    
    def list_tools(self) -> List[Tool]:
        """Return available tools, following agentcortex /list_tools pattern."""
        return self.tools 