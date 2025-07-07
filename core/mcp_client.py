"""Lightweight MCP client for auto-mt-pipeline.

Uses libentry.mcp.client.APIClient to communicate with AgentCortex-LSA MCP executor service,
letting the server handle all parameter injection and tool execution.
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from libentry.mcp.client import APIClient


@dataclass
class MCPConfig:
    """Configuration for MCP integration."""
    executor_url: str
    default_args: Optional[Dict[str, Any]] = None


class MCPClient:
    """Client for interacting with AgentCortex-LSA MCP executor service."""
    
    def __init__(self, config: MCPConfig):
        self.executor_url = config.executor_url
        # Initialize the libentry APIClient
        self.client = APIClient(self.executor_url)
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """Execute a tool via MCP executor service.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments provided by the agent (business parameters only)
            query: Current user query for context
            
        Returns:
            Tool execution result (raw from MCP service)
        """
        # Only use the business parameters provided by the agent
        # Remove any MCP parameters that somehow leaked through
        clean_args = {**tool_args}
        mcp_params = {"trace_id", "uid", "user_info", "terminal", "chat_history", "mentions", "session_preference", 
                     "latitude", "longitude", "device_ip", "get_position_permisson", "event", "bind_mobile_id"}
        for param in mcp_params:
            clean_args.pop(param, None)
        
        # Prepare the plan with tool calling using ONLY agent-provided parameters
        plan_data = {
            "tool_callings": [
                {
                    "name": tool_name,
                    "arguments": clean_args
                }
            ],
            "content": "",
            "thinking": ""
        }
        
        # Use exact default_args from workflow.py lines 105-115
        default_args = {
            "user_info": {"uid": "13716255679", "user_identity": 1, "available_num": 0.0, "current_amount": "0", "enterprise_name": "", "future_expire_num": 0.0, "level_name": "", "entry_source": "shop", "user_province": ""},
            "trace_id": "auto-mt-pipeline-session",  # Using a session identifier
            "uid": "13716255679",
            "terminal": "1",
            "latitude": "23.89447712420573",
            "longitude": "106.6172117534938",
            "device_ip": "117.183.16.69",
            "get_position_permisson": "agree",
            "event": "",
            "bind_mobile_id": 0,
            "query": query
        }
        
        request_data = {
            "plan": plan_data,
            "task": query,
            "intent": None,
            "session_memory": None,
            "default_args": default_args  # Explicitly provide like workflow does
        }
        
        try:
            # Use libentry APIClient to execute tools
            result = self.client.post("execute_tools", request_data)
            
            # Return the original raw MCP response without any filtering
            return result
                
        except Exception as e:
            print(f"❌ MCP tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_tools(self) -> Dict[str, Any]:
        """Get list of available tools from MCP service."""
        try:
            # Use libentry APIClient to list tools
            result = self.client.post("list_tools", {})
            return result
        except Exception as e:
            print(f"❌ Error fetching MCP tools: {e}")
            return {"tools": []}


def filter_mcp_parameters(tool_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out MCP-injected parameters from tool schema.
    
    These parameters are automatically injected by MCP and should not
    be visible to the agent during blueprint generation.
    """
    # Parameters that MCP injects automatically - agent should NEVER see these
    mcp_injected_params = {
        "trace_id", "uid", "user_info", "terminal", "latitude", 
        "longitude", "device_ip", "get_position_permisson", "event",
        "bind_mobile_id", "chat_history", "mentions", "session_preference"
    }
    
    filtered_schema = tool_schema.copy()
    
    # Handle the actual MCP format which uses input_schema
    if "input_schema" in filtered_schema and "properties" in filtered_schema["input_schema"]:
        original_properties = filtered_schema["input_schema"]["properties"]
        filtered_properties = {}
        
        # Only include parameters that are NOT MCP-injected
        for param_name, param_spec in original_properties.items():
            if param_name not in mcp_injected_params:
                filtered_properties[param_name] = param_spec
        
        # Update the schema with filtered parameters
        filtered_schema["input_schema"]["properties"] = filtered_properties
        
        # Also update required fields to remove MCP parameters
        if "required" in filtered_schema["input_schema"]:
            original_required = filtered_schema["input_schema"]["required"]
            filtered_required = [param for param in original_required if param not in mcp_injected_params]
            filtered_schema["input_schema"]["required"] = filtered_required
        
        # Convert to the format expected by Qwen Agent
        filtered_schema["parameters"] = filtered_properties
        
        print(f"🔧 Filtered tool '{tool_schema.get('name', 'unknown')}': {len(original_properties)} -> {len(filtered_properties)} params")
        print(f"   Removed MCP params: {set(original_properties.keys()) - set(filtered_properties.keys())}")
        
    else:
        print(f"⚠️  Tool schema format not recognized for '{tool_schema.get('name', 'unknown')}'")
    
    return filtered_schema


def get_mcp_tool_schemas(mcp_client: MCPClient) -> Dict[str, Dict[str, Any]]:
    """Get filtered tool schemas from MCP service.
    
    Returns schemas with MCP-injected parameters removed so the agent
    only sees the parameters it needs to provide.
    """
    tools_response = mcp_client.list_tools()
    
    # libentry APIClient returns tools directly under 'tools' key
    tools = tools_response.get("tools", [])
    
    filtered_schemas = {}
    for tool in tools:
        if isinstance(tool, dict) and "name" in tool:
            filtered_schema = filter_mcp_parameters(tool)
            filtered_schemas[tool["name"]] = filtered_schema
    
    return filtered_schemas 