from __future__ import annotations

"""Adapters: expose MCP tools as Qwen-Agent tools.

Each MCP tool is wrapped into a `BaseTool` subclass and auto-registered via
`@register_tool`. The wrapper makes direct requests to the MCP executor service.
"""

import json
from typing import Any

from qwen_agent.tools.base import BaseTool, register_tool  # type: ignore

# Import our MCP client
from core.mcp_client import MCPClient, MCPConfig, get_mcp_tool_schemas


# Global MCP client - will be initialized when tools are registered
_mcp_client: MCPClient = None


def initialize_mcp_client(executor_url: str = None):
    """Initialize the global MCP client.
    
    Call this before registering tools to set up the MCP connection.
    """
    if executor_url is None:
        from config import mcp_config
        executor_url = mcp_config["executor_url"]
    global _mcp_client
    config = MCPConfig(executor_url=executor_url)
    _mcp_client = MCPClient(config)


def _make_mcp_tool_cls(name: str, spec: dict):
    """Return a BaseTool subclass that proxies to MCP service."""

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

    @register_tool(name)
    class _MCPTool(BaseTool):
        description = _desc
        parameters = _params

        def call(self, params: str, **kwargs):  # type: ignore[override]
            """Execute tool via MCP service."""
            global _mcp_client
            
            if _mcp_client is None:
                # Initialize with config URL if not already done
                initialize_mcp_client()
            
            # Parse parameters
            data = json.loads(params) if isinstance(params, str) else params
            
            # Extract query for context (if available)
            query = data.get("query", "")
            
            print(f"🔧 Tool wrapper executing: {name}")
            print(f"📤 Tool wrapper params: {data}")
            
            try:
                # Execute via MCP client
                result = _mcp_client.execute_tool(name, data, query)
                
                # Handle raw MCP response format
                if "observation" in result and "status" in result["observation"]:
                    status_list = result["observation"]["status"]
                    if status_list and len(status_list) > 0:
                        first_status = status_list[0]
                        
                        # Check if there's an error in the tool execution
                        if "error" in first_status:
                            error_info = first_status["error"]
                            error_msg = error_info.get("message", "Unknown error")
                            print(f"❌ MCP tool execution failed: {error_msg}")
                            return f"工具执行失败: {error_msg}"
                        elif "result" in first_status:
                            # Success case - return the complete raw result
                            tool_result = first_status["result"]
                            result_json = json.dumps(tool_result, ensure_ascii=False, indent=2)
                            print(f"✅ Tool wrapper returning raw result to Qwen Agent")
                            return result_json
                
                # Fallback for unexpected response format
                print(f"❌ Unexpected MCP response format: {list(result.keys())}")
                return f"工具执行失败: 响应格式异常"
                
            except Exception as e:
                error_msg = f"Tool {name} execution failed: {str(e)}"
                print(f"❌ {error_msg}")
                return f"工具执行出错: {error_msg}"



    _MCPTool.__name__ = f"MCPTool_{name}"
    return _MCPTool


def register_mcp_tools(executor_url: str = None):
    """Register all available MCP tools with Qwen Agent.
    
    Args:
        executor_url: URL of the MCP executor service
    """
    if executor_url is None:
        from config import mcp_config
        executor_url = mcp_config["executor_url"]
    
    # Initialize MCP client
    initialize_mcp_client(executor_url)
    
    # Get filtered tool schemas from MCP service
    tool_schemas = get_mcp_tool_schemas(_mcp_client)
    
    print(f"Registering {len(tool_schemas)} MCP tools with Qwen Agent:")
    
    # Generate Qwen tool classes for each MCP tool
    for tool_name, tool_spec in tool_schemas.items():
        print(f"  - {tool_name}: {tool_spec.get('description', 'No description')[:60]}...")
        _make_mcp_tool_cls(tool_name, tool_spec)
    
    print("MCP tools registration complete.")


# Auto-register tools on import (with fallback for development)
try:
    register_mcp_tools()  # Will use config automatically
except Exception as e:
    print(f"Warning: Could not register MCP tools: {e}")
    print("Falling back to dummy tools for development...")
    
    # Fallback to dummy tools for development when MCP is not available
    from tools import retail_tools as _d
    
    def _make_dummy_tool_cls(name: str, spec: dict):
        """Fallback: create dummy tool wrapper."""
        _desc: str = spec.get("description", name)
        params_schema: dict = spec.get("parameters", {})
        
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
        class _DummyTool(BaseTool):
            description = _desc
            parameters = _params
            
            def call(self, params: str, **kwargs):  # type: ignore[override]
                data = json.loads(params) if isinstance(params, str) else params
                try:
                    result = func(**data)
                except Exception as exc:
                    result = {"error": str(exc)}
                return json.dumps(result, ensure_ascii=False)
        
        _DummyTool.__name__ = f"DummyTool_{name}"
        return _DummyTool
    
    # Register dummy tools as fallback
    for _tool_name, _spec in _d.TOOLS_SCHEMA.items():
        _make_dummy_tool_cls(_tool_name, _spec) 