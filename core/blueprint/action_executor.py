"""Action execution for blueprint validation.

This module provides the ActionExecutor class that executes blueprint actions
through the MCP client to validate them and get real execution results.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.models import ToolCalling
from core.mcp_client import MCPClient


@dataclass
class ExecutionResult:
    """Container for tool execution results."""
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class ActionExecutionSummary:
    """Summary of all action executions."""
    results: List[ExecutionResult]
    total_actions: int
    successful_actions: int
    failed_actions: int
    dependency_chain: List[str]  # Order of execution


class ActionExecutor:
    """Executes blueprint actions using MCP client with dependency resolution."""
    
    def __init__(self, mcp_client: MCPClient, debug: bool = False):
        self.mcp_client = mcp_client
        self.debug = debug
        
    def execute_actions(self, actions: List[ToolCalling], user_intent: str = "") -> ActionExecutionSummary:
        """Execute a list of actions, resolving dependencies automatically.
        
        Args:
            actions: List of tool calls to execute
            user_intent: Original user intent for context
            
        Returns:
            ActionExecutionSummary with all execution results
        """
        if self.debug:
            print(f"🔧 ActionExecutor: Executing {len(actions)} actions")
            
        results: List[ExecutionResult] = []
        dependency_chain: List[str] = []
        execution_context: Dict[str, Any] = {}  # Store results for dependency resolution
        
        # Sort actions by dependencies (simple heuristic for now)
        sorted_actions = self._sort_actions_by_dependencies(actions)
        
        for action in sorted_actions:
            if self.debug:
                print(f"🛠️  Executing: {action.name}")
                
            # Resolve dependencies in arguments
            resolved_args = self._resolve_dependencies(action.arguments, execution_context)
            
            # Execute the action
            result = self._execute_single_action(action.name, resolved_args, user_intent)
            results.append(result)
            dependency_chain.append(action.name)
            
            # Store result for future dependency resolution
            if result.success:
                execution_context[action.name] = result.result
                if self.debug:
                    print(f"✅ {action.name} succeeded")
            else:
                if self.debug:
                    print(f"❌ {action.name} failed: {result.error}")
                    
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return ActionExecutionSummary(
            results=results,
            total_actions=len(actions),
            successful_actions=successful,
            failed_actions=failed,
            dependency_chain=dependency_chain
        )
    
    def _execute_single_action(self, tool_name: str, arguments: Dict[str, Any], query: str) -> ExecutionResult:
        """Execute a single tool action via MCP client."""
        import time
        
        start_time = time.time()
        
        try:
            # Execute via MCP client
            mcp_result = self.mcp_client.execute_tool(tool_name, arguments, query)
            execution_time = time.time() - start_time
            
            # Parse MCP response format
            if "observation" in mcp_result and "status" in mcp_result["observation"]:
                status_list = mcp_result["observation"]["status"]
                if status_list and len(status_list) > 0:
                    first_status = status_list[0]
                    
                    if "error" in first_status:
                        error_info = first_status["error"]
                        error_msg = error_info.get("message", "Unknown error")
                        return ExecutionResult(
                            tool_name=tool_name,
                            arguments=arguments,
                            success=False,
                            result=None,
                            error=error_msg,
                            execution_time=execution_time
                        )
                    elif "result" in first_status:
                        tool_result = first_status["result"]
                        return ExecutionResult(
                            tool_name=tool_name,
                            arguments=arguments,
                            success=True,
                            result=tool_result,
                            execution_time=execution_time
                        )
            
            # Fallback for unexpected format
            return ExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                result=None,
                error=f"Unexpected MCP response format: {list(mcp_result.keys())}",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                tool_name=tool_name,
                arguments=arguments,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
    
    def _sort_actions_by_dependencies(self, actions: List[ToolCalling]) -> List[ToolCalling]:
        """Sort actions to execute dependencies first.
        
        Simple heuristic: product_recommend before product_params_compare, etc.
        """
        # Define dependency order (earlier tools should execute first)
        dependency_order = {
            "product_recommend": 1,
            "product_knowledge_retrieval": 2,
            "product_params_compare": 3,  # Usually depends on product_recommend
            # Add more as needed
        }
        
        def get_priority(action: ToolCalling) -> int:
            return dependency_order.get(action.name, 999)  # Unknown tools last
            
        return sorted(actions, key=get_priority)
    
    def _resolve_dependencies(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve argument dependencies using previous execution results.
        
        For example, extract product IDs from product_recommend results for product_params_compare.
        """
        if not arguments:
            return {}
            
        resolved = arguments.copy()
        
        # Handle product_params_compare dependency on product_recommend
        if "product_ids_to_compare" in resolved:
            product_ids = resolved["product_ids_to_compare"]
            
            # If placeholder values, try to resolve from product_recommend results
            if isinstance(product_ids, list) and len(product_ids) > 0:
                if any(isinstance(pid, str) and ("placeholder" in pid.lower() or "sku" in pid.lower()) 
                       for pid in product_ids):
                    # Look for product_recommend results
                    if "product_recommend" in context:
                        recommend_result = context["product_recommend"]
                        extracted_ids = self._extract_product_ids(recommend_result)
                        if extracted_ids:
                            resolved["product_ids_to_compare"] = extracted_ids[:len(product_ids)]
                            if self.debug:
                                print(f"🔗 Resolved product_ids_to_compare: {extracted_ids[:len(product_ids)]}")
        
        return resolved
    
    def _extract_product_ids(self, product_result: Any) -> List[str]:
        """Extract product IDs/SKUs from product recommendation results."""
        product_ids = []
        
        if isinstance(product_result, dict):
            # Look for common product ID fields
            if "products" in product_result:
                products = product_result["products"]
                if isinstance(products, list):
                    for product in products:
                        if isinstance(product, dict):
                            # Try common ID field names
                            for id_field in ["sku", "id", "product_id", "productId", "SKU"]:
                                if id_field in product:
                                    product_ids.append(str(product[id_field]))
                                    break
            
            # Also look for direct ID lists
            for id_field in ["skus", "ids", "product_ids", "productIds"]:
                if id_field in product_result:
                    ids = product_result[id_field]
                    if isinstance(ids, list):
                        product_ids.extend([str(pid) for pid in ids])
        
        elif isinstance(product_result, list):
            # Direct list of products
            for item in product_result:
                if isinstance(item, dict):
                    for id_field in ["sku", "id", "product_id", "productId", "SKU"]:
                        if id_field in item:
                            product_ids.append(str(item[id_field]))
                            break
        
        return product_ids[:10]  # Limit to reasonable number 