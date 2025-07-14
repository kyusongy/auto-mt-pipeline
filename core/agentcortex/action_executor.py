#!/usr/bin/env python3

"""Simple Action Executor for Blueprint Validation.

This provides a drop-in replacement for action validation that uses
the agentcortex ExecutionService but maintains compatibility with
the existing blueprint generation pipeline.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from core.models import ToolCalling as PipelineToolCalling
from core.agentcortex.execution_service import ExecutionService
from core.agentcortex.utils import convert_pipeline_actions_to_plan
from agent_types.execution import ToolExecutingRequest
from agent_types.common import SessionMemory


@dataclass
class ExecutionResult:
    """Container for tool execution results - compatible with blueprint pipeline."""
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class ActionExecutionSummary:
    """Summary of all action executions - compatible with blueprint pipeline."""
    results: List[ExecutionResult]
    total_actions: int
    successful_actions: int
    failed_actions: int
    dependency_chain: List[str]  # Order of execution


class AgentCortexActionExecutor:
    """Simple action executor using AgentCortex ExecutionService for blueprint validation."""
    
    def __init__(self, executor_url: str, tools_schema: Optional[Dict[str, Any]] = None):
        """Initialize with MCP executor URL and optional tool schemas."""
        self.execution_service = ExecutionService(executor_url, tools_schema=tools_schema)
        
    def execute_actions(
        self, 
        actions: List[PipelineToolCalling], 
        user_intent: str = ""
    ) -> ActionExecutionSummary:
        """Execute actions and return summary compatible with blueprint pipeline.
        
        Args:
            actions: List of pipeline ToolCalling objects to execute
            user_intent: Original user intent for context
            
        Returns:
            ActionExecutionSummary with execution results
        """
        import time
        
        try:
            # Create realistic default_args following agentcortex workflow.py pattern
            default_args = {
                "user_info": {
                    "uid": "13716255679", 
                    "user_identity": 1, 
                    "available_num": 0.0, 
                    "current_amount": "0", 
                    "enterprise_name": "", 
                    "future_expire_num": 0.0, 
                    "level_name": "", 
                    "entry_source": "shop", 
                    "user_province": ""
                },
                "trace_id": "blueprint_validation",
                "uid": "13716255679",
                "terminal": "1",
                "latitude": "23.89447712420573",
                "longitude": "106.6172117534938",
                "device_ip": "117.183.16.69",
                "get_position_permission": "agree",
                "event": "",
                "bind_mobile_id": 0,
                "query": user_intent,
                "chat_history": [],
                "mentions": []
            }
            
            # Create simple session memory
            session_memory = SessionMemory(
                chat_history=[],
                mentions=[]
            )
            
            # Convert pipeline actions to agentcortex Plan
            plan = convert_pipeline_actions_to_plan(actions, content="Blueprint validation execution")
            
            # Execute using ExecutionService
            execution_request = ToolExecutingRequest(
                plan=plan,
                task=user_intent,
                session_memory=session_memory,
                default_args=default_args
            )
            
            start_time = time.time()
            execution_response = self.execution_service.execute_tools(execution_request)
            total_time = time.time() - start_time
            observation = execution_response.observation
            
            # Convert to compatible ExecutionResult format
            results = []
            dependency_chain = []
            
            for i, (action, status) in enumerate(zip(actions, observation.status)):
                execution_time = total_time / len(actions) if len(actions) > 0 else 0
                
                if status.error:
                    result = ExecutionResult(
                        tool_name=action.name,
                        arguments=action.arguments,
                        success=False,
                        result=None,
                        error=status.error.message,
                        execution_time=execution_time
                    )
                else:
                    result = ExecutionResult(
                        tool_name=action.name,
                        arguments=action.arguments,
                        success=True,
                        result=status.result,
                        error=None,
                        execution_time=execution_time
                    )
                
                results.append(result)
                dependency_chain.append(action.name)
            
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            return ActionExecutionSummary(
                results=results,
                total_actions=len(actions),
                successful_actions=successful,
                failed_actions=failed,
                dependency_chain=dependency_chain
            )
            
        except Exception as e:
            # Return failed execution summary
            failed_results = [
                ExecutionResult(
                    tool_name=action.name,
                    arguments=action.arguments,
                    success=False,
                    result=None,
                    error=str(e),
                    execution_time=0.0
                ) for action in actions
            ]
            
            return ActionExecutionSummary(
                results=failed_results,
                total_actions=len(actions),
                successful_actions=0,
                failed_actions=len(actions),
                dependency_chain=[action.name for action in actions]
            )
    
    def validate_single_action(
        self, 
        action: PipelineToolCalling,
        query: str = ""
    ) -> Tuple[bool, str]:
        """Validate a single action by executing it.
        
        Args:
            action: Single ToolCalling to validate
            query: Original user query
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        success, results, error_message = self.execute_actions([action], query)
        return success, error_message 