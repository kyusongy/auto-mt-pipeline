#!/usr/bin/env python3

"""Utility functions for AgentCortex integration.

Simple converters to bridge between existing pipeline structures 
and agentcortex-lsa Plan/ToolCalling formats.
"""

from typing import List

from core.models import ToolCalling as PipelineToolCalling
from agent_types.common import Plan, ToolCalling as AgentCortexToolCalling


def convert_pipeline_to_agentcortex_toolcalling(pipeline_tc: PipelineToolCalling) -> AgentCortexToolCalling:
    """Convert pipeline ToolCalling to agentcortex ToolCalling."""
    return AgentCortexToolCalling(
        name=pipeline_tc.name,
        arguments=pipeline_tc.arguments
    )


def convert_pipeline_actions_to_plan(actions: List[PipelineToolCalling], content: str = "") -> Plan:
    """Convert pipeline ToolCalling list to agentcortex Plan.
    
    Args:
        actions: List of pipeline ToolCalling objects
        content: Optional content for the plan
        
    Returns:
        Plan object that can be used with ExecutionService
    """
    agentcortex_tool_callings = [
        convert_pipeline_to_agentcortex_toolcalling(action) 
        for action in actions
    ]
    
    return Plan(
        tool_callings=agentcortex_tool_callings,
        content=content
    )


def convert_agentcortex_to_pipeline_toolcalling(agentcortex_tc: AgentCortexToolCalling) -> PipelineToolCalling:
    """Convert agentcortex ToolCalling to pipeline ToolCalling."""
    return PipelineToolCalling(
        name=agentcortex_tc.name,
        arguments=agentcortex_tc.arguments
    )


def convert_plan_to_pipeline_actions(plan: Plan) -> List[PipelineToolCalling]:
    """Convert agentcortex Plan to pipeline ToolCalling list."""
    return [
        convert_agentcortex_to_pipeline_toolcalling(tc)
        for tc in plan.tool_callings
    ] 