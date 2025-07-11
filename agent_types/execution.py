#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ToolExecutingRequest",
    "ToolExecutingResponse",
    "ListToolsRequest",
    "ListToolsResponse",
]

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from agent_types.common import Intent, Observation, Plan, Request, Response, SessionMemory, SystemMemory, SystemProfile, \
    Tool, UserMemory


class ToolExecutingRequest(Request):
    """工具执行模块请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "execute_tools"

    plan: Plan = Field(
        title="规划方案",
        description="其中包含ToolCalling序列和对应的thinking"
    )
    task: Optional[str] = Field(
        title="原始任务描述",
        description="原始任务描述，可用于工具参数错误的时候对其进行纠正",
        default=None
    )
    intent: Optional[Intent] = Field(
        title="用户意图",
        description="用户意图",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="工具执行模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="工具执行模块对应的系统级记忆信息",
        default=None
    )
    user_memory: Optional[UserMemory] = Field(
        title="用户记忆对象",
        description="用户级记忆信息",
        default=None
    )
    session_memory: Optional[SessionMemory] = Field(
        title="会话记忆对象",
        description="会话级记忆信息",
        default=None
    )
    default_args: Optional[Dict[str, Any]] = Field(
        title="默认工具参数",
        description=(
            "将要调用的工具中的一部分参数可以不通过Planner给出，而是允许直接根据业务规则指定其取值。"
            "这里指定的默认参数值与工具参数定义中给定的默认值是有所区别的，此处的默认值来源于workflow的上下文，会随着智能体的执行发生变化。"
        ),
        default=None
    )


class ToolExecutingResponse(Response):
    """工具执行模块响应对象"""

    model_config = ConfigDict(extra="allow")

    observation: Observation = Field(
        title="观测信息",
        description=(
            "执行后所获得的观测信息，包含Plan对象，以及Plan对象中各工具的执行状态（结果或错误信息）序列，"
            "当然，如果Plan中工具调用列表为空，则对应执行状态序列也为空，"
            "这种情况主要出现在任务执行完成的时候，所以大多数时候也可用于判断迭代式任务规划是否已经结束"
        )
    )


class ListToolsRequest(Request):
    """列出所有工具"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "list_tools"


class ListToolsResponse(Response):
    """列出所有工具"""

    model_config = ConfigDict(extra="allow")

    tools: List[Tool] = Field(
        title="工具列表",
        description="所有支持MCP协议的工具列表，包括内部的和第三方的"
    )
