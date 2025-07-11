#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "PlanningRequest",
    "PlanningResponse",
]

from typing import Any, Dict, List, Optional, Union

from pydantic import ConfigDict, Field

from agent_types.common import Intent, Observation, Plan, Request, Response, SessionMemory, \
    SystemMemory, SystemProfile, Tool, UserMemory


class PlanningRequest(Request):
    """单次任务规划请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "plan"

    task: str = Field(
        title="任务描述",
        description="任务描述，可以式用户query或改写后的query"
    )
    tools: List[Tool] = Field(
        title="候选工具集合",
        description="完成该任务的候选工具，可以是所有工具，也可以是筛选后的",
        default_factory=list
    )
    builtin_tools: Optional[List[Tool]] = Field(
        title="内置工具集合",
        description="系统内置的通用工具集合，比如memory的读写功能可以被定义为内置工具，如果没有内置工具该列表为空",
        default=None
    )
    intent: Optional[Union[Intent, List[Intent]]] = Field(
        title="用户意图",
        description=(
            "可以由专用的意图理解模块传入，如果为空则表示由规划模块自行解析，"
            "允许多个意图对应一个planning，此时传入的将会是List[Intent]"
        ),
        default=None
    )
    observations: Optional[List[Observation]] = Field(
        title="观测信息列表",
        description=(
            "观测信息主要包含两方面内容：\n"
            "（1）针对该任务之前作出的Plan；\n"
            "（2）这些Plan的执行情况（即ExecutionStatus）。\n\n"
            "这些信息对于不同的Planning策略用法是不一样的：\n"
            "（1）对于One-time Planning来说，observations主要是为重新规划作参考，其中的每个observation代表一次尝试，而不是一个步骤；\n"
            "（2）对于Iterative Planning，每一个observation相当于一个步骤，在马尔科夫决策过程（MDP）视角下，其中的Plan相当于“行为”，"
            "而ExecutionStatus相当于“状态”。\n\n"
            "需要特别注意的是，如果是Iterative Planning，在迭代过程中（任务没有完成），这些中间过程（工具调用结果）是不会被写入SessionMemory的，"
            "因此对于Planning模块来讲，SessionMemory.chat_history包含的最近消息是上一次Agent给出的回复，即role为\"assistant\"的消息。"
        ),
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="任务规划模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="任务规划模块对应的系统级记忆信息",
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


class PlanningResponse(Response):
    """任务规划响应"""

    model_config = ConfigDict(extra="allow")

    plans: Union[Plan, List[Plan]] = Field(
        title="规划方案",
        description="完成该任务的规划，如返回多个Plan对象则表示其包含的子任务的规划",
    )
    finished: bool = Field(
        title="任务是否已完成",
        description=(
            "通常可以认为输出的ToolCalling列表为空就意味着任务已经规划完成，此时对应Plan中的content属性不为空，代表Planner对于任务的总结。"
            "但并不总是每次都需要Planner来总结，我们可能只希望得到工具调用结果；并且，为了与One-time式的Planning兼容，最好在PlanningResponse"
            "中就显式的包含finished这一项。"
        )
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="完成该任务的思考信息（如有）",
        default=None
    )
