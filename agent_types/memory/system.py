#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ReadSystemMemoryRequest",
    "ReadSystemMemoryResponse",
    "ReadFewShotsRequest",
    "ReadFewShotsResponse",
    "ReadDomainKnowledgeRequest",
    "ReadDomainKnowledgeResponse",
    "ReadReflectionsRequest",
    "ReadReflectionsResponse",
    "WriteFewShotsRequest",
    "WriteFewShotsResponse",
    "WriteDomainKnowledgeRequest",
    "WriteDomainKnowledgeResponse",
    "WriteReflectionsRequest",
    "WriteReflectionsResponse",
    "ExtractReflectionRequest",
    "ExtractReflectionResponse",
]

from typing import List, Optional

from pydantic import ConfigDict, Field

from agent_types.common import FewShot, Request, Response, SessionMemory, SystemMemory, SystemProfile, UserMemory


class ReadSystemMemoryRequest(Request):
    """一次性读取SystemMemory"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "read_system_memory"

    query: str = Field(
        title="用户查询",
        description="检索示例所需的用户查询内容"
    )
    module_name: str = Field(
        title="模块名称",
        description="检索示例的模块名称"
    )
    n: int = Field(
        title="读取最大条数",
        description="读取最大条数"
    )
    threshold: Optional[float] = Field(
        title="相关性阈值",
        description=(
            "为了防止读取到无关的记忆信息而引入噪声，需要对读取到的n个系统记忆进行进一步过滤，"
            "这些记忆样本与query的相关性必须高于“相关性阈值”才能被最终返回。"
        ),
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="当前系统记忆对象",
        description="若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadSystemMemoryResponse(Response):
    """一次性读取SystemMemory"""

    model_config = ConfigDict(extra="allow")

    system_memory: Optional[SystemMemory] = Field(
        title="输出系统记忆对象",
        description="只有请求时指定了当前记忆，这里才会输出",
        default=None
    )


class ReadFewShotsRequest(Request):
    """读取示例请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "read_few_shots"

    query: str = Field(
        title="用户查询",
        description="检索示例所需的用户查询内容"
    )
    module_name: str = Field(
        title="模块名称",
        description="检索示例的模块名称"
    )
    n: int = Field(
        title="读取最大条数",
        description="读取最大条数"
    )
    threshold: Optional[float] = Field(
        title="相关性阈值",
        description=(
            "为了防止读取到无关的记忆信息而引入噪声，需要对读取到的n个系统记忆进行进一步过滤，"
            "这些记忆样本与query的相关性必须高于“相关性阈值”才能被最终返回。"
        ),
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="当前系统记忆对象",
        description="若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadFewShotsResponse(Response):
    """读取示例响应"""

    model_config = ConfigDict(extra="allow")

    few_shots: List[FewShot] = Field(
        title="示例信息",
        description="读取到的示例信息"
    )
    system_memory: Optional[SystemMemory] = Field(
        title="输出系统记忆对象",
        description="只有请求时指定了当前记忆，这里才会输出",
        default=None
    )


class ReadDomainKnowledgeRequest(Request):
    """读取领域知识请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "read_domain_knowledge"

    query: str = Field(
        title="用户查询",
        description="检索示例所需的用户查询内容"
    )
    module_name: str = Field(
        title="模块名称",
        description="检索示例的模块名称"
    )
    n: int = Field(
        title="读取最大条数",
        description="读取最大条数"
    )
    threshold: Optional[float] = Field(
        title="相关性阈值",
        description=(
            "为了防止读取到无关的记忆信息而引入噪声，需要对读取到的n个系统记忆进行进一步过滤，"
            "这些记忆样本与query的相关性必须高于“相关性阈值”才能被最终返回。"
        ),
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="当前系统记忆对象",
        description="若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadDomainKnowledgeResponse(Response):
    """读取领域知识响应"""

    model_config = ConfigDict(extra="allow")

    domain_knowledge: List[str] = Field(
        title="领域知识",
        description="读取到的领域知识信息"
    )
    system_memory: Optional[SystemMemory] = Field(
        title="输出系统记忆对象",
        description="只有请求时指定了当前记忆，这里才会输出",
        default=None
    )


class ReadReflectionsRequest(Request):
    """读取反思内容请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "read_reflections"

    query: str = Field(
        title="用户查询",
        description="检索示例所需的用户查询内容"
    )
    module_name: str = Field(
        title="模块名称",
        description="检索示例的模块名称"
    )
    n: int = Field(
        title="读取最大条数",
        description="读取最大条数"
    )
    threshold: Optional[float] = Field(
        title="相关性阈值",
        description=(
            "为了防止读取到无关的记忆信息而引入噪声，需要对读取到的n个系统记忆进行进一步过滤，"
            "这些记忆样本与query的相关性必须高于“相关性阈值”才能被最终返回。"
        ),
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="当前系统记忆对象",
        description="若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadReflectionsResponse(Response):
    """读取反思内容响应"""

    model_config = ConfigDict(extra="allow")

    reflections: List[str] = Field(
        title="反思内容",
        description="读取到的反思内容信息"
    )
    system_memory: Optional[SystemMemory] = Field(
        title="输出系统记忆对象",
        description="只有请求时指定了当前记忆，这里才会输出",
        default=None
    )


class WriteFewShotsRequest(Request):
    """写入示例请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "write_few_shots"

    module_name: str = Field(
        title="模块名称",
        description="写入示例的模块名称"
    )
    few_shots: List[FewShot] = Field(
        title="示例内容",
        description="写入的示例内容"
    )


class WriteFewShotsResponse(Response):
    """示例写入响应"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量", description="成功写入数量，可以不给出", default=None
    )


class WriteDomainKnowledgeRequest(Request):
    """写入领域知识请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "write_domain_knowledge"

    module_name: str = Field(
        title="模块名称",
        description="写入领域知识的模块名称"
    )
    domain_knowledge: List[str] = Field(
        title="领域知识内容",
        description="写入的领域知识内容"
    )


class WriteDomainKnowledgeResponse(Response):
    """写入领域知识响应"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class WriteReflectionsRequest(Request):
    """写入反思内容请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "write_reflections"

    module_name: str = Field(
        title="模块名称",
        description="写入反思的模块名称"
    )
    reflections: List[str] = Field(
        title="反思内容",
        description="写入的反思内容"
    )


class WriteReflectionsResponse(Response):
    """写入反思内容响应"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class ExtractReflectionRequest(Request):
    """反思请求对象"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "write_reflections"

    module_name: str = Field(
        title="模块名称",
        description="写入的模块名称"
    )
    input: str = Field(
        title="输入",
        description="反思对应的输入"
    )
    output: str = Field(
        title="输出",
        description="反思对应的输出"
    )
    feedback: str = Field(
        title="反馈",
        description="针对该输出输出的反馈信息"
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="得到对应输出的思考过程（如有）",
        default=None
    )
    tool_usage: Optional[str] = Field(
        title="工具调用",
        description="得到对应输出的工具使用情况（如有）",
        default=None
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="反思模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="反思模块对应的系统级记忆信息",
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


class ExtractReflectionResponse(Response):
    """反思响应对象"""

    model_config = ConfigDict(extra="allow")

    reflection: Optional[str] = Field(
        title="反思结果",
        description="对应反思的结果，如果请求时指定了集合名称，则可以不返回该项",
        default=None
    )
