#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "Request",
    "Response",
    "SystemProfile",
    "FewShot",
    "SystemMemory",
    "UserMemory",
    "ChatMessage",
    "Mention",
    "SessionMemory",
    "NDArray",
    "CSRArray",
    "Property",
    "Tool",
    "Intent",
    "ToolCalling",
    "Plan",
    "ExecutionError",
    "ExecutionStatus",
    "Observation",
    "LLMConfig",
    "GenerationOptions",
]

import base64
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.sparse import csr_array


class Request(BaseModel):
    """请求对象的基类
    所有接口的输入都继承自这个类

    以Request对象作为输入、以Response作为输出的函数称之为API函数。
    一个请求对象相当于一个API函数的入参的集合，即将函数所需要的的各个参数定义在了一个BaseModel对象中。
    这样做的优势包括：
    （1）能够以统一的形式调用所有的API函数。
    （2）BaseModel能够对函数执行前对各个参数进行类型检查，这对于系统的调试、维护，以及智能体的工具执行尤为重要。
    （3）便于自动生成API的接口描述信息，从而支持模块的即插即用特性。

    具体来讲，一个API函数通常定义为如下形式：

    def do_somthing(request: XxxRequest) -> XxxResponse:
        pass

    其中XxxRequest和XxxResponse定义为：

    class XxxRequest(Request):
        attribute1
        attribute2
        ...

    class XxxResponse(Response):
        attribute1
        attribute2
        ...
    """

    model_config = ConfigDict(extra="allow")

    trace_id: str = Field(
        title="该次请求的标识符",
        description=(
            "主要用于跟踪该次请求的执行路径，一般作为该次请求执行过程中涉及到的所有日志信息的标识。"
            "如果在创建请求对象的时候不指定该属性，则会自动生成一个UUID形式的字符串。"
        ),
        default_factory=lambda: str(uuid.uuid4())
    )

    __request_name__ = None

    @classmethod
    def get_request_path(cls) -> str:
        name = cls.__request_name__
        if name:
            if name.startswith("/"):
                return name
            else:
                return "/" + name
        else:
            name = cls.__name__
            if name.endswith("Request"):
                name = name[:-7]
            return "/" + cls._camel_to_snake(name)

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()


class Response(BaseModel):
    """响应对象的基类
    所有接口的输出都继承自这个类

    Response通常与Request成对定义，并同时使用。
    一个API函数可以返回一个或者多个Response对象，分别对应这个函数的“一次性输出模式”和“流式输出模式”。
    一次性输出模式和普通函数无异，流式输出模式通常有两种定义方式：
    （1）将函数定义为对应Response类型的生成器，例如：
        def do_something(request: XxxRequest) -> Iterable[XxxResponse]:
            ...
            yield XxxResponse()
            ...

    （2）函数仍然定义为普通函数，但其返回值是一个Response对象的生成器：
        def do_something(request: XxxRequest) -> Iterable[XxxResponse]:
            return (
                XxxResponse()
                for i in some_iterable
            )
    """

    model_config = ConfigDict(extra="allow")

    trace_id: Optional[str] = Field(
        title="被响应请求的标识符",
        description=(
            "该属性是对应请求对象中的trace_id属性。"
            "在构建响应对象时如果没有给出，一般也不自动生成，因为自动生成的ID无法和被响应请求的ID匹配，所以没有意义。"
        ),
        default=None
    )


class SystemProfile(BaseModel):
    """Agent或其子模块自身的画像信息"""

    model_config = ConfigDict(extra="allow")

    description: str = Field(
        title="角色描述信息",
        description=(
            "用于为通用模块指定角色信息，针对特定的业务进行更详细的功能描述。"
            "例如对于销售类智能体中的任务规划模块，可以通过角色描述设置该模块每次以“导购”的角色来完成任务。"
        )
    )
    language: Optional[str] = Field(
        title="语言",
        description="模块所使用的语言",
        default=None
    )
    capabilities: Optional[List[str]] = Field(
        title="能力范畴",
        description=(
            "该模块的能力范畴（如能完成什么任务），可分条描述。"
        ),
        default=None
    )
    constrains: Optional[List[str]] = Field(
        title="条件约束",
        description="该模块的约束条件（如立场原则、特殊规则），可分条描述",
        default=None
    )


class FewShot(BaseModel):
    """示例对象"""

    model_config = ConfigDict(extra="allow")

    input: str = Field(
        title="输入内容",
        description="示例对应的输入内容"
    )
    output: str = Field(
        title="输入内容",
        description="示例对应的输出内容"
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="得到对应输出的思考过程（如有）",
        default=None
    )


class SystemMemory(BaseModel):
    """系统级（System Level）记忆"""

    model_config = ConfigDict(extra="allow")

    few_shots: Optional[List[FewShot]] = Field(
        title="示例",
        description="针对当前任务的示例，可以是Agent设计者预先给出",
        default=None
    )
    domain_knowledge: Optional[List[str]] = Field(
        title="领域知识",
        description="针对当前任务的领域知识，主要由于Agent设计者预先给出",
        default=None
    )
    reflections: Optional[List[str]] = Field(
        title="反思信息",
        description="针对当前任务的反思信息，主要由反思模块根据Agent历史表现总结得到",
        default=None
    )


class UserMemory(BaseModel):
    """用户级（User Level）记忆"""

    model_config = ConfigDict(extra="allow")

    user_id: str = Field(
        title="用户标识",
        description="用户标识",
        default="default_user"
    )
    user_preference: Optional[Dict[str, str]] = Field(
        title="用户偏好",
        description="基于所有历史行为总结出的用户偏好信息",
        default=None
    )
    user_profile: Optional[Dict[str, str]] = Field(
        title="用户画像",
        description="基于所有历史行为总结出的用户画像信息",
        default=None
    )


class ChatMessage(BaseModel):
    """对话消息"""

    model_config = ConfigDict(extra="allow")

    content: str = Field(
        title="消息内容",
        description=(
            "整合了各种结构化信息（文本、链接以及表格等）的最终消息内容。"
            "这些结构化信息一般在metadata中给出。"
            "整合过程往往是业务相关的，可以构造专门用于整合的模块来完成。"
        )
    )
    metadata: Optional[Dict[str, Any]] = Field(
        title="消息元数据",
        description="该字典的key为对应数据项的名称，value必须是标准JSON类型。如果该属性为空，表示消息内容只以content为准。",
        default=None
    )
    role: str = Field(
        title="角色",
        description="发出该消息的角色"
    )
    turn_id: Optional[int] = Field(
        title="轮次标识",
        description="对应提及信息的轮次标识",
        default=None
    )
    thinking: Optional[str] = Field(
        title="思考过程",
        description="产生该对话消息时的思考内容（如有）",
        default=None
    )


class Mention(BaseModel):
    """用户提及信息"""

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        title="提及信息的类型",
        description="如“关键字”、“实体”、“解析后的实体”等"
    )
    name: str = Field(
        title="提及信息的名称",
        description="一般是与业务相关的提及信息描述，如“提及商品”、“提及人名”等，该信息可能会被某些算法作为LLM提示的一部分"
    )
    content: Union[str, List[str]] = Field(
        title="提及内容",
        description="用户提及的内容，例如用户感兴趣的话题、实体等"
    )
    role: str = Field(
        title="角色",
        description="必须明确指定角色",
    )
    turn_id: Optional[int] = Field(
        title="轮次标识",
        description="对应提及信息的轮次标识",
        default=None
    )


class SessionMemory(BaseModel):
    """会话级（Session Level）记忆"""

    model_config = ConfigDict(extra="allow")

    chat_history: List[ChatMessage] = Field(
        title="对话历史",
        description="当前会话的历史对话信息，默认为空列表",
        default_factory=list
    )
    mentions: Optional[List[Mention]] = Field(
        title="用户提及信息",
        description="当前会话中所有用户提及信息",
        default_factory=list
    )
    session_preference: Optional[Dict[str, str]] = Field(
        title="用户偏好",
        description="当前会话的用户偏好信息",
        default=None
    )

    def get_messages(self, role: Optional[str] = None) -> List[ChatMessage]:
        return [
            m for m in self.chat_history
            if m.role == role
        ] if role is not None else self.chat_history

    def get_mentions(
            self,
            role: Optional[str] = None,
            type_: Optional[str] = None,
            name: Optional[str] = None
    ) -> List[Mention]:
        return [
            m for m in self.mentions
            if (
                    (role is None or m.role == role) and
                    (type_ is None or m.type == type_) and
                    (name is None or m.name == name)
            )
        ]


class NDArray(BaseModel):
    """多维数组，可与numpy.ndarray相互转换"""

    data: str = Field(
        title="数组数据",
        description="数组数据（base64编码的二进制数据）"
    )
    dtype: str = Field(
        title="数据类型",
        description="数据类型"
    )
    shape: Tuple[int, ...] = Field(
        title="数组shape",
        description="数组shape（支持高维数组）"
    )

    @classmethod
    def from_array(cls, a: np.ndarray):
        bin_data = a.tobytes("C")
        return cls(
            data=base64.b64encode(bin_data).decode("utf-8"),
            dtype=str(a.dtype),
            shape=a.shape
        )

    def to_array(self) -> np.ndarray:
        bin_data = base64.b64decode(self.data)
        return np.frombuffer(
            buffer=bin_data,
            dtype=self.dtype
        ).reshape(self.shape)


class CSRArray(BaseModel):
    """稀疏数组，可与scipy.sparse.csr_array相互转换"""

    data: List[Union[int, float]] = Field(
        title="数组数据",
        description="数组数据（数据的列表）"
    )
    indices: List[int] = Field(
        title="索引序号",
        description="索引序号"
    )
    indptr: List[int] = Field(
        title="行数据范围",
        description="行数据范围"
    )
    dtype: str = Field(
        title="数据类型",
        description="数据类型"
    )
    shape: Tuple[int, int] = Field(
        title="数组shape",
        description="数组shape（仅支持二维数组）"
    )

    @classmethod
    def from_array(cls, a: csr_array):
        return cls(
            data=a.data,
            indices=a.indices,
            indptr=a.indptr,
            dtype=str(a.dtype),
            shape=a.shape
        )

    def to_array(self) -> csr_array:
        return csr_array(
            (self.data, self.indices, self.indptr),
            dtype=self.dtype,
            shape=self.shape
        )


class Property(BaseModel):
    """工具属性信息"""

    model_config = ConfigDict(extra="allow")

    description: Optional[str] = Field(
        title="属性描述",
        description="属性描述",
        default=None
    )
    type: Optional[str] = Field(
        title="数据类型",
        description="该属性的数据类型，通常为object, array, string, integer, float, bool或null",
        default="object"
    )
    anyOf: Optional[List["Property"]] = Field(
        title="Union类型",
        description="有多种类型可选，相当于python的Union类型",
        default=None
    )
    properties: Optional[Dict[str, "Property"]] = Field(
        title="子属性信息",
        description="如果属性类型为object，该项为其子属性信息",
        default=None
    )
    items: Optional["Property"] = Field(
        title="数组元素属性信息",
        description="如果属性类型为array，该项为其元素的属性信息",
        default=None
    )
    required: Optional[List[str]] = Field(
        title="子属性中的必要属性",
        description="子属性中的必要属性",
        default=None
    )


class Tool(BaseModel):
    """工具描述"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="工具名称",
        description="工具名称"
    )
    description: Optional[str] = Field(
        title="工具描述",
        description="工具的功能描述",
        default=None
    )
    input_schema: Optional[Property] = Field(
        title="工具参数信息",
        description="工具的参数信息",
        default=None
    )


class Intent(BaseModel):
    """候选意图描述"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="意图名称",
        description="意图名称，可以用一个词或短语表示，也可以是一句话"
    )
    description: Optional[str] = Field(
        title="意图描述",
        description="意图详细描述",
        default=None
    )


class ToolCalling(BaseModel):
    """工具调用信息"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="工具名称",
        description="被调用工具的名称"
    )
    arguments: Dict[str, Union[str, int, float, bool, Dict, List, "ToolCalling", None]] = Field(
        title="调用参数",
        description=(
            "参数名->参数值，参数值有3种情况："
            "为字符串时表示正常参数，需要执行模块转换成对应数据类型，例如json或python的eval()；"
            "为ToolCalling对象时候，表示依赖该工具调用输出的结果；"
            "为空表示不确定，需要执行模块动态解析"
        ),
        default_factory=dict
    )
    fallback: Optional[str] = Field(
        title="兜底信息",
        description="工具调用失败后返回的内容",
        default=None
    )


class Plan(BaseModel):
    """任务规划结果"""

    model_config = ConfigDict(extra="allow")

    tool_callings: List[ToolCalling] = Field(
        title="工具调用序列",
        description="完成相应任务的工具调用序列",
        default_factory=list
    )
    content: Optional[str] = Field(
        title="消息内容",
        description=(
            "该消息可以带有以下信息："
            "（1）需要用户澄清的问题，比如用户给出的任务描述不完整或者有歧义，可先不调用工具，先通过提问让用户在下一轮对话中给出相关信息，"
            "此时ToolCalling列表为空；"
            "（2）生成该计划（工具调用序列）的思考过程，当然也可以将思考过程放到PlanningResponse中返回。"
        ),
        default=None
    )


class ExecutionError(BaseModel):
    """执行错误信息对象"""

    model_config = ConfigDict(extra="allow")

    message: str = Field(
        title="错误信息",
        description="错误信息"
    )
    error: Optional[str] = Field(
        title="错误名称",
        description="通常是系统中捕获的异常的名称",
        default=None
    )
    traceback: Optional[str] = Field(
        title="详细信息",
        description="错误相关的详细信息，如函数调用栈信息等，用于定位错误",
        default=None
    )


class ExecutionStatus(BaseModel):
    """执行状态信息对象"""

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        title="工具名称",
        description="被执行工具的名称"
    )
    result: Optional[Any] = Field(
        title="执行结果",
        description="工具执行结果",
        default=None
    )
    error: Optional[ExecutionError] = Field(
        title="错误信息",
        description="工具执行错误信息，如果错误信息不为空，则执行结果无效",
        default=None
    )


class Observation(BaseModel):
    """观测信息对象

    观测信息对象的主要用途有以下几个：
    （1）用于给Summarization进行整合；
    （2）回传给一次性Planning模块进行重新规划；
    （3）回传给迭代式Planning模块进行下一步规划的生成；
    （4）可写进SessionMemory.chat_history用于保持上下文。
    """

    model_config = ConfigDict(extra="allow")

    plan: Plan = Field(
        title="规划方案",
        description="规划方案一般由Planning模块给出，进而被Execution模块执行"
    )
    status: List[ExecutionStatus] = Field(
        title="执行状态序列",
        description="列表中每一个元素对应Plan中的每一个ToolCalling对象",
        default_factory=list
    )


class LLMConfig(BaseModel):
    """大模型服务配置"""

    base_url: str = Field(
        title="服务地址",
        description="服务地址"
    )
    model: str = Field(
        title="模型名称",
        description="一个地址可能对应多个模型，需要指定模型名称"
    )
    api_key: Optional[str] = Field(
        title="API密钥",
        description="如果启动大模型服务的时候没有指定，这里就不需要",
        default=None
    )


class GenerationOptions(BaseModel):
    """LLM生成的相关选项"""

    model_config = ConfigDict(extra="allow")

    temperature: Optional[float] = Field(
        title="采样温度",
        description="如果为空或者0，表示不采样",
        default=None,
        ge=0.0
    )
    max_tokens: int = Field(
        title="最大Token数",
        description="该次生成过程中所使用的最大Token数量",
        default=2048
    )
    stop_token_ids: List[int] = Field(
        title="终止Token ID",
        description="只要生成到相应的Token就结束生成",
        default_factory=list
    )
