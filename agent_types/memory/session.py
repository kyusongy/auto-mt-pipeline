#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ReadSessionMemoryRequest",
    "ReadSessionMemoryResponse",
    "ReadChatHistoryRequest",
    "ReadChatHistoryResponse",
    "ReadMentionsRequest",
    "ReadMentionsResponse",
    "ReadSessionPreferenceRequest",
    "ReadSessionPreferenceResponse",
    "WriteChatHistoryRequest",
    "WriteChatHistoryResponse",
    "WriteMentionsRequest",
    "WriteMentionsResponse",
    "WriteSessionPreferenceRequest",
    "WriteSessionPreferenceResponse",
    "ExtractMentionsRequest",
    "ExtractMentionsResponse",
]

from typing import Any, Dict, List, Optional, Union

from pydantic import ConfigDict, Field

from agent_types.common import ChatMessage, Mention, Request, Response, SessionMemory


class ReadSessionMemoryRequest(Request):
    """一次性读取SessionMemory"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "read_session_memory"

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    n: int = Field(
        title="最大对话轮数",
        description="最大对话轮数"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadSessionMemoryResponse(Response):
    """一次性读取SessionMemory"""

    model_config = ConfigDict(extra="allow")

    session_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class ReadChatHistoryRequest(Request):
    """对话历史读取请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "read_chat_history"

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    n: int = Field(
        title="最大对话轮数",
        description="最大对话轮数"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadChatHistoryResponse(Response):
    """对话历史读取响应"""

    model_config = ConfigDict(extra="allow")

    chat_history: List[ChatMessage] = Field(
        title="对话历史",
        description="读取到的对话历史信息"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class ReadMentionsRequest(Request):
    """提及信息读取请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "read_mentions"

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    n: int = Field(
        title="最大信息条数",
        description="最大信息条数"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadMentionsResponse(Response):
    """提及信息读取响应"""

    model_config = ConfigDict(extra="allow")

    mentions: List[Mention] = Field(
        title="用户提及信息",
        description="读取到的用户提及信息"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class ReadSessionPreferenceRequest(Request):
    """当前会话的用户偏好读取请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "read_session_preference"

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="源记忆对象",
        description="源记忆对象，若给定，读取后的信息将与该对象整合到一起",
        default=None
    )


class ReadSessionPreferenceResponse(Response):
    """当前会话的用户偏好读取响应"""

    model_config = ConfigDict(extra="allow")

    session_preference: Dict[str, str] = Field(
        title="会话用户偏好",
        description="读取到的会话用户偏好信息"
    )
    session_memory: Optional[SessionMemory] = Field(
        title="输出记忆对象",
        description="写入读取内容后的记忆对象，只有请求时指定了源记忆，这里才会输出",
        default=None
    )


class WriteChatHistoryRequest(Request):
    """写入对话历史请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "write_chat_history"

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    content: str = Field(
        title="消息内容",
        description="消息内容"
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
    thinking: Optional[str] = Field(
        title="思考过程",
        description="产生该对话消息时的思考内容（如有）",
        default=None
    )
    turn_id: Optional[int] = Field(
        title="轮次标识",
        description="不指定轮次表示最后一轮",
        default=None
    )


class WriteChatHistoryResponse(Response):
    """写入对话历史响应"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class WriteMentionsRequest(Request):
    """写入提及信息请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "write_mentions"

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    mentions: Union[Mention, List[Mention]] = Field(
        title="提及信息",
        description="当前轮次要写入的提及信息"
    )
    turn_id: Optional[int] = Field(
        title="轮次标识",
        description="不指定轮次表示最后一轮",
        default=None
    )


class WriteMentionsResponse(Response):
    """写入提及信息响应"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class WriteSessionPreferenceRequest(Request):
    """写入当前会话用户偏好"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "write_session_preference"

    session_id: Union[str, int] = Field(
        title="会话标识",
        description="会话标识"
    )
    session_preference: Dict[str, str] = Field(
        title="会话用户偏好",
        description="要写入的会话用户偏好"
    )
    turn_id: Optional[int] = Field(
        title="轮次标识",
        description="不指定轮次表示最后一轮",
        default=None
    )


class WriteSessionPreferenceResponse(Response):
    """写入当前会话用户偏好"""

    model_config = ConfigDict(extra="allow")

    num_written: Optional[int] = Field(
        title="成功写入数量",
        description="成功写入数量，可以不给出",
        default=None
    )


class ExtractMentionsRequest(Request):
    """提取提及信息
    仅为接口定义，不同的业务需要针对其业务逻辑各自实现
    """

    model_config = ConfigDict(extra="allow")
    __request_name__ = "extract_mentions"

    query: str = Field(
        title="用户查询",
        description="也可以是系统的回复"
    )
    role: str = Field(
        title="角色",
        description="必须明确指定角色",
    )


class ExtractMentionsResponse(Response):
    """提取提及信息"""

    model_config = ConfigDict(extra="allow")

    mentions: List[Mention] = Field(
        title="用户提及信息",
        description="列表中可以包含多种类型的提及信息，具体是哪些类型、名称是什么由具体的业务实现决定。",
        default_factory=list
    )
