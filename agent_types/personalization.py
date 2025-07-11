#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ExtractUserPreferenceRequest",
    "ExtractUserPreferenceResponse",
    "ExtractUserProfileRequest",
    "ExtractUserProfileResponse",
]

from typing import Dict, Optional

from pydantic import ConfigDict, Field

from agent_types.common import Request, Response, SessionMemory, SystemMemory, SystemProfile, UserMemory


class ExtractUserPreferenceRequest(Request):
    """用户偏好提取请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "extract_user_preference"

    query: str = Field(
        title="用户查询",
        description="用于分析用户偏好的查询内容"
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="用户偏好提取模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="用户偏好提取模块对应的系统级记忆信息",
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


class ExtractUserPreferenceResponse(Response):
    """用户偏好提取响应"""

    model_config = ConfigDict(extra="allow")

    user_preference: Dict[str, str] = Field(
        title="用户偏好信息",
        description="用户偏好信息，如果在请求时给定了collection，则可以不返回",
        default=None
    )


class ExtractUserProfileRequest(Request):
    """用户画像提取请求"""

    model_config = ConfigDict(extra="allow")
    __request_name__ = "extract_user_profile"

    query: str = Field(
        title="用户查询",
        description="用于分析用户画像的查询内容"
    )
    system_profile: Optional[SystemProfile] = Field(
        title="系统画像",
        description="用户画像提取模块对应的系统画像信息",
        default=None
    )
    system_memory: Optional[SystemMemory] = Field(
        title="系统记忆对象",
        description="用户画像提取模块对应的系统级记忆信息",
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


class ExtractUserProfileResponse(Response):
    """用户画像提取响应"""

    model_config = ConfigDict(extra="allow")

    user_profile: Optional[Dict[str, str]] = Field(
        title="用户画像信息",
        description="用户画像信息，如果在请求时给定了collection，则可以不返回",
        default=None
    )
