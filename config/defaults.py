"""Default configurations for auto_mt_pipeline.

This module contains all the detailed retail configurations, domain rules, personas,
and sample data. Users typically don't need to modify this file - instead,
they should use the simple config.yaml for their settings.
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Pydantic Models for Configuration Validation
# =============================================================================

class GenerationOptions(BaseModel):
    """LLM generation configuration options."""

    model_config = ConfigDict(extra="allow")

    temperature: Optional[float] = Field(
        title="Sampling temperature",
        description="If None or 0, no sampling is used",
        default=None,
        ge=0.0
    )
    max_tokens: int = Field(
        title="Maximum tokens",
        description="Maximum number of tokens for this generation",
        default=8192
    )
    stream: bool = False
    presence_penalty: Optional[float] = None
    top_p: Optional[int] = None
    extra_body: Optional[Dict] = None
    timeout: int = Field(
        title="Request timeout",
        description="Timeout for LLM service requests (seconds)",
        default=120,
    )
    debug: bool = False  # if True, prints raw LLM replies for debugging


class LLMConfig(BaseModel):
    """LLM service configuration."""

    base_url: str = Field(
        title="Service URL",
        description="Base URL for the LLM service"
    )
    model: str = Field(
        title="Model name",
        description="Model name or path to use"
    )
    api_key: Optional[str] = Field(
        title="API key",
        description="API key for authentication (if required)",
        default=None
    )


class PipelineConfig(BaseModel):
    """Configuration for the complete pipeline."""
    
    max_blueprint_attempts: int = Field(
        default=5,
        description="Maximum attempts for blueprint generation"
    )
    bon_n: int = Field(
        default=1,
        description="Best-of-N sampling for trajectory collection"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug output"
    )


# =============================================================================
# Lenovo Domain Configuration
# =============================================================================

DOMAIN_RULES = """# 联想官网销售助手策略
作为联想官网的销售助手，您可以直接帮助用户进行产品咨询、购买建议、维修服务、设备管理、门店查询等服务。

## 产品线覆盖
### 台式机系列
- 扬天系列: 商务分体机，面向中小企业用户
- 天逸系列: 中端分体机，适合家庭和企业用户  
- 小新系列: 轻薄高性价比，面向学生和年轻白领
- YOGA系列: 高端一体机，注重设计和用户体验
- 拯救者系列: 游戏台式机，高性能硬件配置
- ThinkStation系列: 专业工作站，面向设计师、工程师
- ThinkCentre系列: 商用台式机，稳定性和扩展性
- GeekPro系列: 高性能需求用户，性价比高
- 来酷系列: 轻薄便携，面向学生和年轻白领

### 平板系列
- 拯救者系列: 游戏平板，高性能体验
- 小新系列: 入门级性价比平板
- YOGA系列: 大屏高端平板，创新设计
- 启天系列: 企业级商用平板
- 异能者系列: 移动办公平板，支持通信功能

## 服务原则
- 用户询问产品时，直接使用工具查询推荐，无需预先验证身份
- 当工具返回"没有找到合适的推荐商品"时，礼貌建议用户调整需求
- 根据用户逐步提出的需求，使用相应工具提供帮助
- 保持友好自然的对话，专注于产品推荐和信息查询
- 遇到复杂问题时转接人工客服"""

PERSONAS = [
    "预算敏感的大学生小李，关注性价比和学习需求",  # 关注价格和基本功能
    "注重品质的设计师王工，需要高性能工作设备",     # 专业需求，重视性能
    "忙碌的企业主陈总，寻求高效办公解决方案",       # 企业采购，重视效率
]

SAMPLED_USER_DETAILS = """
"""

SAMPLED_ORDERS = """
"""

EXAMPLE_TASK = """
<thought>
用户想要咨询适合游戏和设计工作的台式机，预算在15000元左右。需要：
1) 根据用户需求推荐合适的产品
2) 查询相关产品知识
3) 提供购买建议
注意：只有在product_recommend返回实际SKU ID后，才能使用product_params_compare进行对比
</thought>
<answer>
{
  "intent": "你是一名设计师，也喜欢玩游戏，想买一台台式机，预算大概15000元左右。希望能运行3D设计软件，也能流畅玩主流游戏。请推荐几款合适的产品。",
  "actions": [
    {
      "name": "product_recommend",
      "arguments": {"query": "设计师游戏台式机 预算15000元 3D设计软件 主流游戏", "category": ["台式机"]}
    },
    {
      "name": "product_knowledge_retrieval",
      "arguments": {"query": "拯救者台式机配置参数 设计软件兼容性"}
    }
  ],
  "outputs": [
    "根据您的需求，推荐拯救者刃系列台式机，配置满足3D设计和游戏需求",
    "提供详细的软件兼容性和性能测试信息",
    "基于产品推荐结果提供购买建议"
  ]
}
</answer>
"""


# =============================================================================
# Generation Defaults for Different Components
# =============================================================================

def get_blueprint_generation_options(user_config: dict) -> GenerationOptions:
    """Get blueprint generation options with user overrides."""
    return GenerationOptions(
        temperature=user_config.get("generation", {}).get("blueprint_temperature", 1.0),
        max_tokens=user_config.get("generation", {}).get("blueprint_max_tokens", 8192),
        timeout=user_config.get("generation", {}).get("timeout", 120)
    )


def get_blueprint_committee_options(user_config: dict) -> GenerationOptions:
    """Get blueprint committee review options (deterministic for scoring)."""
    return GenerationOptions(
        temperature=0.1,
        max_tokens=2048,
        timeout=user_config.get("generation", {}).get("timeout", 60),
        extra_body={"enable_reasoning": False}
    )


def get_trajectory_agent_options(user_config: dict) -> GenerationOptions:
    """Get trajectory agent options with user overrides."""
    return GenerationOptions(
        temperature=user_config.get("generation", {}).get("trajectory_temperature", 0.3),
        max_tokens=user_config.get("generation", {}).get("trajectory_max_tokens", 4096),
        timeout=user_config.get("generation", {}).get("timeout", 120),
        extra_body={"enable_reasoning": True}
    )


def get_assistant_agent_options(user_config: dict) -> GenerationOptions:
    """Get retail assistant agent options with user overrides."""
    return GenerationOptions(
        temperature=user_config.get("generation", {}).get("assistant_temperature", 0.7),
        max_tokens=user_config.get("generation", {}).get("assistant_max_tokens", 2048),
        timeout=user_config.get("generation", {}).get("timeout", 120),
        extra_body={"enable_reasoning": True}
    )


def get_trajectory_judge_options(user_config: dict) -> GenerationOptions:
    """Get trajectory judge options (deterministic for scoring)."""
    return GenerationOptions(
        temperature=0.0,
        max_tokens=2048,
        timeout=user_config.get("generation", {}).get("timeout", 60),
        extra_body={"enable_reasoning": False}
    )


def get_default_generation_options(user_config: dict) -> GenerationOptions:
    """Get default generation options."""
    return GenerationOptions(
        temperature=0.7,
        max_tokens=8192,
        timeout=user_config.get("generation", {}).get("timeout", 120)
    )


 