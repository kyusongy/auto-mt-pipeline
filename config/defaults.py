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

DOMAIN_RULES = """
- 你是一个联想商城的智能助手，具备判断是否需要调用外部工具来完成用户请求的能力。
- 政治相关、危险行为等敏感话题一定要拒绝回答，此时语气要和善且坚决。
- 当用户提问和联想以及联想商品相关时，尽量找合适的工具来获取信息，基于最新的信息来回答。
- 当工具返回"没有找到合适的推荐商品"时，礼貌建议用户调整需求
- 根据用户逐步提出的需求，使用相应工具提供帮助
"""

PERSONAS = [
    "预算敏感的大学生小李，关注性价比和学习需求",  # 关注价格和基本功能
    "注重品质的设计师王工，需要高性能工作设备",     # 专业需求，重视性能
    "忙碌的企业主陈总，寻求高效办公解决方案",       # 企业采购，重视效率
    "爱拍照的大学女生小芳，关注手机摄影和时尚外观",   # 手机与时尚
    "重视健身的白领张先生，需要便携平板和智能手表",   # 健康与可穿戴设备
    "技术极客刘工，关注最新高端旗舰笔记本和DIY升级空间", # 高端配置，可扩展性
    "省心的退休教师李阿姨，寻求易用的家用一体机与售后无忧", # 长者友好，售后服务
    "游戏玩家阿杰，想买高刷新率电竞显示器和外设",     # 电竞外设
    "高校信息化主管赵老师，批量采购计算机教室设备并关注教育优惠", # 批量采购，教育优惠
    "跨国远程办公的产品经理王女士，需要轻薄商务本和全球联保服务", # 远程办公，全球联保
]

SAMPLED_USER_DETAILS = """
"""

SAMPLED_ORDERS = """
"""

EXAMPLE_TASK = """
<thought>
用户想要咨询适合游戏和设计工作的台式机，预算在15000元左右，并希望对比不同产品。需要：
1) 根据用户需求推荐合适的产品 (product_recommend)
2) 用户可能需要对比推荐的商品，使用推荐结果中的SKU ID进行产品参数对比 (product_params_compare)
3) 用户可能对于某件商品具有疑问，查询相关产品知识补充信息 (product_knowledge_retrieval)
注意：product_params_compare需要使用product_recommend返回的实际SKU ID
</thought>
<answer>
{
  "intent": "你是一名注重产品品质的设计师，想买一台台式机，希望能流畅运行3D软件。同时你还很喜欢玩游戏，希望买到的电脑可以兼顾玩游戏的需求。你对联想的产品了解不多，预算大概15000元左右。你的性格较为内向，不善于表达，但是你很注重产品的品质和性能，希望买到性价比高的产品。",
  "actions": [
    {
      "name": "product_recommend", 
      "arguments": {"query": "台式机 3D设计软件 游戏 预算15000元 性价比", "category": ["台式机"], "request_brand_flag": false}
    },
    {
      "name": "product_params_compare",
      "arguments": {"product_ids_to_compare": ["316000162", "316000102"]}
    },
    {
      "name": "product_knowledge_retrieval",
      "arguments": {"query": "台式机 3D设计 游戏性能 配置推荐", "category": ["台式机"]}
    }
  ],
  "outputs": [
    "根据您的需求推荐适合3D设计和游戏的台式机产品，在预算范围内提供最佳性价比选择",
    "详细对比推荐产品的处理器、显卡、内存等关键配置参数，突出性能差异和优势",
    "提供3D设计软件兼容性和游戏性能基准测试结果，帮助您做出最佳选择"
  ]
}
</answer>

<thought>
用户是小企业主，需要为团队采购笔记本，并且需要了解企业优惠政策。这是一个多步骤场景：
1) 先推荐符合团队需求的商用笔记本 (product_recommend)
2) 比较推荐产品的具体参数差异 (product_params_compare，使用第1步返回的SKU ID)
3) 查询企业会员相关的采购优惠信息 (lenovo_member_retrieval)
4) 获取当前可用的企业优惠券 (get_user_coupon)
注意工具依赖关系：product_params_compare需要使用product_recommend的输出
</thought>
<answer>
{
  "intent": "你是一家刚成立两年的科技创业公司的CEO，公司主要做软件开发业务，目前团队有15个人。你需要为新招聘的5名开发工程师采购笔记本电脑，主要用于软件开发、代码编写、测试等工作。你对成本控制比较敏感，希望每台预算控制在8000-12000元之间。作为企业采购，你希望能享受到批量采购的优惠，同时也关注售后服务和质量保障。你之前主要购买其他品牌的设备，对联想的商用产品线不太熟悉，但听说联想在企业级服务方面做得不错。你性格比较务实，喜欢通过数据和具体对比来做决策。",
  "actions": [
    {
      "name": "product_recommend",
      "arguments": {"query": "商用笔记本 软件开发 编程 预算8000-12000元 企业采购", "category": ["笔记本"], "request_brand_flag": False}
    },
    {
      "name": "product_params_compare", 
      "arguments": {"product_ids_to_compare": ["317000001", "317000002"]}
    },
    {
      "name": "lenovo_member_retrieval",
      "arguments": {"query": "企业会员采购优惠政策 批量购买折扣"}
    },
    {
      "name": "get_user_coupon",
      "arguments": {}
    }
  ],
  "outputs": [
    "推荐适合软件开发的商用笔记本，配置满足编程需求且价格在预算范围内",
    "详细对比推荐产品的处理器性能、内存配置、扩展接口等关键参数，帮助您基于数据做出采购决策",
    "介绍企业会员的采购优惠政策，包括批量购买折扣、专属服务支持和售后保障",
    "展示您账户下可用的企业优惠券和限时活动，最大化采购性价比"
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


 