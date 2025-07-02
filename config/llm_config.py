"""Centralized configuration for auto_mt_pipeline.

This module centralizes all initialization parameters, LLM configurations,
sampling settings, and domain-specific data that were previously scattered
throughout the codebase.
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field, ConfigDict


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


# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

DEFAULT_LLM_CONFIG = LLMConfig(
    base_url="http://127.0.0.1:12345/v1",        # your OpenAI-compatible endpoint
    model="qwen-32b",                            # model name or path
    api_key="tokenabc123",                       # any string if server ignores auth
)

# ---------------------------------------------------------------------------
# Generation Options for Different Components
# ---------------------------------------------------------------------------

# Blueprint generation options (higher temperature for creativity)
BLUEPRINT_GENERATION_OPTIONS = GenerationOptions(
    temperature=1.0,
    max_tokens=4096,
    timeout=120
)

# Trajectory collection - agent options (lower temperature for consistency)
TRAJECTORY_AGENT_OPTIONS = GenerationOptions(
    temperature=0.3,
    max_tokens=4096,
    timeout=120,
    extra_body={"enable_reasoning": True}
)

# Trajectory collection - judge options (deterministic for scoring)
TRAJECTORY_JUDGE_OPTIONS = GenerationOptions(
    temperature=0.0,
    max_tokens=1024,
    timeout=60,
    extra_body={"enable_reasoning": True}
)

# Default generation options
DEFAULT_GENERATION_OPTIONS = GenerationOptions(
    temperature=0.7,
    max_tokens=8192,
    timeout=120
)

# ---------------------------------------------------------------------------
# Domain-Specific Configuration
# ---------------------------------------------------------------------------

DOMAIN_RULES = "Delivered orders cannot be canceled. Exchanges only allowed within 30 days."

PERSONAS = [
    "Budget-conscious student Emma",  # prefers deals, concise questions
    "Detail-oriented engineer Yusuf",  # wants everything in one go
    "Busy parent Carla",               # cares about speed and convenience
]

# Dummy user & order snapshots – used as prompt context for the blueprint LLM
SAMPLED_USER_DETAILS = """
- User ID: user_YRos_19122, Name: Yusuf Rossi, ZIP: 19122, History: Bought peripherals & smart devices.
- User ID: user_EClar_27513, Name: Emma Clark, ZIP: 27513, History: Bought textbooks & earbuds.
"""

SAMPLED_ORDERS = """
- Order ID: #W2378156, Status: Delivered, Items: [keyboard-id 1151293680, thermostat-id 4983901480]
- Order ID: #X9934712, Status: Shipped,   Items: [earbuds-id 3311224488]
"""

# ---------------------------------------------------------------------------
# Example Task Template
# ---------------------------------------------------------------------------

EXAMPLE_TASK = """
<thought>
The user has received order #W2378156 (keyboard & thermostat) and wants to
exchange both items for alternatives that better match their preferences.
To fulfil this we must:
  1) `find_user_id_by_name_zip` to get internal user ID;
  2) `get_order_details` to verify delivery status;
  3-4) `get_product_details` for each desired replacement item;
  5) `exchange_delivered_order_items` with correct mappings.
</thought>
<answer>
{
  "intent": "You are Yusuf Rossi in 19122. You received your order #W2378156 and wish to exchange the mechanical keyboard for a similar one but with clicky switches and the smart thermostat for one compatible with Google Home instead of Apple HomeKit. If there is no keyboard that is clicky, RGB backlight, full size, you'd go for no backlight. You are detail-oriented and want to make sure everything is addressed in one go.",
  "actions": [
    {
      "name": "find_user_id_by_name_zip",
      "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}
    },
    {
      "name": "get_order_details",
      "arguments": {"order_id": "#W2378156"}
    },
    {
      "name": "get_product_details",
      "arguments": {"product_id": "7706410293"}
    },
    {
      "name": "get_product_details",
      "arguments": {"product_id": "7747408585"}
    },
    {
      "name": "exchange_delivered_order_items",
      "arguments": {
        "order_id": "#W2378156",
        "item_ids": ["1151293680", "4983901480"],
        "new_item_ids": ["7706410293", "7747408585"],
        "payment_method_id": "credit_card_9513926"
      }
    }
  ],
  "outputs": [
    "Your exchange order is being processed."
  ]
}
</answer>
"""

# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

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


DEFAULT_PIPELINE_CONFIG = PipelineConfig(
    max_blueprint_attempts=5,
    bon_n=1,
    debug=True
)
