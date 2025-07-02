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
# Retail Domain Configuration
# =============================================================================

DOMAIN_RULES = """# Retail agent policy
As a retail agent, you can help users cancel or modify pending orders, return or exchange delivered orders, modify their default user address, or provide information about their own profile, orders, and related products.
- At the beginning of the conversation, you have to authenticate the user identity by locating their user id via email, or via name + zip code. This has to be done even when the user already provides the user id.
- Once the user has been authenticated, you can provide the user with information about order, product, profile information, e.g. help the user look up order id.
- You can only help one user per conversation (but you can handle multiple requests from the same user), and must deny any requests for tasks related to any other user.
- Before taking consequential actions that update the database (cancel, modify, return, exchange), you have to list the action detail and obtain explicit user confirmation (yes) to proceed.
- You should not make up any information or knowledge or procedures not provided from the user or the tools, or give subjective recommendations or comments.
- You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call.
- You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.

## Domain basic
- All times in the database are EST and 24 hour based. For example "02:30:00" means 2:30 AM EST.
- Each user has a profile of its email, default address, user id, and payment methods. Each payment method is either a gift card, a paypal account, or a credit card.
- Our retail store has 50 types of products. For each type of product, there are variant items of different options. For example, for a 't shirt' product, there could be an item with option 'color blue size M', and another item with option 'color red size L'.
- Each product has an unique product id, and each item has an unique item id. They have no relations and should not be confused.
- Each order can be in status 'pending', 'processed', 'delivered', or 'cancelled'. Generally, you can only take action on pending or delivered orders.
- Exchange or modify order tools can only be called once. Be sure that all items to be changed are collected into a list before making the tool call!!!

## Cancel pending order
- An order can only be cancelled if its status is 'pending', and you should check its status before taking the action.
- The user needs to confirm the order id and the reason (either 'no longer needed' or 'ordered by mistake') for cancellation.
- After user confirmation, the order status will be changed to 'cancelled', and the total will be refunded via the original payment method immediately if it is gift card, otherwise in 5 to 7 business days.

## Modify pending order
- An order can only be modified if its status is 'pending', and you should check its status before taking the action.
- For a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.

### Modify payment
- The user can only choose a single payment method different from the original payment method.
- If the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.
- After user confirmation, the order status will be kept 'pending'. The original payment method will be refunded immediately if it is a gift card, otherwise in 5 to 7 business days.

### Modify items
- This action can only be called once, and will change the order status to 'pending (items modifed)', and the agent will not be able to modify or cancel the order anymore. So confirm all the details are right and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all items to be modified.
- For a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.
- The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

## Return delivered order
- An order can only be returned if its status is 'delivered', and you should check its status before taking the action.
- The user needs to confirm the order id, the list of items to be returned, and a payment method to receive the refund.
- The refund must either go to the original payment method, or an existing gift card.
- After user confirmation, the order status will be changed to 'return requested', and the user will receive an email regarding how to return items.

## Exchange delivered order
- An order can only be exchanged if its status is 'delivered', and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.
- For a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.
- The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.
- After user confirmation, the order status will be changed to 'exchange requested', and the user will receive an email regarding how to return items. There is no need to place a new order."""

PERSONAS = [
    "Budget-conscious student Emma",  # prefers deals, concise questions
    "Detail-oriented engineer Yusuf",  # wants everything in one go
    "Busy parent Carla",               # cares about speed and convenience
]

SAMPLED_USER_DETAILS = """
- User ID: user_YRos_19122, Name: Yusuf Rossi, ZIP: 19122, History: Bought peripherals & smart devices.
- User ID: user_EClar_27513, Name: Emma Clark, ZIP: 27513, History: Bought textbooks & earbuds.
"""

SAMPLED_ORDERS = """
- Order ID: #W2378156, Status: Delivered, Items: [keyboard-id 1151293680, thermostat-id 4983901480]
- Order ID: #X9934712, Status: Shipped,   Items: [earbuds-id 3311224488]
"""

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


 