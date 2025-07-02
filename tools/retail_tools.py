from typing import Dict, Any, List

"""Dummy retail-domain APIs for prototyping APIGen-MT τ-bench case study.

Only the **tool schema** is required by the blueprint/trajectory pipelines –
actual Python callables are *not* executed in this demo.  Nevertheless, we
provide placeholder functions so you can easily plug in real logic later.
"""

# ---------------------------------------------------------------------------
# Tool schema (OpenAI / function-calling compliant)
# ---------------------------------------------------------------------------

TOOLS_SCHEMA: Dict[str, Dict[str, Any]] = {
    "find_user_id_by_name_zip": {
        "description": "Look up a user's unique ID given first name, last name, and ZIP code.",
        "parameters": {
            "first_name": {"type": "string", "required": True},
            "last_name": {"type": "string", "required": True},
            "zip": {"type": "string", "required": True},
        },
    },
    "get_order_details": {
        "description": "Retrieve order metadata and current status for an order ID.",
        "parameters": {
            "order_id": {"type": "string", "required": True},
        },
    },
    "get_product_details": {
        "description": "Return catalog information for a given product ID, including name, category, specs, price, and availability.",
        "parameters": {
            "product_id": {"type": "string", "required": True},
        },
    },
    "exchange_delivered_order_items": {
        "description": "Initiate an exchange on delivered items within an order, replacing them with new SKUs.",
        "parameters": {
            "order_id": {"type": "string", "required": True},
            "item_ids": {"type": "array", "items": {"type": "string"}, "required": True},
            "new_item_ids": {"type": "array", "items": {"type": "string"}, "required": True},
            "payment_method_id": {"type": "string", "required": True},
        },
    },
    "cancel_order": {
        "description": "Cancel an existing order if it has not been delivered.",
        "parameters": {
            "order_id": {"type": "string", "required": True},
            "reason": {"type": "string", "required": False},
        },
    },
    "track_order": {
        "description": "Track the shipping status of an order.",
        "parameters": {
            "order_id": {"type": "string", "required": True},
        },
    },
}


# ---------------------------------------------------------------------------
# Dummy runtime stubs (never executed by current prototype but handy for tests)
# ---------------------------------------------------------------------------

def _simple_stub(**kwargs):  # pragma: no cover – placeholder
    """Return a deterministic mapping of inputs ➜ outputs for quick testing."""
    return {"ok": True, "echo": kwargs}


def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str) -> str:  # type: ignore[override]
    return f"user_{first_name[:1]}{last_name[:3]}_{zip}"


def get_order_details(order_id: str):  # type: ignore[override]
    # Very naive mock – in reality you'd query a DB
    return {"order_id": order_id, "status": "Delivered" if order_id.endswith("6") else "Shipped"}


def get_product_details(product_id: str):  # type: ignore[override]
    return {"product_id": product_id, "name": f"Product-{product_id[-3:]}", "stock": 42}


def exchange_delivered_order_items(order_id: str, item_ids: List[str], new_item_ids: List[str], payment_method_id: str):  # type: ignore[override]
    return {"order_id": order_id, "exchanged": True, "new_items": new_item_ids}


def cancel_order(order_id: str, reason: str | None = None):  # type: ignore[override]
    status = get_order_details(order_id)["status"]
    if status == "Delivered":
        return {"ok": False, "error": "Delivered order cannot be canceled"}
    return {"ok": True, "canceled": True}


def track_order(order_id: str):  # type: ignore[override]
    return get_order_details(order_id)


# Optional: map names to callables for quick execution experiments
RUNTIME_FUNCTIONS = {
    name: globals().get(name, _simple_stub) for name in TOOLS_SCHEMA
} 