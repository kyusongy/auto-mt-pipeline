"""Simple configuration using environment variables **or** inline defaults.

Instructions:
1. Preferred: set environment variables (they override everything)::
     export AUTO_MT_LLM_BASE_URL="your-llm-endpoint"
     export AUTO_MT_API_KEY="your-api-key"
     export AUTO_MT_MODEL="your-model-name"

2. Alternative: edit the DEFAULT_* values below. These are only used
   if the corresponding environment variable is **not** set.
"""

import os

# -----------------------------------------------------------------------------
# (1) Inline defaults – edit these if you don't want to use env variables
# -----------------------------------------------------------------------------
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"  # <- change me if needed
DEFAULT_LLM_MODEL = "gpt-4"                         # <- change me if needed
DEFAULT_API_KEY = None                               # <- put key here (not recommended)

# -----------------------------------------------------------------------------
# (2) LLM configuration – env vars win, otherwise fallback to defaults above
# -----------------------------------------------------------------------------
LLM_CONFIG = {
    "base_url": os.getenv("AUTO_MT_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    "model": os.getenv("AUTO_MT_MODEL", DEFAULT_LLM_MODEL),
    "api_key": os.getenv("AUTO_MT_API_KEY", DEFAULT_API_KEY),
}

# =============================================================================
# Generation Parameters
# =============================================================================
GENERATION_CONFIG = {
    # Blueprint generation (creativity for diverse scenarios)
    "blueprint_temperature": 1.0,
    "blueprint_max_tokens": 8192,
    
    # Trajectory collection - customer simulator (consistency for realistic behavior)
    "trajectory_temperature": 0.4,
    "trajectory_max_tokens": 4096,
    
    # Retail assistant agent (balance between helpful and deterministic)
    "assistant_temperature": 0.3,
    "assistant_max_tokens": 4096,
    
    # Request timeout for all LLM calls (seconds)
    "timeout": 120,
}

# =============================================================================
# Pipeline Settings
# =============================================================================
PIPELINE_CONFIG = {
    "max_blueprint_attempts": 5,  # Max retries for blueprint generation
    "bon_n": 3,                   # Best-of-N sampling for trajectory collection
    "debug": True,                # Enable debug output for development
}

# =============================================================================
# Validation
# =============================================================================
def validate_config():
    """Validate that required configuration values are present."""

    if not LLM_CONFIG["api_key"]:
        raise ValueError(
            "Missing LLM API key.\n"
            "Please either:\n"
            "  • set environment variable AUTO_MT_API_KEY, or\n"
            "  • edit DEFAULT_API_KEY in config/config.py"
        )

    if not LLM_CONFIG["base_url"]:
        raise ValueError(
            "Missing LLM base URL.\n"
            "Please either set AUTO_MT_LLM_BASE_URL or edit DEFAULT_LLM_BASE_URL."
        )

    # Uncomment for verbose confirmation
    # print("✅ Config OK →", {k: (v[:8] + '…' if k == 'api_key' and v else v) for k, v in LLM_CONFIG.items()})

    print("✅ Configuration validated successfully")
    print(f"   Using model: {LLM_CONFIG['model']}")
    print(f"   Service URL: {LLM_CONFIG['base_url']}")
    print(f"   API key: {LLM_CONFIG['api_key'][:10]}...")

# Auto-validate when imported
if __name__ != "__main__":
    validate_config() 