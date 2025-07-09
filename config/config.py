"""Simple configuration using environment variables **or** inline defaults.

Instructions:
1. Preferred: set environment variables (they override everything)::
     export AUTO_MT_LLM_BASE_URL="your-llm-endpoint"
     export AUTO_MT_API_KEY="your-api-key"
     export AUTO_MT_MODEL="your-model-name"

2. Alternative: edit the DEFAULT_* values below. These are only used
   if the corresponding environment variable is **not** set.
"""

# Standard library imports
import os

# -----------------------------------------------------------------------------
# (1) Inline defaults – edit these if you don't want to use env variables
# -----------------------------------------------------------------------------
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"  # <- change me if needed
DEFAULT_LLM_MODEL = "gpt-4"                         # <- change me if needed
DEFAULT_API_KEY = ""                                # <- put key here (not recommended)

# MCP Integration defaults
DEFAULT_MCP_EXECUTOR_URL = "http://10.110.130.250:15000"  # <- AgentCortex executor URL

# -----------------------------------------------------------------------------
# (2) LLM configuration – env vars win, otherwise fallback to defaults above
# -----------------------------------------------------------------------------
llm_config = {
    "base_url": os.getenv("AUTO_MT_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    "model": os.getenv("AUTO_MT_MODEL", DEFAULT_LLM_MODEL),
    "api_key": os.getenv("AUTO_MT_API_KEY", DEFAULT_API_KEY),
}

# =============================================================================
# MCP Configuration
# =============================================================================
mcp_config = {
    "executor_url": os.getenv("AUTO_MT_MCP_EXECUTOR_URL", DEFAULT_MCP_EXECUTOR_URL),
    "enabled": os.getenv("AUTO_MT_MCP_ENABLED", "true").lower() == "true",
}

# =============================================================================
# Generation Parameters
# =============================================================================
generation_config = {
    # Blueprint generation (creativity for diverse scenarios)
    "blueprint_temperature": 1.0,
    "blueprint_max_tokens": 8192,
    
    # Trajectory collection - customer simulator (consistency for realistic behavior)
    "trajectory_temperature": 0.3,
    "trajectory_max_tokens": 4096,
    
    # Retail assistant agent (balance between helpful and deterministic)
    "assistant_temperature": 0.3,
    "assistant_max_tokens": 4096,
    
    # General timeout setting
    "timeout": 120
}

# =============================================================================
# Pipeline Settings
# =============================================================================
pipeline_config = {
    "max_blueprint_attempts": 5,  # Max retries for blueprint generation
    "bon_n": 3,                   # Best-of-N sampling for trajectory collection
    "debug": True,                # Enable debug output for development
    
    # Blueprint generation settings (two-stage iterative process)
    "max_blueprint_iterations": 3,  # Max iterations per stage in blueprint generation
}

# =============================================================================
# Validation
# =============================================================================
def validate_config():
    """Validate that required configuration values are present."""

    if not llm_config["api_key"]:
        raise ValueError(
            "Missing LLM API key.\n"
            "Please either:\n"
            "  • set environment variable AUTO_MT_API_KEY, or\n"
            "  • edit DEFAULT_API_KEY in config/config.py"
        )

    if not llm_config["base_url"]:
        raise ValueError(
            "Missing LLM base URL.\n"
            "Please either set AUTO_MT_LLM_BASE_URL or edit DEFAULT_LLM_BASE_URL."
        )

    # Uncomment for verbose confirmation
    # print("✅ Config OK →", {k: (v[:8] + '…' if k == 'api_key' and v else v) for k, v in llm_config.items()})

    print("✅ Configuration validated successfully")
    print(f"   Using model: {llm_config['model']}")
    print(f"   Service URL: {llm_config['base_url']}")
    print(f"   API key: {llm_config['api_key'][:10]}...")
    
    if mcp_config["enabled"]:
        print(f"   MCP enabled: {mcp_config['executor_url']}")
    else:
        print("   MCP disabled - using dummy tools")

# Auto-validate when imported
if __name__ != "__main__":
    validate_config() 