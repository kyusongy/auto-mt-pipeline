"""Configuration loader for auto_mt_pipeline.

This module provides a simple interface for accessing configuration.
Uses environment variables for sensitive data like API keys - much simpler than YAML parsing!
"""

# Standard library imports
import os
from pathlib import Path
from typing import Dict, Any

# Local application imports
from .defaults import (
    LLMConfig, PipelineConfig, GenerationOptions,
    get_blueprint_generation_options, get_blueprint_committee_options,
    get_trajectory_agent_options, get_assistant_agent_options, get_trajectory_judge_options,
    get_default_generation_options,
    DOMAIN_RULES, PERSONAS, SAMPLED_USER_DETAILS, SAMPLED_ORDERS, EXAMPLE_TASK
)
from .config import llm_config, generation_config, pipeline_config, mcp_config


class Config:
    """Configuration manager that provides easy access to all settings using environment variables."""
    
    def __init__(self):
        """Initialize configuration using environment variables and defaults."""
        # Configuration is loaded from config.py which uses environment variables
        # No need for complex YAML loading anymore!
        pass
    
    # =========================================================================
    # LLM Configuration
    # =========================================================================
    
    @property
    def llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return LLMConfig(
            base_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )
    
    # =========================================================================
    # Generation Options for Different Components
    # =========================================================================
    
    @property
    def blueprint_generation_options(self) -> GenerationOptions:
        """Get blueprint generation options with user overrides."""
        return get_blueprint_generation_options(generation_config)
    
    @property
    def blueprint_committee_options(self) -> GenerationOptions:
        """Get blueprint committee review options."""
        return get_blueprint_committee_options(generation_config)
    
    @property
    def trajectory_agent_options(self) -> GenerationOptions:
        """Get trajectory agent options with user overrides."""
        return get_trajectory_agent_options(generation_config)
    
    @property
    def assistant_agent_options(self) -> GenerationOptions:
        """Get retail assistant agent options with user overrides."""
        return get_assistant_agent_options(generation_config)
    
    @property
    def trajectory_judge_options(self) -> GenerationOptions:
        """Get trajectory judge options."""
        return get_trajectory_judge_options(generation_config)
    
    @property
    def default_generation_options(self) -> GenerationOptions:
        """Get default generation options."""
        return get_default_generation_options(generation_config)
    
    # =========================================================================
    # Pipeline Configuration
    # =========================================================================
    
    @property
    def pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        return PipelineConfig(
            max_blueprint_attempts=pipeline_config["max_blueprint_attempts"],
            bon_n=pipeline_config["bon_n"],
            debug=pipeline_config["debug"]
        )
    

    
    # =========================================================================
    # Convenience Properties for Legacy Compatibility
    # =========================================================================
    
    @property
    def DOMAIN_RULES(self) -> str:
        """Legacy compatibility - get domain rules."""
        return DOMAIN_RULES
    
    @property
    def PERSONAS(self) -> list:
        """Legacy compatibility - get personas."""
        return PERSONAS
    
    @property
    def SAMPLED_USER_DETAILS(self) -> str:
        """Legacy compatibility - get sample user details."""
        return SAMPLED_USER_DETAILS
    
    @property
    def SAMPLED_ORDERS(self) -> str:
        """Legacy compatibility - get sample orders."""
        return SAMPLED_ORDERS
    
    @property
    def EXAMPLE_TASK(self) -> str:
        """Legacy compatibility - get example task."""
        return EXAMPLE_TASK
    
    @property
    def DEFAULT_LLM_CONFIG(self) -> LLMConfig:
        """Legacy compatibility - get LLM config."""
        return self.llm_config
    
    @property
    def BLUEPRINT_GENERATION_OPTIONS(self) -> GenerationOptions:
        """Legacy compatibility - get blueprint generation options."""
        return self.blueprint_generation_options
    
    @property
    def BLUEPRINT_COMMITTEE_OPTIONS(self) -> GenerationOptions:
        """Legacy compatibility - get blueprint committee options."""
        return self.blueprint_committee_options
    
    @property
    def TRAJECTORY_AGENT_OPTIONS(self) -> GenerationOptions:
        """Legacy compatibility - get trajectory agent options."""
        return self.trajectory_agent_options
    
    @property
    def ASSISTANT_AGENT_OPTIONS(self) -> GenerationOptions:
        """Legacy compatibility - get assistant agent options."""
        return self.assistant_agent_options
    
    @property
    def TRAJECTORY_JUDGE_OPTIONS(self) -> GenerationOptions:
        """Legacy compatibility - get trajectory judge options."""
        return self.trajectory_judge_options
    
    @property
    def DEFAULT_GENERATION_OPTIONS(self) -> GenerationOptions:
        """Legacy compatibility - get default generation options."""
        return self.default_generation_options
    
    @property
    def DEFAULT_PIPELINE_CONFIG(self) -> PipelineConfig:
        """Legacy compatibility - get pipeline config."""
        return self.pipeline_config


# =============================================================================
# Global Configuration Instance
# =============================================================================

# Create a global config instance that can be imported by other modules
config = Config()

# Export everything for backward compatibility - using properties from the global config instance
DEFAULT_LLM_CONFIG = config.DEFAULT_LLM_CONFIG
BLUEPRINT_GENERATION_OPTIONS = config.BLUEPRINT_GENERATION_OPTIONS
BLUEPRINT_COMMITTEE_OPTIONS = config.BLUEPRINT_COMMITTEE_OPTIONS
TRAJECTORY_AGENT_OPTIONS = config.TRAJECTORY_AGENT_OPTIONS
ASSISTANT_AGENT_OPTIONS = config.ASSISTANT_AGENT_OPTIONS
TRAJECTORY_JUDGE_OPTIONS = config.TRAJECTORY_JUDGE_OPTIONS
DEFAULT_GENERATION_OPTIONS = config.DEFAULT_GENERATION_OPTIONS
DEFAULT_PIPELINE_CONFIG = config.DEFAULT_PIPELINE_CONFIG

__all__ = [
    'Config', 'config',
    # Legacy exports for backward compatibility
    "DEFAULT_LLM_CONFIG",
    "BLUEPRINT_GENERATION_OPTIONS",
    "BLUEPRINT_COMMITTEE_OPTIONS",
    "TRAJECTORY_AGENT_OPTIONS", 
    "ASSISTANT_AGENT_OPTIONS",
    "TRAJECTORY_JUDGE_OPTIONS",
    "DEFAULT_GENERATION_OPTIONS",
    "DOMAIN_RULES",
    "PERSONAS",
    "SAMPLED_USER_DETAILS",
    "SAMPLED_ORDERS",
    "EXAMPLE_TASK",
    "DEFAULT_PIPELINE_CONFIG",
    "LLMConfig",
    "PipelineConfig",
    "GenerationOptions",
]
