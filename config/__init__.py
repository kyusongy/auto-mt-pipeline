"""Configuration loader for auto_mt_pipeline.

This module provides a simple interface for loading and accessing configuration.
It automatically loads the user's config.yaml file and merges it with sensible defaults.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

from .defaults import (
    LLMConfig, PipelineConfig, GenerationOptions,
    get_blueprint_generation_options, get_blueprint_committee_options,
    get_trajectory_agent_options, get_assistant_agent_options, get_trajectory_judge_options,
    get_default_generation_options,
    DOMAIN_RULES, PERSONAS, SAMPLED_USER_DETAILS, SAMPLED_ORDERS, EXAMPLE_TASK
)


class Config:
    """Configuration manager that loads user config and provides easy access to all settings."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, looks for config.yaml 
                        in the same directory as this module.
        """
        if config_path is None:
            config_dir = Path(__file__).parent
            config_path = config_dir / "config.yaml"
        
        self._user_config = self._load_yaml_config(config_path)
        self._validate_config()
    
    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create a config.yaml file. See config.yaml for an example."
            )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ValueError(f"Configuration file is empty: {config_path}")
        
        return config
    
    def _validate_config(self):
        """Validate required configuration sections."""
        if "llm" not in self._user_config:
            raise ValueError("Missing required 'llm' section in config.yaml")
        
        llm_config = self._user_config["llm"]
        required_llm_fields = ["base_url", "model"]
        
        for field in required_llm_fields:
            if field not in llm_config:
                raise ValueError(f"Missing required LLM configuration field: {field}")
    
    # =========================================================================
    # LLM Configuration
    # =========================================================================
    
    @property
    def llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        llm_settings = self._user_config["llm"]
        return LLMConfig(
            base_url=llm_settings["base_url"],
            model=llm_settings["model"],
            api_key=llm_settings.get("api_key")
        )
    
    # =========================================================================
    # Generation Options for Different Components
    # =========================================================================
    
    @property
    def blueprint_generation_options(self) -> GenerationOptions:
        """Get blueprint generation options with user overrides."""
        return get_blueprint_generation_options(self._user_config)
    
    @property
    def blueprint_committee_options(self) -> GenerationOptions:
        """Get blueprint committee review options."""
        return get_blueprint_committee_options(self._user_config)
    
    @property
    def trajectory_agent_options(self) -> GenerationOptions:
        """Get trajectory agent options with user overrides."""
        return get_trajectory_agent_options(self._user_config)
    
    @property
    def assistant_agent_options(self) -> GenerationOptions:
        """Get retail assistant agent options with user overrides."""
        return get_assistant_agent_options(self._user_config)
    
    @property
    def trajectory_judge_options(self) -> GenerationOptions:
        """Get trajectory judge options."""
        return get_trajectory_judge_options(self._user_config)
    
    @property
    def default_generation_options(self) -> GenerationOptions:
        """Get default generation options."""
        return get_default_generation_options(self._user_config)
    
    # =========================================================================
    # Pipeline Configuration
    # =========================================================================
    
    @property
    def pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        pipeline_settings = self._user_config.get("pipeline", {})
        return PipelineConfig(
            max_blueprint_attempts=pipeline_settings.get("max_blueprint_attempts", 5),
            bon_n=pipeline_settings.get("bon_n", 3),
            debug=pipeline_settings.get("debug", False)
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
DOMAIN_RULES = config.DOMAIN_RULES
PERSONAS = config.PERSONAS
SAMPLED_USER_DETAILS = config.SAMPLED_USER_DETAILS
SAMPLED_ORDERS = config.SAMPLED_ORDERS
EXAMPLE_TASK = config.EXAMPLE_TASK
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
    "GenerationOptions",
    "PipelineConfig",
]
