#!/usr/bin/env python3

"""AgentCortex Service Configuration

Configuration for all AgentCortex services matching the original workflow.py setup.
This includes all the different service URLs from lenovo_workflow.yml.
"""

import os
from typing import Dict, Any
from pydantic import BaseModel


class AgentCortexConfig(BaseModel):
    """Configuration for AgentCortex services following original workflow.py pattern."""
    
    # Core service URLs (from lenovo_workflow.yml)
    intent_url: str
    session_memory_url: str
    system_memory_url: str
    planning_url: str
    summarization_url: str
    execution_url: str
    personalization_url: str
    extract_mentions_url: str
    
    # Pipeline settings
    max_iterations: int = 5
    
    # Test/mock settings
    mock_mode: bool = False
    test_data_path: str = ""
    test_output_dir: str = ""


def get_agentcortex_config() -> AgentCortexConfig:
    """Get AgentCortex configuration from environment variables.
    
    This matches the original lenovo_workflow.yml structure but uses
    environment variables for flexibility.
    """
    
    # Check if AgentCortex is enabled. Default to 'false' if not set.
    if os.getenv("AGENTCORTEX_ENABLED", "false").lower() != "true":
        # Return disabled config, ensuring all fields are present
        return AgentCortexConfig(
            intent_url="",
            session_memory_url="",
            system_memory_url="",
            planning_url="",
            summarization_url="",
            execution_url="",
            personalization_url="",
            extract_mentions_url="",
            max_iterations=5,
            mock_mode=True
        )
    
    # Default URLs from lenovo_workflow.yml (can be overridden by env vars)
    default_base_url = os.getenv("AGENTCORTEX_BASE_URL", "http://10.110.130.250")
    default_personalization_base = os.getenv("AGENTCORTEX_PERSONALIZATION_BASE", "http://10.110.131.30")
    
    return AgentCortexConfig(
        intent_url=os.getenv("AGENTCORTEX_INTENT_URL", f"{default_base_url}:22222"),
        session_memory_url=os.getenv("AGENTCORTEX_SESSION_MEMORY_URL", f"{default_base_url}:12306"),
        system_memory_url=os.getenv("AGENTCORTEX_SYSTEM_MEMORY_URL", f"{default_base_url}:12307"),
        planning_url=os.getenv("AGENTCORTEX_PLANNING_URL", f"{default_base_url}:11111"),
        summarization_url=os.getenv("AGENTCORTEX_SUMMARIZATION_URL", f"{default_base_url}:10087"),
        execution_url=os.getenv("AGENTCORTEX_EXECUTION_URL", f"{default_base_url}:15000"),
        personalization_url=os.getenv("AGENTCORTEX_PERSONALIZATION_URL", f"{default_personalization_base}:8889"),
        extract_mentions_url=os.getenv("AGENTCORTEX_EXTRACT_MENTIONS_URL", f"{default_personalization_base}:8890"),
        max_iterations=int(os.getenv("AGENTCORTEX_MAX_ITERATIONS", "5")),
        mock_mode=os.getenv("AGENTCORTEX_MOCK_MODE", "false").lower() == "true",
        test_data_path=os.getenv("AGENTCORTEX_TEST_DATA_PATH", ""),
        test_output_dir=os.getenv("AGENTCORTEX_TEST_OUTPUT_DIR", "")
    )


# Global configuration instance
agentcortex_config = get_agentcortex_config()


def is_agentcortex_enabled() -> bool:
    """Check if AgentCortex integration is enabled."""
    return (agentcortex_config.execution_url != "" and 
            agentcortex_config.planning_url != "" and
            not agentcortex_config.mock_mode)


def get_service_config() -> Dict[str, Any]:
    """Get service configuration dict compatible with original workflow.py."""
    return {
        "intent_url": agentcortex_config.intent_url,
        "session_memory_url": agentcortex_config.session_memory_url,
        "system_memory_url": agentcortex_config.system_memory_url,
        "planning_url": agentcortex_config.planning_url,
        "summarization_url": agentcortex_config.summarization_url,
        "execution_url": agentcortex_config.execution_url,
        "personalization_url": agentcortex_config.personalization_url,
        "extract_mentions_url": agentcortex_config.extract_mentions_url,
        "max_iterations": agentcortex_config.max_iterations
    } 