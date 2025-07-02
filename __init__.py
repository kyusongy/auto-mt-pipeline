"""auto_mt_pipeline: Reorganized APIGen-MT Pipeline

A clean, well-organized implementation of the APIGen-MT two-phase pipeline:
1. Blueprint generation & validation  
2. Trajectory collection via simulated conversations

This package reorganizes the original LLM-Application codebase with:
- Centralized configuration management
- Clear separation of concerns
- Template-based directory structure
"""

__version__ = "1.0.0"
__author__ = "Auto-reorganized from LLM-Application"

from .config import DEFAULT_LLM_CONFIG, DEFAULT_PIPELINE_CONFIG

__all__ = ["DEFAULT_LLM_CONFIG", "DEFAULT_PIPELINE_CONFIG"]
