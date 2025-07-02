# Auto MT Pipeline

A reorganized implementation of the APIGen-MT blueprint generation and trajectory collection pipeline.

## Overview

This codebase has been reorganized from the original LLM-Application with the following improvements:

1. **Centralized Configuration**: All LLM configurations, sampling settings, and domain-specific parameters are now centralized in `config/llm_config.py`
2. **Clean Directory Structure**: Follows the mt_pipeline template organization pattern
3. **Separation of Concerns**: Clear separation between core functionality, tools, configuration, and examples
4. **Preserved Logic**: All original code logic, especially Qwen agent implementation, remains unchanged

## Directory Structure

```
auto_mt_pipeline/
├── config/                 # Centralized configuration
│   ├── llm_config.py       # All LLM and pipeline settings
│   └── __init__.py
├── core/                   # Core functionality
│   ├── blueprint/          # Blueprint generation & validation
│   ├── trajectory/         # Trajectory collection
│   ├── llm_client.py       # LLM client
│   ├── models.py           # Data models
│   └── __init__.py
├── tools/                  # Tool definitions and schemas
│   ├── retail_tools.py     # Retail domain tools
│   └── __init__.py
├── examples/               # Example data and outputs
├── utils/                  # Utility functions
├── prompts/               # Prompt templates
├── data/                  # Output directory
├── workspace/             # Working directory
├── run_pipeline.py        # Main entry point
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Configuration

All configuration is centralized in `config/llm_config.py`:

- **LLM Settings**: endpoint URL, model name, API key
- **Generation Options**: different settings for blueprint generation, trajectory collection, etc.
- **Domain Data**: business rules, personas, sample data
- **Pipeline Settings**: debug flags, retry limits, etc.

To configure your LLM endpoint, edit `config/llm_config.py`:

```python
DEFAULT_LLM_CONFIG = LLMConfig(
    base_url="http://127.0.0.1:12345/v1",  # ← your endpoint
    model="qwen-32b",                      # ← your model
    api_key="tokenabc123",                 # ← your API key
)
```

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your LLM endpoint in `config/llm_config.py`

3. Run the pipeline:
   ```bash
   python run_pipeline.py
   ```

4. Check output files in the `data/` directory

## Features

- **Two-Phase Pipeline**: Blueprint generation → Trajectory collection
- **Qwen Agent Integration**: Preserved original Qwen agent implementation
- **Configurable**: Easy to adjust LLM settings, sampling parameters, and domain data
- **Debug Support**: Comprehensive logging and debug output options
- **Extensible**: Clean structure makes it easy to add new tools and prompts

## Migration from Original

This reorganized version maintains 100% compatibility with the original LLM-Application functionality while providing:

- Better organization and maintainability
- Centralized configuration management
- Cleaner import structure
- Template-based directory layout

All original algorithms, especially the Qwen agent implementation, remain unchanged.
