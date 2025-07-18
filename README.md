# Auto MT Pipeline

A reorganized and cleaned implementation of the APIGen-MT blueprint generation and trajectory collection pipeline.

## Overview

This codebase has been reorganized from the original LLM-Application with the following improvements:

1. **Centralized Configuration**: All LLM configurations, sampling settings, and domain-specific parameters are now centralized in `config/llm_config.py`
2. **Clean Directory Structure**: Follows the mt_pipeline template organization pattern
3. **Separation of Concerns**: Clear separation between core functionality, tools, configuration, and utilities

## Directory Structure

```
auto_mt_pipeline/
├── config/                 # Centralized configuration
│   ├── config.yaml         # Main user configuration (YAML)
│   ├── defaults.py         # Advanced settings & domain data
│   ├── __init__.py         # Configuration loader
│   └── README.md           # Configuration guide
├── core/                   # Core functionality
│   ├── blueprint/          # Blueprint generation & validation
│   │   ├── pipeline.py
│   │   └── __init__.py
│   ├── trajectory/         # Trajectory collection & validation
│   │   ├── pipeline.py
│   │   ├── validator.py    # Trajectory validation
│   │   ├── qwen_tool_wrappers.py
│   │   └── __init__.py
│   ├── llm_client.py       # LLM client
│   ├── models.py           # Data models
│   └── __init__.py
├── tools/                  # Tool definitions and schemas
│   ├── retail_tools.py     # Retail domain tools
│   └── __init__.py
├── data/                   # Output directory (generated at runtime)
├── run_pipeline.py         # Main entry point
├── test_validation.py      # Test script for validation functionality
├── requirements.txt        # Dependencies
└── README.md              # This file
```



## Configuration

The project uses a **simple YAML-based configuration** similar to ML training frameworks. You only need to edit `config/config.yaml`:

```yaml
# Essential settings - just update these 3 fields!
llm:
  base_url: "http://127.0.0.1:12345/v1"  # Your LLM endpoint
  model: "qwen-32b"                       # Model name
  api_key: "tokenabc123"                  # API key

# Optional fine-tuning (advanced users)
generation:
  blueprint_temperature: 1.0   # Creativity for scenario generation
  trajectory_temperature: 0.3  # Customer simulator consistency
  assistant_temperature: 0.7   # Retail assistant behavior
```

All complex settings (retail domain rules, personas, sample data) are automatically handled. See `config/README.md` for advanced options.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your LLM endpoint in `config/config.yaml`

3. Run the pipeline:
   ```bash
   python run_pipeline.py
   ```

4. Check output files in the `data/` directory

### Testing Validation

To test the trajectory validation functionality:

```bash
python test_validation.py
```

This will create a test blueprint and trajectory, then validate them using the configured LLM.

## Features

- **Two-Stage Blueprint Generation**: 
  - Stage 1: Intent + Actions validation with original checks + review committee
  - Stage 2: Action execution validation with iterative refinement
- **Real Action Validation**: All blueprint actions are executed and validated before approval
- **Trajectory Validation**: Phase 2 includes LLM-based validation of collected conversations against the blueprint
- **Qwen Agent Integration**: Preserved original Qwen agent implementation for trajectory collection  
- **MCP Integration**: Uses MCP client for real tool execution during blueprint validation
- **Configurable**: Easy to adjust LLM settings, sampling parameters, and domain data
- **Debug Support**: Comprehensive logging and debug output options
- **Extensible**: Clean structure makes it easy to add new tools and prompts

## Trajectory Validation

The pipeline now includes comprehensive validation of collected trajectories in Phase 2:

- **Intent Expression**: Validates that the simulated human fully expresses their intent from the blueprint
- **Action Flow Adherence**: Ensures the conversation follows the planned actions and tool calls
- **Output Achievement**: Checks that expected outputs from the blueprint are achieved
- **Thought Process Alignment**: Verifies the conversation reflects the reasoning from Phase 1
- **Natural Conversation Flow**: Evaluates the realism and naturalness of the dialogue

Validation uses a separate LLM (configurable) to score trajectories on a 0-10 scale and approve/reject them based on quality thresholds.

