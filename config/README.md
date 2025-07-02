# Configuration Guide

This project uses a simple YAML-based configuration system similar to ML training frameworks. You only need to modify a few key parameters in `config.yaml`.

## Quick Start

1. **Edit `config.yaml`** - This contains all the essential settings:
   ```yaml
   llm:
     base_url: "http://your-server:port/v1"  # Your LLM endpoint
     model: "your-model-name"                 # Model to use
     api_key: "your-api-key"                  # API key if needed
   ```

2. **Fine-tune generation** (optional):
   ```yaml
   generation:
     blueprint_temperature: 1.0    # Creativity for diverse scenarios
     trajectory_temperature: 0.3   # Customer simulator consistency
     assistant_temperature: 0.7    # Retail assistant creativity
   ```

3. **Adjust pipeline settings** (optional):
   ```yaml
   pipeline:
     bon_n: 3                     # Best-of-N sampling
     max_blueprint_attempts: 5    # Max retries
     debug: true                  # Enable debug output
   ```

## Examples

### High Creativity Setup
```yaml
generation:
  blueprint_temperature: 1.5    # More creative blueprint generation
  trajectory_temperature: 0.5   # More varied customer behavior
  assistant_temperature: 0.9    # More creative retail assistant responses
```

### Production Setup
```yaml
generation:
  timeout: 60                   # Faster timeouts
pipeline:
  debug: false                  # Clean logs
  bon_n: 1                      # Faster execution
  assistant_temperature: 0.5    # More deterministic assistant
```

## Agent Configuration Guide

The pipeline uses multiple AI agents, each configurable:

- **Blueprint Generator** (`blueprint_temperature`): Creates diverse task scenarios (high temp = more creative scenarios)
- **Customer Simulator** (`trajectory_temperature`): Simulates realistic human behavior (low temp = more consistent customers)  
- **Retail Assistant** (`assistant_temperature`): The actual helpdesk agent being trained (medium temp = helpful but reliable)
- **Review Committee** & **Judges**: Fixed settings for consistent evaluation

**Typical Settings:**
- **Research/Diversity**: High temperatures (0.8-1.5) for creative scenarios
- **Production Training**: Medium temperatures (0.3-0.7) for realistic but consistent behavior
- **Evaluation**: Low temperatures (0.0-0.3) for deterministic scoring

## Advanced Configuration

All complex settings (retail domain rules, personas, sample data) are automatically handled in `defaults.py`. Most users won't need to modify this file.

The configuration system maintains full backward compatibility - existing code continues to work unchanged. 