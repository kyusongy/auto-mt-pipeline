#!/usr/bin/env python3
"""
Auto MT Pipeline - Reorganized APIGen-MT Implementation

A clean demonstration of the APIGen-MT two-phase pipeline:
1. Blueprint generation & validation
2. Trajectory collection via simulated conversations

This reorganized version centralizes all configuration and follows
a clean directory structure based on the mt_pipeline template.

Usage:
    python run_pipeline.py

Configure your LLM endpoint in config/llm_config.py.
"""

# Standard library imports
import json
from pathlib import Path

# Local application imports
from config import (
    DEFAULT_LLM_CONFIG,
    BLUEPRINT_GENERATION_OPTIONS,
    DOMAIN_RULES,
    PERSONAS,
    SAMPLED_USER_DETAILS,
    SAMPLED_ORDERS,
    EXAMPLE_TASK,
    DEFAULT_PIPELINE_CONFIG,
    mcp_config,
)
from core.blueprint.pipeline import generate_valid_blueprint
from core.trajectory.pipeline import TrajectoryCollector
from core.mcp_client import MCPClient, MCPConfig, get_mcp_tool_schemas
# Remove unused imports

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use centralized configuration
llm_config = DEFAULT_LLM_CONFIG
pipeline_config = DEFAULT_PIPELINE_CONFIG

# Output configuration
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

# Enable debug output from blueprint generation
import core.blueprint.pipeline as bp_pipeline
bp_pipeline.GenerationOptions = lambda **kwargs: BLUEPRINT_GENERATION_OPTIONS.model_copy(update=kwargs)

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def get_tool_schemas():
    """Get tool schemas - try MCP first, fallback to dummy tools."""
    if not mcp_config["enabled"]:
        print("🔌 MCP integration disabled - using dummy tools")
        from tools.retail_tools import TOOLS_SCHEMA
        return TOOLS_SCHEMA, None
        
    try:
        executor_url = mcp_config["executor_url"]
        print(f"🔗 Connecting to MCP executor at {executor_url}")
        mcp_client_config = MCPConfig(executor_url=executor_url)
        mcp_client = MCPClient(mcp_client_config)
        
        # Get filtered schemas (without MCP-injected parameters)
        schemas = get_mcp_tool_schemas(mcp_client)
        
        if schemas:
            print(f"✅ Loaded {len(schemas)} tools from MCP service:")
            for name, schema in schemas.items():
                print(f"   - {name}: {schema.get('description', 'No description')[:60]}...")
            return schemas, mcp_client
        else:
            raise Exception("No tools returned from MCP service")
            
    except Exception as e:
        print(f"⚠️  MCP connection failed: {e}")
        print("📦 Falling back to dummy retail tools...")
        
        # Fallback to dummy tools
        from tools.retail_tools import TOOLS_SCHEMA
        return TOOLS_SCHEMA, None


def main():
    """Run the complete MT pipeline."""
    print("="*60)
    print("🚀 Auto MT Pipeline - MCP Integration")
    print("="*60)
    
    # Get tool schemas (MCP or fallback)
    tools_schema, mcp_client = get_tool_schemas()

    # -------------------------------------------------------------------
    # Load cached API dependency edges (built separately)
    # -------------------------------------------------------------------

    edges_file = output_dir / "api_edges.json"

    api_edges_json = edges_file.read_text(encoding="utf-8")
 
    # -------------------------------------------------------------------
    
    # Phase 1: Generate validated blueprint
    print("\n📋 Phase 1: Blueprint Generation & Validation")
    print("-" * 40)
    
    blueprint = generate_valid_blueprint(
        llm_config,
        tools_schema,
        PERSONAS,
        max_attempts=pipeline_config.max_blueprint_attempts,
        prompt_kwargs={
            "domain_rules": DOMAIN_RULES,
            "sampled_user_details": SAMPLED_USER_DETAILS,
            "sampled_orders": SAMPLED_ORDERS,
            "examples": EXAMPLE_TASK,
            "task_rules": "",
            "api_dependencies": api_edges_json,
        },
    )

    print("\n📝 Generated Blueprint:")
    print(f"  Intent: {blueprint.user_intent}")
    print(f"  Actions: {len(blueprint.actions)} tool calls")
    print(f"  Expected outputs: {len(blueprint.expected_outputs)} items")
    
    # Show the blueprint structure
    blueprint_data = {
        "intent": blueprint.user_intent,
        "actions": [a.model_dump() for a in blueprint.actions],
        "outputs": blueprint.expected_outputs,
    }
    
    print("\n📋 Blueprint JSON:")
    print(json.dumps(blueprint_data, indent=2, ensure_ascii=False))
    
    # Save blueprint for inspection
    blueprint_file = output_dir / "blueprint.json"
    blueprint_file.write_text(json.dumps(blueprint_data, indent=2, ensure_ascii=False))
    print(f"\n💾 Blueprint saved to: {blueprint_file}")

    # Phase 2: Collect trajectory
    print("\n💬 Phase 2: Trajectory Collection")
    print("-" * 40)
    
    if not blueprint.actions or blueprint.user_intent.startswith("[PARSING"):
        print("\n❌ Blueprint generation failed – skipping trajectory collection.")
        return False

    # Configure trajectory collector with MCP if available
    if mcp_client:
        print("🔗 Using MCP tools for trajectory collection")
        # Update Qwen tool wrappers to use our MCP client
        from core.trajectory.qwen_tool_wrappers import initialize_mcp_client
        initialize_mcp_client()  # Will use config automatically
    else:
        print("📦 Using dummy tools for trajectory collection")

    collector = TrajectoryCollector(
        llm_config,  # human_cfg
        llm_config,  # agent_cfg
        tools_schema=tools_schema,
        debug=pipeline_config.debug,
        bon_n=pipeline_config.bon_n,  # Enable best-of-N sampling
    )
    
    trajectory = collector.collect(blueprint)

    if trajectory:
        print(f"\n✅ Successfully collected trajectory with {len(trajectory.turns)} turns")
        print("\n📞 Conversation:")
        for t in trajectory.turns:
            if t.role == "user":
                tag = "USER"
            elif t.role == "assistant":
                tag = "ASSISTANT"
            else:
                tag = t.role.upper()
            # Only show non-system messages for readability
            if t.role != "system":
                print(f"[{tag}] {t.content}")
        
        # Save complete trajectory  
        trajectory_data = {
            "blueprint": blueprint_data,
            "trajectory": {
                "turns": [{"role": t.role, "content": t.content} for t in trajectory.turns],
                "tool_calls": [tc.model_dump() for tc in trajectory.tool_calls],
            },
            "metadata": {
                "total_turns": len(trajectory.turns),
                "tool_calls_made": len(trajectory.tool_calls),
                "used_mcp": mcp_client is not None
            }
        }
        
        trajectory_file = output_dir / "trajectory.json"
        trajectory_file.write_text(json.dumps(trajectory_data, indent=2, ensure_ascii=False))
        print(f"\n💾 Complete trajectory saved to: {trajectory_file}")
        
    else:
        print("\n❌ Failed to collect a valid trajectory")
        return False

    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Check {output_dir}/ for output files")
    if mcp_client:
        print("🔗 MCP integration was used successfully")
    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
