#!/usr/bin/env python3
"""
Test Phase 2 Only - Trajectory Collection

Skip the time-consuming blueprint generation and test trajectory collection directly
using the existing blueprint.json file.

Usage:
    python test_trajectory_only.py
"""

# Standard library imports
import json
from pathlib import Path

# Local application imports
from config import (
    DEFAULT_LLM_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    mcp_config,
)
from core.trajectory.pipeline import TrajectoryCollector
from core.blueprint.pipeline import Blueprint, ToolCalling
from core.mcp_client import MCPClient, MCPConfig, get_mcp_tool_schemas

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

def load_blueprint_from_file(blueprint_file: Path) -> Blueprint:
    """Load blueprint from JSON file."""
    if not blueprint_file.exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_file}")
    
    blueprint_data = json.loads(blueprint_file.read_text(encoding="utf-8"))
    
    # Convert actions from dict to ToolCalling objects
    actions = []
    for action_dict in blueprint_data["actions"]:
        actions.append(ToolCalling(
            name=action_dict["name"],
            arguments=action_dict["arguments"]
        ))
    
    return Blueprint(
        user_intent=blueprint_data["intent"],
        actions=actions,
        expected_outputs=blueprint_data["outputs"]
    )

def main():
    """Run Phase 2 only - Trajectory Collection."""
    print("="*60)
    print("🚀 Test Phase 2 Only - Trajectory Collection")
    print("="*60)
    
    # Load existing blueprint
    blueprint_file = Path("data/blueprint.json")
    print(f"📖 Loading blueprint from: {blueprint_file}")
    
    try:
        blueprint = load_blueprint_from_file(blueprint_file)
        print(f"✅ Loaded blueprint:")
        print(f"   Intent: {blueprint.user_intent[:100]}...")
        print(f"   Actions: {len(blueprint.actions)} tool calls")
        print(f"   Expected outputs: {len(blueprint.expected_outputs)} items")
    except Exception as e:
        print(f"❌ Failed to load blueprint: {e}")
        return False
    
    # Get tool schemas (MCP or fallback)
    tools_schema, mcp_client = get_tool_schemas()

    # Phase 2: Collect trajectory
    print("\n💬 Phase 2: Trajectory Collection")
    print("-" * 40)

    # Configure trajectory collector with MCP if available
    if mcp_client:
        print("🔗 Using MCP tools for trajectory collection")
        # Update Qwen tool wrappers to use our MCP client
        from core.trajectory.qwen_tool_wrappers import initialize_mcp_client
        initialize_mcp_client()  # Will use config automatically
    else:
        print("📦 Using dummy tools for trajectory collection")

    # Use centralized configuration
    llm_config = DEFAULT_LLM_CONFIG
    pipeline_config = DEFAULT_PIPELINE_CONFIG

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
        
        # Save trajectory with timestamp
        output_dir = Path("data")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        trajectory_data = {
            "blueprint": {
                "intent": blueprint.user_intent,
                "actions": [a.model_dump() for a in blueprint.actions],
                "outputs": blueprint.expected_outputs,
            },
            "trajectory": {
                "turns": [{"role": t.role, "content": t.content} for t in trajectory.turns],
                "tool_calls": [tc.model_dump() for tc in trajectory.tool_calls],
            },
            "metadata": {
                "total_turns": len(trajectory.turns),
                "tool_calls_made": len(trajectory.tool_calls),
                "used_mcp": mcp_client is not None,
                "test_timestamp": timestamp
            }
        }
        
        trajectory_file = output_dir / f"trajectory_test_{timestamp}.json"
        trajectory_file.write_text(json.dumps(trajectory_data, indent=2, ensure_ascii=False))
        print(f"\n💾 Test trajectory saved to: {trajectory_file}")
        
    else:
        print("\n❌ Failed to collect a valid trajectory")
        return False

    print("\n🎉 Phase 2 test completed successfully!")
    print(f"📁 Check data/ for trajectory output")
    if mcp_client:
        print("🔗 MCP integration was used successfully")
    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 