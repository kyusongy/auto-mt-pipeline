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
    is_agentcortex_enabled,
    agentcortex_config,
)
from core.blueprint.iterative_generator import generate_validated_blueprint
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
# Note: GenerationOptions is imported from config, not from bp_pipeline

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
            print(f"✅ Loaded {len(schemas)} tools from MCP service")
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
    
    # Check if AgentCortex integration is enabled
    if is_agentcortex_enabled():
        print("🧠 AgentCortex Integration: ENABLED (Full AgentCortex Services)")
        print("   ✓ Planning service:", agentcortex_config.planning_url)
        print("   ✓ Execution service:", agentcortex_config.execution_url)
        print("   ✓ Session memory service:", agentcortex_config.session_memory_url)
        print("   ✓ Personalization service:", agentcortex_config.personalization_url)
    else:
        print("❌ AgentCortex Integration: DISABLED")
        print("   To use AgentCortex integration, set AGENTCORTEX_ENABLED=true")
        print("   This pipeline requires AgentCortex services to be running.")
        print("   Exiting...")
        return False
    print()
    
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
    
    # Use new two-stage iterative blueprint generation
    if not mcp_client:
        raise RuntimeError("MCP client is required for the new iterative blueprint generation")
    
    print("🔄 Using two-stage iterative blueprint generation with action validation")
    blueprint = generate_validated_blueprint(
        llm_config,
        tools_schema,
        PERSONAS,
        mcp_client,
        max_attempts=pipeline_config.max_blueprint_attempts,
        max_generation_attempts_per_persona=pipeline_config.max_blueprint_iterations,
        max_execution_iterations=pipeline_config.max_blueprint_iterations,
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
    
    # Show method used
    method_used = "two_stage_iterative_validated"
    print(f"  Generation method: {method_used}")
    
    # Show the blueprint structure
    blueprint_data = {
        "intent": blueprint.user_intent,
        "actions": [a.model_dump() for a in blueprint.actions],
        "outputs": blueprint.expected_outputs,
        "metadata": {
            "generation_method": method_used,
            "used_mcp": True,
            "stage_1": "intent_actions_validation",
            "stage_2": "execution_validation"
        }
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
        use_plan_execute_agent=True,  # Enable Qwen+AgentCortex hybrid execution
        enable_validation=True,  # Enable trajectory validation
        validation_llm_config=llm_config,  # Use same LLM for validation (can be different)
    )

    # --------------------------------------------------------------
    # Try to collect a valid trajectory up to 3 times for this
    # approved blueprint (stochastic generation may sometimes fail)
    # --------------------------------------------------------------
    max_collect_attempts = 3
    trajectory = None
    for collect_attempt in range(1, max_collect_attempts + 1):
        print(f"\n🎬 Trajectory collection attempt {collect_attempt}/{max_collect_attempts}")
        trajectory = collector.collect(blueprint)
        if trajectory:
            break
        print("⚠️  Trajectory rejected. Retrying...")

    if trajectory:
        num_assistant_turns = sum(1 for t in trajectory.turns if t.role == "assistant")
        print(f"\n✅ Successfully collected trajectory with {num_assistant_turns} turns")
        

        def safe_json_loads(s):
            try:
                return json.loads(s)
            except Exception:
                return s

        simplified_turns = []
        user_buffer = None
        tool_calls_buffer = []
        turn_id_counter = 0
        pending_tool = None

        for t in trajectory.turns:
            if t.role == "user":
                user_buffer = t.content
                tool_calls_buffer = []
                pending_tool = None
            elif t.role == "function_call":
                call = safe_json_loads(t.content)
                name = call.get("name", "") if isinstance(call, dict) else ""
                arguments = call.get("arguments", "{}") if isinstance(call, dict) else "{}"
                arguments = safe_json_loads(arguments)
                pending_tool = {
                    "name": name,
                    "arguments": arguments,
                    "output": None
                }
            elif t.role in {"function", "observation"}:
                if pending_tool is not None:
                    pending_tool["output"] = safe_json_loads(t.content)
                    tool_calls_buffer.append(pending_tool)
                    pending_tool = None
            elif t.role == "assistant" and user_buffer is not None:
                turn_id_counter += 1
                # Ensure all tool_info entries are serializable
                for tool in tool_calls_buffer:
                    if not isinstance(tool["arguments"], (dict, list, str, int, float, bool, type(None))):
                        tool["arguments"] = str(tool["arguments"])
                    if not isinstance(tool["output"], (dict, list, str, int, float, bool, type(None))):
                        tool["output"] = str(tool["output"])
                simplified_turns.append({
                    "turn_id": turn_id_counter,
                    "query": user_buffer,
                    "response": t.content,
                    "tool_info": tool_calls_buffer
                })
                user_buffer = None
                tool_calls_buffer = []
                pending_tool = None

        session_data = {"turns": simplified_turns}

        conversations_path = output_dir / "collected_sessions.json"
        conversations_path.parent.mkdir(parents=True, exist_ok=True)
        if conversations_path.exists():
            try:
                existing = json.loads(conversations_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    existing.append(session_data)
                else:
                    existing = [session_data]
            except Exception:
                existing = [session_data]
        else:
            existing = [session_data]

        conversations_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n💾 Session appended to: {conversations_path}")
        
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
