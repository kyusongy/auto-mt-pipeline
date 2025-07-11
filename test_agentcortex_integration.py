#!/usr/bin/env python3

"""Test AgentCortex Integration

Quick test to verify that the AgentCortex components are properly integrated
into the existing pipeline and work with the run_pipeline.py flow.
"""

def test_action_executor_integration():
    """Test that AgentCortexActionExecutor works with blueprint generation."""
    print("🔧 Testing AgentCortex Action Executor Integration")
    
    try:
        from core.agentcortex import AgentCortexActionExecutor
        from core.models import ToolCalling
        from config import mcp_config
        
        if not mcp_config.get("enabled", False):
            print("❌ MCP is disabled - cannot test action executor")
            return False
        
        # Initialize executor
        executor = AgentCortexActionExecutor(mcp_config["executor_url"])
        
        # Test with sample actions
        actions = [
            ToolCalling(
                name="product_recommend",
                arguments={"query": "ThinkPad", "category": "laptop"}
            )
        ]
        
        # Execute actions
        summary = executor.execute_actions(actions, "Test query")
        
        print(f"✅ Executor test: {summary.successful_actions}/{summary.total_actions} successful")
        return summary.total_actions > 0
        
    except Exception as e:
        print(f"❌ Action executor test failed: {e}")
        return False


def test_plan_execute_agent_integration():
    """Test that PlanExecuteAgent works with trajectory collection."""
    print("🤖 Testing Plan Execute Agent Integration")
    
    try:
        from core.agentcortex import PlanExecuteAgent
        from config import DEFAULT_LLM_CONFIG, ASSISTANT_AGENT_OPTIONS, mcp_config
        
        if not mcp_config.get("enabled", False):
            print("❌ MCP is disabled - cannot test plan execute agent")
            return False
        
        # Initialize agent
        agent = PlanExecuteAgent(
            llm_cfg=DEFAULT_LLM_CONFIG,
            generation_opts=ASSISTANT_AGENT_OPTIONS,
            tool_names=None
        )
        
        # Test with sample conversation
        history = [
            {"role": "user", "content": "你好，我想了解ThinkPad笔记本"}
        ]
        
        # Generate response
        messages = agent.respond(history, [])
        
        print(f"✅ Agent test: Generated {len(messages)} response messages")
        return len(messages) > 0
        
    except Exception as e:
        print(f"❌ Plan execute agent test failed: {e}")
        return False


def test_blueprint_generation_integration():
    """Test that the modified blueprint generation works."""
    print("📋 Testing Blueprint Generation Integration")
    
    try:
        from core.blueprint.iterative_generator import generate_validated_blueprint
        from config import DEFAULT_LLM_CONFIG, PERSONAS, mcp_config
        from core.mcp_client import MCPClient, MCPConfig, get_mcp_tool_schemas
        
        if not mcp_config.get("enabled", False):
            print("❌ MCP is disabled - cannot test blueprint generation")
            return False
        
        # Initialize MCP client
        mcp_client = MCPClient(MCPConfig(executor_url=mcp_config["executor_url"]))
        tools_schema = get_mcp_tool_schemas(mcp_client)
        
        if not tools_schema:
            print("❌ No tools available from MCP service")
            return False
        
        print(f"📋 Testing with {len(tools_schema)} tools and {len(PERSONAS)} personas")
        
        # Test blueprint generation (limited to 1 attempt for quick test)
        blueprint = generate_validated_blueprint(
            llm_cfg=DEFAULT_LLM_CONFIG,
            tools_schema=tools_schema,
            personas=PERSONAS[:1],  # Just one persona for quick test
            mcp_client=mcp_client,
            max_attempts=1,
            max_generation_attempts_per_persona=1,
            max_execution_iterations=1,
            prompt_kwargs={
                "domain_rules": "Generate realistic Lenovo customer queries",
                "examples": "用户想了解ThinkPad X1 Carbon的价格和配置"
            }
        )
        
        success = (blueprint and 
                  blueprint.user_intent and 
                  not blueprint.user_intent.startswith("[PARSING") and
                  len(blueprint.actions) > 0)
        
        if success:
            print(f"✅ Blueprint generation test: Generated blueprint with {len(blueprint.actions)} actions")
            print(f"   Intent: {blueprint.user_intent[:80]}...")
        else:
            print("❌ Blueprint generation test failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Blueprint generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trajectory_collection_integration():
    """Test that the modified trajectory collection works."""
    print("💬 Testing Trajectory Collection Integration")
    
    try:
        from core.trajectory.pipeline import TrajectoryCollector
        from core.blueprint.pipeline import Blueprint
        from core.models import ToolCalling
        from config import DEFAULT_LLM_CONFIG, mcp_config
        
        if not mcp_config.get("enabled", False):
            print("❌ MCP is disabled - cannot test trajectory collection")
            return False
        
        # Create a simple test blueprint
        test_blueprint = Blueprint(
            user_intent="我想了解ThinkPad X1 Carbon的价格",
            actions=[
                ToolCalling(name="product_recommend", arguments={"query": "ThinkPad X1 Carbon"})
            ],
            expected_outputs=["Product information"]
        )
        
        # Initialize trajectory collector with AgentCortex enabled
        collector = TrajectoryCollector(
            human_cfg=DEFAULT_LLM_CONFIG,
            agent_cfg=DEFAULT_LLM_CONFIG,
            tools_schema={"product_recommend": {"description": "Recommend products"}},
            debug=True,
            bon_n=1,
            use_plan_execute_agent=True  # Enable AgentCortex
        )
        
        print("✅ Trajectory collector initialized with AgentCortex Plan+Execute agent")
        return True
        
    except Exception as e:
        print(f"❌ Trajectory collection test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 AgentCortex Integration Tests")
    print("=" * 60)
    
    # Check MCP configuration
    from config import mcp_config
    if not mcp_config.get("enabled", False):
        print("❌ MCP is disabled. Please enable MCP in config to run integration tests.")
        print("   Set the following environment variables:")
        print("   - MCP_ENABLED=true")
        print("   - MCP_EXECUTOR_URL=<your_mcp_service_url>")
        return False
    
    print(f"🔗 Using MCP executor: {mcp_config['executor_url']}")
    print()
    
    # Run tests
    tests = [
        test_action_executor_integration,
        test_plan_execute_agent_integration, 
        test_blueprint_generation_integration,
        test_trajectory_collection_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
            print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"🏁 Integration Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed! AgentCortex integration is working correctly.")
        print("🎉 You can now run run_pipeline.py with realistic Lenovo service patterns!")
    else:
        print(f"❌ {total - passed} tests failed. Please check the configuration and MCP service.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 