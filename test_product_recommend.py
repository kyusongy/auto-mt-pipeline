#!/usr/bin/env python3
"""
Test Product Recommend Tool

Test the product_recommend tool directly with various queries to understand
why it's not finding suitable products for users.

Usage:
    python test_product_recommend.py
"""

# Standard library imports
import json
from pathlib import Path

# Local application imports
from config import mcp_config
from core.mcp_client import MCPClient, MCPConfig

def test_product_recommend_queries():
    """Test various product recommendation queries."""
    
    if not mcp_config["enabled"]:
        print("❌ MCP integration disabled - cannot test product_recommend")
        return False
        
    try:
        executor_url = mcp_config["executor_url"]
        print(f"🔗 Connecting to MCP executor at {executor_url}")
        mcp_client_config = MCPConfig(executor_url=executor_url)
        mcp_client = MCPClient(mcp_client_config)
        print("✅ Connected to MCP service")
    except Exception as e:
        print(f"❌ Failed to connect to MCP: {e}")
        return False

    # Test queries with different parameters
    test_cases = [
        {
            "name": "Basic laptop search",
            "arguments": {
                "query": "笔记本",
                "category": ["笔记本"]
            }
        },
        {
            "name": "Budget laptop search (6000 yuan)",
            "arguments": {
                "query": "笔记本 6000元 学生",
                "category": ["笔记本"],
                "request_brand_flag": False
            }
        },
        {
            "name": "Student laptop with budget",
            "arguments": {
                "query": "大学生笔记本电脑 预算6000元 学习 游戏",
                "category": ["笔记本"],
                "request_brand_flag": True,
                "request_brand": ["小新", "ThinkPad"]
            }
        },
        {
            "name": "Lenovo specific laptop search",
            "arguments": {
                "query": "联想笔记本 6000元左右 性价比高",
                "category": ["笔记本"],
                "request_brand_flag": True,
                "request_brand": ["小新", "ThinkPad", "拯救者"]
            }
        },
        {
            "name": "Gaming laptop budget search",
            "arguments": {
                "query": "笔记本 游戏 办公 6000",
                "category": ["笔记本"],
                "request_brand_flag": False
            }
        },
        {
            "name": "Very broad search",
            "arguments": {
                "query": "电脑",
                "category": ["笔记本"],
                "request_brand_flag": False
            }
        }
    ]
    
    print("\n🧪 Testing product_recommend with various queries:")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"Query: {test_case['arguments']['query']}")
        print(f"Arguments: {json.dumps(test_case['arguments'], ensure_ascii=False, indent=2)}")
        
        try:
            # Execute the tool
            result = mcp_client.execute_tool(
                "product_recommend", 
                test_case['arguments'], 
                test_case['arguments']['query']
            )
            
            # Show raw result
            print("📋 RAW TOOL OUTPUT:")
            print("=" * 50)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("=" * 50)
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("-" * 40)
    
    print("\n🔍 Analysis:")
    print("- Check if products are being returned")
    print("- Verify price ranges match user budgets")
    print("- Look for clarification requests that might indicate missing parameters")
    print("- Check if brand filtering is working correctly")
    
    return True

def main():
    """Test the product_recommend tool."""
    print("="*60)
    print("🧪 Product Recommend Tool Test")
    print("="*60)
    
    success = test_product_recommend_queries()
    
    if success:
        print("\n🎉 Test completed!")
    else:
        print("\n❌ Test failed!")
    
    return success

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