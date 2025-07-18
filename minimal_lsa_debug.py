from core.trajectory.pipeline import LsaWorkflowAgent
from config import LLMConfig, agentcortex_config, ASSISTANT_AGENT_OPTIONS
from business_components.workflow import WorkflowConfig
import json

# Minimal LLM config (adjust as needed)
llm_cfg = LLMConfig(
    model="Qwen2.5-72B",  # Replace with your model name
    base_url="https://modelfactory.lenovo.com/service-large-168-1743666716725/llm/v1",  # Replace with your LLM endpoint
    api_key="AnpXAL97M888frrAHmrbHmrdFJs27wrKca776zmZ786zwPJ6MrdJ8Csr6p6wGB7RJ99xk41VF5FHrbrk8zFh86RH8p8bfnKSNrCDqjan8rTal6j2wsxV9lZrJ7VlRfPH"
)

# Use a real tool from blueprint.json
# Example tool: product_recommend
# Example arguments from your blueprint.json
product_recommend_tool = {
    "type": "function",
    "function": {
        "name": "product_recommend",
        "description": "推荐合适的产品。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "category": {"type": "array", "items": {"type": "string"}},
                "request_brand_flag": {"type": "boolean"}
            },
            "required": ["query", "category", "request_brand_flag"]
        }
    }
}

tools_schema = [product_recommend_tool]

# Simulate a user message
history = [
    {"role": "user", "content": "我想批量采购办公用的笔记本，有推荐吗？"}
]

# Instantiate the agent
agent = LsaWorkflowAgent(
    llm_cfg=llm_cfg,
    generation_opts=ASSISTANT_AGENT_OPTIONS,
    tool_names=["product_recommend"]
)

# Call respond and print the output
try:
    output = agent.respond(history, tools_schema)
    print("=== Agent Output ===")
    print(json.dumps(output, indent=2, ensure_ascii=False))
except Exception as e:
    print("Error during agent respond:", e) 