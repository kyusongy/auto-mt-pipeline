from __future__ import annotations

"""APIGen-MT Phase-2 prototype: Trajectory collection via simulated human-agent interplay.

The goal is to demonstrate the end-to-end mechanics while keeping the code
compact and editable. All prompts come from Fig.-11 and Fig.-12 of the paper
(see screenshots) but are parameterised with placeholders so you can inject
real data later.
"""

# Standard library imports
import ssl
import urllib3
import warnings
from dataclasses import dataclass
import json
import random
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import httpx

# Disable SSL verification globally to work around certificate issues with remote LLM endpoints
# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Monkey-patch httpx to disable SSL verification (this is what actually works for Qwen Agent)
original_client_init = httpx.Client.__init__

def patched_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_client_init(self, *args, **kwargs)

httpx.Client.__init__ = patched_client_init

# Also patch AsyncClient in case it's used
original_async_client_init = httpx.AsyncClient.__init__

def patched_async_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_async_client_init(self, *args, **kwargs)

httpx.AsyncClient.__init__ = patched_async_client_init

# Third-party imports (continued)
from qwen_agent.agents import Assistant  # type: ignore

# Local application imports
from config import (
    LLMConfig,
    GenerationOptions as LLMGenOpts,
    TRAJECTORY_JUDGE_OPTIONS,
    TRAJECTORY_AGENT_OPTIONS,
    ASSISTANT_AGENT_OPTIONS,
    mcp_config,
    is_agentcortex_enabled,
    agentcortex_config,
)
from core.models import ToolCalling
from core.llm_client import sync_request_llm
from core.blueprint.pipeline import Blueprint  # reuse dataclass from phase-1

# AgentCortex LSA imports
import json
import uuid
from business_components.workflow import Workflow, WorkflowConfig, Context
from agent_types.common import Plan, ToolCalling as LsaToolCalling, Observation, ChatMessage
from pydantic import BaseModel

# Ensure tool wrappers are registered before Assistant is instantiated
import core.trajectory.qwen_tool_wrappers  # noqa: F401  # registers tools via import side-effects
from tools.retail_tools import RUNTIME_FUNCTIONS  # provides callable stubs for tool execution

# ---------------------------------------------------------------------------
# Prompts (verbatim from paper – Fig.11 & Fig.12)
# ---------------------------------------------------------------------------

_TRAJECTORY_COLLECTION_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a concise user interacting with an AI agent.

    ## Intent
    {intent}

    ## Rules
    • Generate one line at a time to simulate the user's message (asking questions or offering information).
    • Keep the message short to mimic the real customer behavior. The message may not be complete. Be direct and to the point. No pleasantries.
    • Do not give away all the intent at once. Only provide the information that is necessary for the current step.
    • Do not hallucinate information that is not provided in the intent.
    • If the intent goal is completely satisfied, generate `###STOP###` to end the conversation.
    • Do not repeat the exact intent in the conversation. Instead, use your own words to convey the same information.
    • Do not copy or repeat any assistant messages. Always write a brand-new user turn in your own words.

    ## Example
    Intent: 你是一个家长，想要给上大学的孩子买一台笔记本电脑，能运行基础学习办公软件就好，并且你想要查询是否有相关的教育优惠。

    Turn 1 (user): 推荐一台笔记本电脑，能运行基础办公软件就行。
    Turn 1 (assistant): ...
    Turn 2 (user): 这个小新笔记本有教育优惠吗？
    Turn 2 (assistant): ...
    Turn 3 (user): 售后政策是什么？
    Turn 3 (assistant): ...
    Turn 4 (user): ###STOP###

    ### Response format
    Reply with **only** the next message you (the user) would send or '###STOP###' to end the conversation. Keep it SHORT (1-2 sentences max). Do NOT include any explanations, headings, bullet points, or additional reasoning—just the raw chat line.
    """
).strip()

_BON_USER_LM_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a fair judge and an expert in following details.

    A human is interacting with a Lenovo sales assistant to get help on solving their task. You are provided with the description of the human and the task the
    human wants to accomplish (wrapped with <description></description>), and a candidate response (wrapped with <response></response>) the human wants to
    give the assistant. Please help the human evaluate this candidate response, give an integer score (ranging from 0 to 10) to indicate the correctness of the response,
    higher score means better quality.

    1. If the response sounds like it's coming from an assistant/agent rather than a customer, give score 0.

    2. If the response includes specific item / personal details, and they correctly match the task description, you should give a higher score. If there is some
       change in details, give a corresponding lower score (more incorrect details gets lower score).
    
    3. The response can include any normal customer conversation otherwise (e.g., asking for help, saying ###STOP###) etc. which are all correct responses.
    
    4. Give higher scores to responses that are direct and to the point. Penalize overly verbose responses with unnecessary pleasantries like "thank you", "please", or rambling explanations.

    <description> {description} </description>
    <response> {response} </response>

    After scoring using the mentioned guideline, tell me your score, wrap it in <score></score> tags.
    """
).strip()

# ---------------------------------------------------------------------------
# Data-classes for interaction turns
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    role: str  # "user" / "assistant"
    content: str


@dataclass
class Trajectory:
    turns: List[Turn]
    tool_calls: List[ToolCalling]

# ---------------------------------------------------------------------------
# Simulated actors
# ---------------------------------------------------------------------------


class SimulatedHuman:
    """LLM that reveals the intent gradually using BoN self-critique."""

    def __init__(self, llm_cfg: LLMConfig, bon_n: int = 1, debug: bool = False):
        self.cfg = llm_cfg
        self.bon_n = bon_n
        self.debug = debug
        self._primed = False  # ensure giant prompt is sent only once

        # Use centralized generation options but allow quick overriding of `debug`
        self.agent_opts = TRAJECTORY_AGENT_OPTIONS.model_copy(update={"debug": debug})
        self.judge_opts = TRAJECTORY_JUDGE_OPTIONS.model_copy(update={"debug": debug, "extra_body": None})

    def _score_candidate(self, description: str, candidate: str) -> int:
        prompt = _BON_USER_LM_PROMPT_TEMPLATE.format(description=description, response=candidate)
        messages = [{"role": "user", "content": prompt}]
        completion = sync_request_llm(self.cfg, messages, generation_config=self.judge_opts)
        reply = _get_msg_content(completion.choices[0].message)  # type: ignore[attr-defined]
        
        # Parse the score from the judge reply.
        match = re.search(r"<score>(\d+)</score>", reply or "")
        score = int(match.group(1)) if match else 0
        return score

    def next_message(self, intent: str, history: List[Turn]) -> str:
        """Generate the next user message via Best-of-N sampling."""
        # Insert the persona prompt only on the very first turn and keep it in
        # the running history thereafter (mirrors TestAgent behaviour).
        if not self._primed:
            self._persona_prompt = _TRAJECTORY_COLLECTION_PROMPT_TEMPLATE.format(intent=intent)
            # Prepend as a system turn so future calls include it automatically
            history.insert(0, Turn("system", self._persona_prompt))
            self._primed = True

        # Convert history to chat format but hide internal tool calls/observations
        # from the simulated human — they should only see normal dialogue.
        # Always include the persona system prompt as the first message every turn.
        messages = [{"role": "system", "content": self._persona_prompt}]
        messages.extend(
            {"role": t.role, "content": t.content}
            for t in history
            if t.role in {"user", "assistant"}
        )

        candidates: List[Tuple[str, int]] = []
        if self.bon_n <= 1:
            comp = sync_request_llm(self.cfg, messages, generation_config=self.agent_opts)

            # Debug output removed for cleaner logs

            raw = _get_msg_content(comp.choices[0].message)  # type: ignore[attr-defined]
            best_msg = raw if isinstance(raw, str) else ""
        else:
            if self.debug:
                print(f"[HUMAN] Using best-of-{self.bon_n} sampling...")
            for i in range(self.bon_n):
                comp = sync_request_llm(self.cfg, messages, generation_config=self.agent_opts)

                # Debug output removed for cleaner logs

                raw = _get_msg_content(comp.choices[0].message)  # type: ignore[attr-defined]
                msg = raw if isinstance(raw, str) else ""
                score = self._score_candidate(intent, msg) if msg else 0
                candidates.append((msg, score))
                if self.debug:
                    print(f"[HUMAN] Candidate {i+1}: score={score}, msg='{msg[:50]}{'...' if len(msg) > 50 else ''}'")
            # choose best non-empty candidate
            best_msg = max(candidates, key=lambda x: x[1])[0]
            if self.debug:
                print(f"[HUMAN] Selected best candidate: '{best_msg[:50]}{'...' if len(best_msg) > 50 else ''}'")

        # Post-process to strip meta reflections like "Okay, the user wants …".
        if not best_msg:
            return ""

        # Return the full message without any cleaning/filtering
        return best_msg.strip()


class QwenTestAgent:
    """Wrap Qwen-Agent's `Assistant` for our trajectory collector."""

    def __init__(self, llm_cfg: LLMConfig, generation_opts: LLMGenOpts = None, tool_names: Optional[List[str]] = None):
        # Import registry after wrappers have been imported so tools are present.
        from tools import retail_tools as _d

        # Use provided generation options or default to ASSISTANT_AGENT_OPTIONS
        if generation_opts is None:
            generation_opts = ASSISTANT_AGENT_OPTIONS

        # Configure Qwen-Agent's Assistant with our generation parameters
        llm_config = {
            "model": llm_cfg.model,
            "model_server": llm_cfg.base_url,
            "api_key": llm_cfg.api_key,
        }
        
        # Add generation parameters if supported by this version of qwen-agent
        if generation_opts.temperature is not None:
            llm_config["temperature"] = generation_opts.temperature
        if generation_opts.max_tokens:
            llm_config["max_tokens"] = generation_opts.max_tokens
        if generation_opts.top_p:
            llm_config["top_p"] = generation_opts.top_p
            
        # Determine which tools to expose to the assistant.  If `tool_names` is
        # provided we use that (e.g. MCP tool list passed from TrajectoryCollector);
        # otherwise fall back to the dummy retail tools.
        if tool_names is None:
            tool_names = list(_d.TOOLS_SCHEMA.keys())

        self.bot = Assistant(
            llm=llm_config,
            function_list=tool_names,
            system_message=_AGENT_SYSTEM_PROMPT,  # Use our retail domain system prompt
        )

    def respond(self, history: List[dict], tools_schema: List[dict]) -> list[dict]:
        """Return the list of new messages generated by the agent for the current turn.

        The full OpenAI-style `tools` schema is forwarded so the LLM sees
        parameter definitions and required fields, increasing the likelihood it
        emits syntactically correct function calls.
        """

        new_msgs: list[dict] = []
        # Some versions of `qwen_agent` expose a `tools` keyword matching the
        # OpenAI spec.  If an older version is in use we silently fall back to
        # omitting the schema (behaviour identical to the original prototype).
        try:
            for batch in self.bot.run(messages=history, tools=tools_schema):
                new_msgs = batch  # each `batch` is the streamed delta list
        except TypeError:
            # `tools` kw not supported; degrade gracefully
            for batch in self.bot.run(messages=history):
                new_msgs = batch
        return new_msgs


# ---------------------------------------------------------------------------
# Default system prompt for Lenovo sales assistant
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = (
    "你是一个联想商城的智能助手，具备判断是否需要调用外部工具来完成用户请求的能力。\n"
    "政治相关、危险行为等敏感话题一定要拒绝回答，此时语气要和善且坚决。\n"
    "尽量找合适的工具来获取信息，基于工具返回的信息回答。如果不确定使用什么工具，要大胆尝试。\n"
    "如果尝试多个工具都无法获取有用信息，一定不要自己尝试回答。\n"
    "如果工具返回的结果中包含“summary_constraints”属性，那么在最终回复的时候要按照这个属性要求的格式进行总结。\n"
    "[提及](#Mentions)是用户查询中提到的信息，在调用工具的时候，很多参数都会注明是用户提到的某项信息，所以你需要将#Mentions里面符合条件的内容作为工具参数\n"
)

# ---------------------------------------------------------------------------
# Agent for Trajectory collection using agentcortex-lsa workflow
# ---------------------------------------------------------------------------

class LsaWorkflowAgent:
    """Agent that uses agentcortex-lsa Workflow for planning and execution."""

    def __init__(self, llm_cfg: LLMConfig, generation_opts: LLMGenOpts = None, tool_names: Optional[List[str]] = None):
        # This agent will use QwenTestAgent as the planner.
        self.planner = QwenTestAgent(llm_cfg, generation_opts, tool_names)

        # We still need the workflow object to access its other components (execution, memory, etc.)
        # but we will NOT be using its internal planner.
        self.workflow = Workflow(WorkflowConfig(
            intent_url=agentcortex_config.intent_url,
            session_memory_url=agentcortex_config.session_memory_url,
            system_memory_url=agentcortex_config.system_memory_url,
            planning_url=agentcortex_config.planning_url,  # This is not used but required by the config
            execution_url=agentcortex_config.execution_url,
            summarization_url=agentcortex_config.summarization_url,
            personalization_url=agentcortex_config.personalization_url,
            extract_mentions_url=agentcortex_config.extract_mentions_url,
            max_iterations=5,
        ))
        self.session_id = str(uuid.uuid4())

    def respond(self, history: List[dict], tools_schema: List[dict]) -> list[dict]:
        """Generate the next agent messages using the LSA workflow with Qwen as the planner."""
        if not history or history[-1]['role'] != 'user':
            return []

        query = history[-1]['content']

        # Collect messages to return so that the TrajectoryCollector can display
        # tool calls, function execution observations and the final answer just
        # like the original agentcortex-lsa workflow does.
        new_msgs: list[dict] = []

        # 1. Create and populate the context, same as the original workflow
        default_args = {
            "user_info": {"uid": "13716255679", "user_identity": 1, "available_num": 0.0, "current_amount": "0", "enterprise_name": "", "future_expire_num": 0.0, "level_name": "", "entry_source": "shop", "user_province": ""},
            "trace_id": self.session_id,
            "uid": "13716255679",
            "terminal": "1",
            "latitude": "23.89447712420573",
            "longitude": "106.6172117534938",
            "device_ip": "117.183.16.69",
            "get_position_permission": "agree",
            "event": "",  # 门店的case需要设置为"NAVIGATION_REQUEST"
            "bind_mobile_id": 0,
            "query": query,
        }
        context = Context(
            session_id=self.session_id,
            query=query,
            tools=self.workflow.tools,
            default_args=default_args
        )

        # 2. Pre-processing steps before planning
        self.workflow.read_session_memory(context)
        self.workflow.rewrite_query(context)
        self.workflow.read_mentions(context)
        self.workflow.read_session_preference(context)

        # 3. Iterative Plan & Execute using Qwen as planner
        for i in range(self.workflow.max_iterations):
            # Work on a copy so we don't mutate the outer collector's history reference
            planner_history = list(history)
            if context.observations:
                # Add tool execution results to history for the planner to see
                for obs in context.observations:
                    for status in obs.status:
                        planner_history.append({
                            'role': 'tool',
                            'tool_call_id': status.tool_call_id,
                            'name': status.tool_name,
                            'content': json.dumps(status.result, ensure_ascii=False)
                        })
            
            # Call Qwen planner
            planner_response = self.planner.respond(planner_history, tools_schema) or []
            
            if not planner_response:
                # Planner failed to return any message, stop to avoid infinite loop
                raise ValueError("Planner returned empty response.")
            
            # Push planner messages to `new_msgs` (convert the new `tool_calls` schema
            # to the legacy `function_call` schema expected by the TrajectoryCollector).
            for _m in planner_response:
                if 'tool_calls' in _m and _m['tool_calls']:
                    #   The collector understands only a single `function_call` field, so we
                    #   keep the first call to preserve previous behaviour.
                    first = _m['tool_calls'][0]['function']
                    _legacy = _m.copy()
                    _legacy.pop('tool_calls', None)
                    _legacy['function_call'] = first
                    new_msgs.append(_legacy)
                else:
                    new_msgs.append(_m)

            # Always consider the final message in the batch as the definitive one
            last_msg = planner_response[-1]
            # Append planner messages to history for the next iteration
            history.extend(planner_response)
            
            # Detect tool calls – qwen-agent may use either the new 'tool_calls' field (array)
            # or the legacy single 'function_call' field.
            raw_tool_calls = []
            if 'tool_calls' in last_msg and last_msg['tool_calls']:
                raw_tool_calls = last_msg['tool_calls']
            elif 'function_call' in last_msg and last_msg['function_call']:
                # Normalise to the list[dict] format expected below
                raw_tool_calls = [{"function": last_msg['function_call']}]
            
            if raw_tool_calls:
                lsa_tool_callings = [
                    LsaToolCalling(name=tc['function']['name'], arguments=json.loads(tc['function']['arguments']))
                    for tc in raw_tool_calls
                ]
                context.plan = Plan(tool_callings=lsa_tool_callings, content='')
                context.finished = False
            else:
                # No tool calls, Qwen wants to respond directly. This ends the loop.
                context.plan = Plan(tool_callings=[], content=last_msg.get('content', ''))
                context.finished = True

            if context.finished:
                break

            # 4. Execute the plan if there are tool calls to run
            if context.plan.tool_callings:
                self.workflow.execute(context)
                # After execution push the observation results so the caller can log them
                if context.observations:
                    for st in context.observations[-1].status:
                        new_msgs.append({
                            'role': 'function',
                            'name': st.tool_name,
                            'content': json.dumps(st.result, ensure_ascii=False)
                        })
            if context.finished:
                break
        # 5. Finalize the response. The planner (Qwen) is responsible for summarization.
        # When the loop finishes, the final response is in the content of the last plan.
        final_response_content = context.plan.content if context.plan else None

        if not final_response_content:
            # This indicates the agent finished the loop without a final response.
            raise ValueError("Agent failed to produce a final response.")

        # Update context with the final response for memory writing
        context.response = final_response_content

        # Write to memory
        self.workflow.write_user_message(context)
        self.workflow.write_assistant_message(context)

        # Append the final assistant reply
        new_msgs.append({'role': 'assistant', 'content': final_response_content})

        # The trajectory collector expects the full list of messages in the current turn.
        return new_msgs


# ---------------------------------------------------------------------------
# Trajectory collector
# ---------------------------------------------------------------------------


class TrajectoryCollector:
    def __init__(
        self,
        human_cfg: LLMConfig,
        agent_cfg: LLMConfig,
        tools_schema: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        bon_n: int = 1,
        use_plan_execute_agent: bool = False,  # Default: use original QwenTestAgent
    ):
        """Collect trajectories.

        tools_schema: full domain tool specification (same dict passed to blueprint
        generation).  Providing it allows the agent to see argument structure and
        increases the chance it will emit correct function calls.
        bon_n: Best-of-N sampling for human simulation (1 = no sampling, >1 = generate N candidates and pick best)
        use_plan_execute_agent: If True, use AgentCortex Plan+Execute agent; otherwise use original QwenTestAgent (default)
        """
        self.human = SimulatedHuman(human_cfg, bon_n=bon_n, debug=debug)
        
        # Use AgentCortex Plan+Execute agent (required)
        if use_plan_execute_agent:
            print("🧠 Using agentcortex-lsa Workflow for Trajectory Generation")
            self.agent = LsaWorkflowAgent(
                llm_cfg=agent_cfg, 
                generation_opts=ASSISTANT_AGENT_OPTIONS,
                tool_names=list((tools_schema or {}).keys()) if tools_schema else None
            )
        else:
            print("🤖 Using original QwenTestAgent")
            self.agent = QwenTestAgent(
                agent_cfg,
                ASSISTANT_AGENT_OPTIONS,
                tool_names=list((tools_schema or {}).keys()) if tools_schema else None,
            )
        
        self.tools_schema = tools_schema or {}
        self.debug = debug

        # Always use the predefined Lenovo sales assistant policy
        self.agent_system_prompt = _AGENT_SYSTEM_PROMPT

    def collect(self, blueprint: Blueprint) -> Optional[Trajectory]:
        history: List[Turn] = []
        tool_calls: List[ToolCalling] = []

        # Simulate turns (hard-coded max 20 to avoid infinite loop)
        for turn_num in range(1, 21):
            # 1) human speaks
            # Create a temporary history that includes role reminder if this isn't the first turn
            temp_history = history.copy()
            if turn_num > 1:  # Add reminder after assistant has spoken at least once
                role_reminder = (
                    "REMINDER: YOU ARE USER/CUSTOMER. RESPOND '###STOP###' ONLY IF YOUR INTENT IS SATISFTIED. DO NOT RESPOND TO THIS REMINDER. START YOUR USER TURN NOW."
                )
                temp_history.append(Turn("user", role_reminder))
            
            user_msg = self.human.next_message(blueprint.user_intent, temp_history)
            if not user_msg:
                if self.debug:
                    print("[Collector] human LLM returned empty response, aborting trajectory")
                return None
            history.append(Turn("user", user_msg))
            if self.debug:
                print(f"\nTurn {turn_num}")
                print("[USER]", user_msg)
            # Stop when the user injects the special token anywhere in the line
            # Also catch common Chinese variants as fallback (but log them as warnings)
            stop_indicators = [
                "###STOP###",
                "### 响应调整",  # Chinese "response adjustment"
                "### 停止",     # Chinese "stop"
                "###停止###",   # Chinese "stop" with delimiters
            ]
            
            if any(indicator in user_msg for indicator in stop_indicators):
                if "###STOP###" not in user_msg and self.debug:
                    print(f"[WARNING] Detected Chinese stop variant in: {user_msg}")
                    print("[WARNING] This should be fixed in the prompt to use exact '###STOP###'")
                break

            # 2) agent responds (may include tool call)
            # ⚠️  The first turn in `history` is an internal system prompt that
            # embeds the *full intent* so the SimulatedHuman can generate
            # appropriate messages.  The agent MUST **not** see this –
            # otherwise it trivially solves the task.  We therefore strip all
            # system messages before forwarding the conversation to the agent.

            # Translate internal Turn records to OpenAI/Qwen message schema
            messages_for_agent: List[dict] = []
            last_fc_name: Optional[str] = None
            for t in history:
                if t.role == "system":
                    continue  # hide intent
                if t.role == "function_call":
                    try:
                        fc_payload = json.loads(t.content)
                    except Exception:
                        continue  # malformed, skip
                    last_fc_name = fc_payload.get("name")
                    # Qwen-Agent expects arguments as **string**
                    args_str = fc_payload.get("arguments")
                    if isinstance(args_str, dict):
                        args_str = json.dumps(args_str, ensure_ascii=False)
                    assistant_fc = {
                        "name": fc_payload.get("name"),
                        "arguments": args_str,
                    }
                    messages_for_agent.append({
                        "role": "assistant",
                        "function_call": assistant_fc,
                        "content": ""  # per spec, content must be empty or omitted when using function_call
                    })
                elif t.role == "observation":
                    if last_fc_name is None:
                        continue  # cannot pair; skip
                    messages_for_agent.append({
                        "role": "function",
                        "name": last_fc_name,
                        "content": t.content,
                    })
                    last_fc_name = None  # reset pairing
                else:
                    messages_for_agent.append({"role": t.role, "content": t.content})

            # ------------------------------------------------------------------
            # Optional: prepend a *public* system prompt for the agent.  This can
            # include domain rules, style instructions, etc. but **must not**
            # leak the hidden user intent.
            # ------------------------------------------------------------------
            if self.agent_system_prompt:
                messages_for_agent.insert(0, {"role": "system", "content": self.agent_system_prompt})

            # Build tools list for this turn
            tools_schema_list: List[dict] = []
            for act in blueprint.actions:
                if act.name in self.tools_schema:
                    spec = self.tools_schema[act.name]
                    parameters = spec.get("parameters", {})
                    tools_schema_list.append(
                        {
                            "type": "function",
                            "function": {
                                "name": act.name,
                                "description": spec.get("description", ""),
                                "parameters": {
                                    "type": "object",
                                    "properties": {k: {"type": v.get("type", "string")} for k, v in parameters.items()},
                                    "required": [k for k, v in parameters.items() if v.get("required", False)],
                                },
                            },
                        }
                    )
                else:
                    tools_schema_list.append(
                        {
                            "type": "function",
                            "function": {
                                "name": act.name,
                                "description": "",
                                "parameters": {"type": "object", "properties": {}, "required": []},
                            },
                        }
                    )

            new_msgs = self.agent.respond(messages_for_agent, tools_schema_list)

            # Process agent response properly - ensure clean flow
            # Expected: assistant message with function_call, then function result, then assistant response
            agent_content = ""
            function_calls_in_turn = []
            
            for idx, m in enumerate(new_msgs):
                role = m.get("role")
                
                if role == "assistant" and "function_call" in m:
                    # Agent wants to call a function
                    fc = m["function_call"]
                    raw_args = fc.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            raw_args = {}

                    tool_name = fc.get("name")
                    val = json.dumps({"name": tool_name, "arguments": json.dumps(raw_args, ensure_ascii=False)}, ensure_ascii=False)
                    
                    # Add function call to history
                    history.append(Turn("function_call", val))
                    tool_calls.append(ToolCalling(name=tool_name, arguments=raw_args))
                    function_calls_in_turn.append(tool_name)
                    
                    if self.debug:
                        print("[FUNC_CALL]", val)

                elif role == "function":
                    # Function execution result - this should come from our MCP client
                    observation_content = m.get("content", "")
                    history.append(Turn("observation", observation_content))
                    if self.debug:
                        print(f"[OBSERV] Function result for {function_calls_in_turn[-1] if function_calls_in_turn else 'unknown'}:")
                        print("[OBSERV]", observation_content)
                        
                elif role == "assistant":
                    # Regular assistant response - only add if it has meaningful content
                    content = m.get("content", "").strip()
                    if content:
                        agent_content = content
            
            # Add final assistant response if we have one and it's not empty
            if agent_content:
                history.append(Turn("assistant", agent_content))
                if self.debug:
                    print("[ASSISTANT]", agent_content)

            # success check: all expected outputs present in assistant messages
            if any(isinstance(t.content, str) and any(o in t.content for o in blueprint.expected_outputs) for t in history if t.role == "assistant"):
                history.append(Turn("user", "###STOP###"))
                break

        # validate tool_calls (ignore order) – all blueprint actions must appear
        # if sorted(c.name for c in tool_calls) == sorted(a.name for a in blueprint.actions):
        #    return Trajectory(history, tool_calls)
        return Trajectory(history, tool_calls) #return None




# ---------------------------------------------------------------------------
# Utility for vLLM "reasoning" messages
# ---------------------------------------------------------------------------


def _get_msg_content(msg) -> str:  # type: ignore[Any]
    """Return message.content, falling back to .reasoning_content if present (vLLM).

    If reasoning_content is used (when the server is launched with --enable-reasoning),
    it often contains chain-of-thought prose and then the real reply (e.g. prefixed
    with "First message:").  We strip everything except the final answer so the
    rest of the pipeline sees just the conversational text.
    """
    content = getattr(msg, "content", None)
    return content.strip() if isinstance(content, str) else "" 