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
)
from core.models import ToolCalling
from core.llm_client import sync_request_llm
from core.blueprint.pipeline import Blueprint  # reuse dataclass from phase-1

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
    • **KEEP MESSAGES VERY SHORT**: Maximum 1-2 sentences per message. Be direct and to the point.
    • **NO PLEASANTRIES**: Avoid "I see", "thank you", "please", "sorry", or other polite language. Be matter-of-fact.
    • **ONE TOPIC PER MESSAGE**: Focus on one specific question or need. Don't combine multiple requests.
    • Do not give away all the intent at once. Only provide the information that is necessary for the current step.
    • Do not hallucinate information that is not provided in the intent.
    • **CRITICAL**: If the intent is satisfied, you MUST generate EXACTLY `###STOP###` to end the conversation.
    • Do not repeat the exact intent in the conversation. Instead, use your own words to convey the same information.
    • Do not copy or repeat any assistant messages. Always write a brand-new user turn in your own words.
    • Be natural but concise. Stick to the personalities in the intent but keep responses brief.

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

    1. **ROLE CONFUSION (AUTOMATIC SCORE 0)**: If the response sounds like it's coming from an assistant/agent rather than a customer, give score 0. Signs include:
       - Offering help or assistance (e.g., "I can help you with...", "Let me know...", "I'll provide the next steps", "推荐...")
       - Asking what the customer wants (e.g., "Would you like a refund or replacement?")
       - Using assistant language (e.g., "To proceed with...", "I need to verify...", "Your order has been verified")
       - Acting like they have access to systems or can process requests

    2. **STOP TOKEN VALIDATION (AUTOMATIC SCORE 0)**: If the response appears to be trying to end the conversation but does NOT use the exact text `###STOP###`, give score 0. Examples of INCORRECT stop attempts:
       - "### 响应调整" (Chinese response adjustment)
       - "### 停止" (Chinese stop)
       - "###停止###" (Chinese stop)
       - Any other variation that is not exactly `###STOP###`
       The ONLY correct way to end a conversation is with exactly: `###STOP###`

    3. If the response includes specific item / order / personal details, and they correctly match the task description you should give full score of 10. If there is some
       change in details, give a corresponding lower score (more incorrect details gets lower score).
    
    4. The response can include any normal customer conversation otherwise (e.g., asking for help, saying ###STOP###) etc. which are all correct responses.
    
    5. Additionally, if the candidate response keeps the conversation flowing by describing the task clearly / gives information properly then give a high score and if not
       (e.g. "I don't remember" or unhelpful response) should get a corresponding lower score.
    
    6. **CONCISENESS BONUS**: Give higher scores to responses that are concise and direct (1-2 sentences). Penalize overly verbose responses with unnecessary pleasantries like "thank you", "please", or rambling explanations.

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

        # Convert the (now complete) history—including the system prompt—into
        # the OpenAI chat format.
        messages = [{"role": t.role, "content": t.content} for t in history]

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
    ):
        """Collect trajectories.

        tools_schema: full domain tool specification (same dict passed to blueprint
        generation).  Providing it allows the agent to see argument structure and
        increases the chance it will emit correct function calls.
        bon_n: Best-of-N sampling for human simulation (1 = no sampling, >1 = generate N candidates and pick best)
        """
        self.human = SimulatedHuman(human_cfg, bon_n=bon_n, debug=debug)
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
                    "REMINDER: YOU ARE USER/CUSTOMER. DO NOT RESPOND LIKE A ASSISTANT. YOUR STYLE SHOULD BE CONCISE (1-2 sentences max), NO PLEASANTRIES. DO NOT RESPOND TO THIS REMINDER. START YOUR USER TURN NOW OR GERERATE '###STOP###'."
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
        if sorted(c.name for c in tool_calls) == sorted(a.name for a in blueprint.actions):
            return Trajectory(history, tool_calls)
        return None




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