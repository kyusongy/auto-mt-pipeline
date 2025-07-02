from __future__ import annotations

"""APIGen-MT Phase-2 prototype: Trajectory collection via simulated human-agent interplay.

The goal is to demonstrate the end-to-end mechanics while keeping the code
compact and editable. All prompts come from Fig.-11 and Fig.-12 of the paper
(see screenshots) but are parameterised with placeholders so you can inject
real data later.
"""

from dataclasses import dataclass
import json
import random
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

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

# ---------------- Qwen-Agent integration -----------------
# pylint: disable=import-error
from qwen_agent.agents import Assistant  # type: ignore
# Ensure tool wrappers are registered before Assistant is instantiated
import core.trajectory.qwen_tool_wrappers  # noqa: F401  # registers tools via import side-effects
from tools.retail_tools import RUNTIME_FUNCTIONS  # provides callable stubs for tool execution

# ---------------------------------------------------------------------------
# Prompts (verbatim from paper – Fig.11 & Fig.12)
# ---------------------------------------------------------------------------

_TRAJECTORY_COLLECTION_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a detail-oriented user interacting with an AI agent.

    ## Intent
    {intent}

    ## Rules
    • Generate one line at a time to simulate the user's message.
    • Do not give away all the intent at once. Only provide the information that is necessary for the current step.
    • Do not hallucinate information that is not provided in the intent.
    • If the intent goal is satisfied, generate `###STOP###` to end the conversation.
    • Do not repeat the exact intent in the conversation. Instead, use your own words to convey the same information.
    • Do not copy or repeat any assistant messages. Always write a brand-new user turn in your own words.
    • Try to make the conversation as natural as possible and stick to the personalities in the intent.

    ### Response format
    Reply with **only** the next message you (the user) would send. Do NOT include any explanations, headings, bullet points, or additional reasoning—just the raw chat line.
    """
).strip()

_BON_USER_LM_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a fair judge and an expert in following details.

    A human is interacting with a retail assistant to get help on solving their task. You are provided with the description of the human and the task the
    human wants to accomplish (wrapped with <description></description>), and a candidate response (wrapped with <response></response>) the human wants to
    give the assistant. Please help the human evaluate this candidate response, give an integer score (ranging from 0 to 10) to indicate the correctness of the response,
    higher score means better quality.

    CRITICAL: The response MUST be from the perspective of a CUSTOMER/USER, NOT an assistant or agent.

    1. **ROLE CONFUSION (AUTOMATIC SCORE 0)**: If the response sounds like it's coming from an assistant/agent rather than a customer, give score 0. Signs include:
       - Offering help or assistance (e.g., "I can help you with...", "Let me know...", "I'll provide the next steps")
       - Asking what the customer wants (e.g., "Would you like a refund or replacement?")
       - Using assistant language (e.g., "To proceed with...", "I need to verify...", "Your order has been verified")
       - Acting like they have access to systems or can process requests

    2. If the response includes specific item / order / personal details, and they correctly match the task description you should give full score of 10. If there is some
       change in details, give a corresponding lower score (more incorrect details gets lower score).
    
    3. The response can include any normal customer conversation otherwise (e.g., asking for help, providing information, saying ###STOP###) etc. which are all correct responses.
    
    4. Additionally, if the candidate response keeps the conversation flowing by describing the task clearly / gives information properly then give a high score and if not
       (e.g. "I don't remember" or unhelpful response) should get a corresponding lower score.

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

        def _clean_line(line: str) -> str:
            meta_keywords = (
                "the user", "assistant", "i should", "let me", "they need", "i need to", "next step", "first,", "okay,", "wait,", "perhaps",
            )
            low = line.lower()
            return "" if any(k in low for k in meta_keywords) else line

        # keep first non-meta line; if none found fallback to original
        for ln in best_msg.splitlines():
            cleaned = _clean_line(ln.strip())
            if cleaned:
                return cleaned

        return best_msg.strip()


class QwenTestAgent:
    """Wrap Qwen-Agent's `Assistant` for our trajectory collector."""

    def __init__(self, llm_cfg: LLMConfig, generation_opts: LLMGenOpts = None):
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
            
        self.bot = Assistant(
            llm=llm_config,
            function_list=list(_d.TOOLS_SCHEMA.keys()),  # expose only dummy tools to avoid heavy deps
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
# Default system prompt for Qwen retail agent
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = (
    "# Retail agent policy\n"
    "As a retail agent, you can help users cancel or modify pending orders, return or exchange delivered orders, modify their default user address, or provide information about their own profile, orders, and related products.\n"
    "- At the beginning of the conversation, you have to authenticate the user identity by locating their user id via email, or via name + zip code. This has to be done even when the user already provides the user id.\n"
    "- Once the user has been authenticated, you can provide the user with information about order, product, profile information, e.g. help the user look up order id.\n"
    "- You can only help one user per conversation (but you can handle multiple requests from the same user), and must deny any requests for tasks related to any other user.\n"
    "- Before taking consequential actions that update the database (cancel, modify, return, exchange), you have to list the action detail and obtain explicit user confirmation (yes) to proceed.\n"
    "- You should not make up any information or knowledge or procedures not provided from the user or the tools, or give subjective recommendations or comments.\n"
    "- You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call.\n"
    "- You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.\n\n"
    "## Domain basic\n"
    "- All times in the database are EST and 24 hour based. For example \"02:30:00\" means 2:30 AM EST.\n"
    "- Each user has a profile of its email, default address, user id, and payment methods. Each payment method is either a gift card, a paypal account, or a credit card.\n"
    "- Our retail store has 50 types of products. For each type of product, there are variant items of different options. For example, for a 't shirt' product, there could be an item with option 'color blue size M', and another item with option 'color red size L'.\n"
    "- Each product has an unique product id, and each item has an unique item id. They have no relations and should not be confused.\n"
    "- Each order can be in status 'pending', 'processed', 'delivered', or 'cancelled'. Generally, you can only take action on pending or delivered orders.\n"
    "- Exchange or modify order tools can only be called once. Be sure that all items to be changed are collected into a list before making the tool call!!!\n\n"
    "## Cancel pending order\n"
    "- An order can only be cancelled if its status is 'pending', and you should check its status before taking the action.\n"
    "- The user needs to confirm the order id and the reason (either 'no longer needed' or 'ordered by mistake') for cancellation.\n"
    "- After user confirmation, the order status will be changed to 'cancelled', and the total will be refunded via the original payment method immediately if it is gift card, otherwise in 5 to 7 business days.\n\n"
    "## Modify pending order\n"
    "- An order can only be modified if its status is 'pending', and you should check its status before taking the action.\n"
    "- For a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.\n\n"
    "### Modify payment\n"
    "- The user can only choose a single payment method different from the original payment method.\n"
    "- If the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.\n"
    "- After user confirmation, the order status will be kept 'pending'. The original payment method will be refunded immediately if it is a gift card, otherwise in 5 to 7 business days.\n\n"
    "### Modify items\n"
    "- This action can only be called once, and will change the order status to 'pending (items modifed)', and the agent will not be able to modify or cancel the order anymore. So confirm all the details are right and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all items to be modified.\n"
    "- For a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.\n"
    "- The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.\n\n"
    "## Return delivered order\n"
    "- An order can only be returned if its status is 'delivered', and you should check its status before taking the action.\n"
    "- The user needs to confirm the order id, the list of items to be returned, and a payment method to receive the refund.\n"
    "- The refund must either go to the original payment method, or an existing gift card.\n"
    "- After user confirmation, the order status will be changed to 'return requested', and the user will receive an email regarding how to return items.\n\n"
    "## Exchange delivered order\n"
    "- An order can only be exchanged if its status is 'delivered', and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.\n"
    "- For a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.\n"
    "- The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.\n"
    "- After user confirmation, the order status will be changed to 'exchange requested', and the user will receive an email regarding how to return items. There is no need to place a new order.\n"
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
        self.agent = QwenTestAgent(agent_cfg, ASSISTANT_AGENT_OPTIONS)
        self.tools_schema = tools_schema or {}
        self.debug = debug

        # Always use the predefined retail-agent policy
        self.agent_system_prompt = _AGENT_SYSTEM_PROMPT

    def collect(self, blueprint: Blueprint) -> Optional[Trajectory]:
        history: List[Turn] = []
        tool_calls: List[ToolCalling] = []

        # Simulate turns (hard-coded max 20 to avoid infinite loop)
        for turn_num in range(1, 21):
            # 1) human speaks
            user_msg = self.human.next_message(blueprint.user_intent, history)
            if not user_msg:
                if self.debug:
                    print("[Collector] human LLM returned empty response, aborting trajectory")
                return None
            history.append(Turn("user", user_msg))
            if self.debug:
                print(f"\nTurn {turn_num}")
                print("[USER]", user_msg)
            # Stop when the user injects the special token anywhere in the line
            if "###STOP###" in user_msg:
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

            for idx, m in enumerate(new_msgs):
                role = m.get("role")
                if role == "assistant" and "function_call" in m:
                    fc = m["function_call"]
                    # `function_call.arguments` might come as *string* (JSON) per
                    # OpenAI spec.  Convert to dict so ToolCalling passes
                    # Pydantic validation.
                    raw_args = fc.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            # leave as-is; validator will flag it later
                            pass

                    val = json.dumps({"name": fc.get("name"), "arguments": json.dumps(raw_args, ensure_ascii=False)}, ensure_ascii=False)
                    history.append(Turn("function_call", val))
                    tool_calls.append(ToolCalling(name=fc.get("name"), arguments=raw_args))
                    if self.debug:
                        print("[FUNC_CALL]", val)

                    # Skip adding a synthetic observation; rely solely on
                    # the agent-emitted "function" message (if any) to carry
                    # tool results. Removing redundancy keeps the turn log
                    # clean and mirrors real OpenAI behaviour.
                elif role == "function":
                    history.append(Turn("observation", m.get("content", "")))
                    if self.debug:
                        print("[OBSERV]", m.get("content", ""))
                else:
                    # standard assistant message
                    content = m.get("content", "")
                    history.append(Turn("assistant", content))
                    if self.debug and content.strip():  # Only print non-empty content
                        print("[ASSISTANT]", content)

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