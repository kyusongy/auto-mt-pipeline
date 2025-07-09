from __future__ import annotations

"""Blueprint generation & validation prototype.

This module implements a minimal pipeline covering the *first* phase of
APIGen-MT (task blueprint generation & validation).
The goal is to demonstrate how the pieces fit together while re-using the
existing LLM helper utilities already available in the repository.

High-level flow
---------------
1. *BlueprintGenerator* – single LLM call that produces a blueprint JSON
   (user intent + actions + expected outputs) based on the provided API
   schema, business policies and a randomly chosen user persona.
2. *BlueprintValidator* – light-weight, rule-based checks that make sure the
   blueprint structure is sound and that every tool call can execute against
   the API schema.  (No real DB/state changes are performed – we only do
   static validation.)
3. *ReviewCommittee* – a small committee of LLM reviews that rate the
   blueprint on correctness/completeness/etc. Majority voting decides
   pass/fail and produces feedback when it fails.
4. *generate_valid_blueprint()* – orchestration helper that keeps iterating
   (generate ➜ validate ➜ review ➜ feedback) until the blueprint passes or
   the maximum number of attempts is reached.

This is *only a prototype*: many aspects (e.g. policy engine, execution
environment) are simplified or stubbed out – the focus is to provide a clear
end-to-end skeleton you can extend.
"""

# Standard library imports
from dataclasses import dataclass
import json
import random
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
from openai.types.chat import ChatCompletionMessageParam

# Local application imports
from config import (
    GenerationOptions as LLMGenerationOptions,
    LLMConfig,
    BLUEPRINT_GENERATION_OPTIONS,
    BLUEPRINT_COMMITTEE_OPTIONS,
)
from core.models import ToolCalling
from core.llm_client import sync_request_llm

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Blueprint:
    """Simple container for a task blueprint."""

    user_intent: str
    actions: List[ToolCalling]
    expected_outputs: List[Any]
    raw_response: str | None = None  # original LLM output for debugging


# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------


def _build_generator_prompt(
    persona: str,
    tools_schema: Dict[str, Any],
    *,
    task_rules: str = "",
    domain_rules: str = "",
    sampled_user_details: str = "",
    sampled_orders: str = "",
    examples: str = "",
    prev_feedback: str = "",
    api_dependencies: str = "",
) -> List[ChatCompletionMessageParam]:
    """Create **one** combined prompt string following Figure-8 of the paper.

    This returns a single-item messages list (role="user") because the original
    article shows the whole prompt as one block rather than split into system
    / user messages.
    """

    prompt = textwrap.dedent(
        f"""
        ## Instructions
        Generate a task instruction that mimics realistic human users and their intentions, such as with different personality and goals. The task instruction should be
        followed by `actions` which is a list of the tool_calls to be taken to solve this task and `outputs` which is a list of the answers to specific information requests made
        by the user. Think step by step to come up with the action(s) and the corresponding tool_call(s) translating this thought that would be necessary to fulfil the user's
        request or solve their intentions. Focus on common Lenovo customer scenarios following the provided task instruction guidelines.

        ## Guidelines for Generating Task Instruction (q)
        {domain_rules}
        
        ## Guidelines for generating Groundtruth Actions (a_g t)
        1.  The main focus is to generate actions that help users with Lenovo products and services.
        2.  For actions that provide information requests, use appropriate tools like product_recommend, product_knowledge_retrieval, etc.
        3.  Include multiple tool calls when the scenario requires comprehensive assistance (e.g., product recommendation + parameter comparison).
        4.  Provide precise tool calls with all necessary parameters for each action.
        5.  Ensure all actions adhere to Lenovo service policies and help users make informed decisions.
        6.  **Tool Chaining & Dependencies**: Some tools require outputs from previous tools as inputs:
            - product_params_compare needs 'product_ids_to_compare' which comes from product_recommend output
            - When creating multi-step workflows, structure actions in the correct order
            - For dependent tools, use placeholder values that represent the expected output format
            - Example: product_recommend → extract SKU IDs → product_params_compare with those IDs

        ## API Dependencies
        {api_dependencies}

        ## Tools
        The available tool combination in Python format is as follows:
        {json.dumps(tools_schema, ensure_ascii=False, indent=2)}

        ## Output Format
        Generate your response according to the following format. Enclose the thought process within `<thought></thought>` tags, and the final structured response within
        `<answer></answer>` tags. The structured response should be in strict JSON format, without any additional comments or explanations.


        ## Example Tasks
        {examples}

        ## Feedback from previous attempt
        {prev_feedback}

        Do not directly copy instruction and the action patterns from the examples. Ground the generation from the above provided data.
        Generate the task now.
        """
    ).strip()

    return [{"role": "user", "content": prompt}]


# ---------------------------------------------------------------------------
# Helper to robustly extract JSON from LLM outputs
# ---------------------------------------------------------------------------


def _extract_json_block(text: str) -> str:
    """Return a clean JSON string by stripping <answer>/<scores> tags and ``` fences."""
    cleaned = text.strip()

    # 1) Remove complete <tag>...</tag> blocks (handles well-formed tags)
    for tag in ("answer", "scores"):
        m = re.search(fr"<\s*{tag}\s*>(.*?)<\s*/{tag}\s*>", cleaned, re.S | re.I)
        if m:
            cleaned = m.group(1).strip()

    # 2) Remove any stray opening / closing tags left behind (for cases where the LLM forgets the closing tag).
    cleaned = re.sub(r"<\s*/?\s*(answer|scores)\s*>", "", cleaned, flags=re.I)

    # 3) Remove leading / trailing fenced code blocks.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.I)

    # 4) Heuristically grab the first JSON object if extraneous text remains.
    brace_match = re.search(r"\{.*\}", cleaned, re.S)
    if brace_match:
        cleaned = brace_match.group(0)

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Blueprint generator
# ---------------------------------------------------------------------------


class BlueprintGenerator:
    """Generates a blueprint via a single LLM call."""

    def __init__(self, llm_config: LLMConfig, gen_opts: Optional[LLMGenerationOptions] = None):
        self.llm_config = llm_config
        # Use centralized default if caller does not provide custom options
        self.gen_opts = gen_opts or BLUEPRINT_GENERATION_OPTIONS

    def generate(
        self,
        persona: str,
        tools_schema: Dict[str, Any],
        **prompt_kwargs: Any,
    ) -> Blueprint:
        """Generate a blueprint. Additional template fields can be passed via kwargs."""

        messages = _build_generator_prompt(
            persona,
            tools_schema,
            task_rules=prompt_kwargs.get("task_rules", ""),
            domain_rules=prompt_kwargs.get("domain_rules", ""),
            sampled_user_details=prompt_kwargs.get("sampled_user_details", ""),
            sampled_orders=prompt_kwargs.get("sampled_orders", ""),
            examples=prompt_kwargs.get("examples", ""),
            prev_feedback=prompt_kwargs.get("prev_feedback", ""),
            api_dependencies=prompt_kwargs.get("api_dependencies", ""),
        )
        completion = sync_request_llm(
            self.llm_config,
            messages,
            tools=None,
            generation_config=self.gen_opts,
        )
        raw_content = completion.choices[0].message.content  # type: ignore[attr-defined]
        if self.gen_opts.debug:
            print("\n----- Generator raw output -----\n", raw_content, "\n-------------------------------\n")
        if not isinstance(raw_content, str):
            raise ValueError("Expected string content from LLM response")

        # Extract content from <answer> tag, falling back to full response
        json_content = _extract_json_block(raw_content)

        # Parse JSON
        try:
            bp_dict = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Blueprint is not valid JSON: {e}\nRaw: {json_content}") from e

        intent = bp_dict.get("intent") or bp_dict.get("q") or ""
        actions_raw: List[Dict[str, Any]] = bp_dict.get("actions", [])
        outputs = bp_dict.get("outputs") or bp_dict.get("o_g_t") or []

        actions: List[ToolCalling] = []
        for a_raw in actions_raw:
            tool_name = a_raw.get("name") or a_raw.get("tool_call")
            tool_args = a_raw.get("arguments") or a_raw.get("parameters")
            if tool_name and tool_args is not None:
                actions.append(ToolCalling(name=tool_name, arguments=tool_args))

        return Blueprint(intent, actions, outputs, raw_response=json_content)


# ---------------------------------------------------------------------------
# Validator (format & static execution)
# ---------------------------------------------------------------------------


class BlueprintValidator:
    """Performs simple structural & static execution checks."""

    def __init__(self, tools_schema: Dict[str, Any]):
        self.schema = tools_schema

    # ---------- helpers ----------
    def _check_action_against_schema(self, action: ToolCalling) -> Tuple[bool, str]:
        if action.name not in self.schema:
            return False, f"Unknown tool '{action.name}'"
        spec = self.schema[action.name]
        allowed_args = set(spec.get("parameters", {}).keys())
        given_args = set((action.arguments or {}).keys())
        unknown = given_args - allowed_args
        if unknown:
            return False, f"Unknown argument(s) {unknown} for tool '{action.name}'"
        missing = {
            k for k, v in spec.get("parameters", {}).items() if v.get("required", False) and k not in given_args
        }
        if missing:
            return False, f"Missing required argument(s) {missing} for tool '{action.name}'"
        return True, "OK"

    # ---------- main API ----------
    def validate(self, bp: Blueprint) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not bp.user_intent:
            errors.append("Empty intent")
        if not bp.actions:
            errors.append("No actions specified")
        for idx, act in enumerate(bp.actions):
            ok, msg = self._check_action_against_schema(act)
            if not ok:
                errors.append(f"Action #{idx}: {msg}")
        return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Review Committee
# ---------------------------------------------------------------------------


_REVIEW_PROMPT_TEMPLATE = """
You are an AI judge and your goal is to judge the quality and validity of the provided task object based on the guidelines, following the rubric.

## Guidelines
• The task object contains an `intent` (q) from a user, `actions` (a_g t), and `outputs` (o_g t).
• The `actions` correspond to the tool_calls made by an AI assistant to satisfy the instruction.
• A description of the `tools` available to the AI assistant is provided.
• Perform a brief reflection on the task based on the below Rubrics.
• Think step-by-step to generate a score of 0 or 1 for each of these criteria (1 means follows criterion and 0 means does not)

## Rubric
• Correctness: Do the actions (a_g t) accurately implement the instruction (q)?
• Completeness: Is the instruction (q) sufficiently detailed, and is it fully addressed by the actions? (Includes rule-based checks).
• Satisfaction: Do the expected outputs (o_g t) fulfil any explicit or implicit information requests within the instruction (q)?
• Creativity: Does the task represent a non-trivial, plausible, and potentially interesting scenario within the domain?

## Task Object
{task}

## Tools in Python format
{tools}


## Output format
<scores>
{{
    "reflection": str,
    "correctness": int(0/1),
    "completeness": int(0/1),
    "satisfaction": int(0/1),
    "creativity": int(0/1),
    "total": int(0-4),
    "correction": str
}}
""".strip()


class ReviewCommittee:
    """Queries multiple judge LLMs and uses majority voting."""

    def __init__(
        self,
        llm_config: LLMConfig,
        size: int = 3,
        pass_threshold: float = 3.0,
        gen_opts: Optional[LLMGenerationOptions] = None,
        tools_schema: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = llm_config
        self.size = size
        self.pass_threshold = pass_threshold
        # Use centralized committee options as default for blueprint review
        self.gen_opts = gen_opts or BLUEPRINT_COMMITTEE_OPTIONS
        self.tools_schema = tools_schema or {}

    def _build_messages(self, bp: Blueprint) -> List[ChatCompletionMessageParam]:
        task_json = json.dumps(
            {
                "intent": bp.user_intent,
                "actions": [a.model_dump() for a in bp.actions],
                "outputs": bp.expected_outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
        tools_json = json.dumps(self.tools_schema, ensure_ascii=False, indent=2)
        prompt = _REVIEW_PROMPT_TEMPLATE.format(
            task=task_json,
            tools=tools_json,
        )
        return [{"role": "user", "content": prompt}]

    def review(self, bp: Blueprint) -> Tuple[bool, List[Dict[str, Any]]]:
        messages = self._build_messages(bp)
        votes: List[Dict[str, Any]] = []
        passes = 0
        for _ in range(self.size):
            comp = sync_request_llm(self.cfg, messages, generation_config=self.gen_opts)
            raw = comp.choices[0].message.content  # type: ignore[attr-defined]
            reply = raw if isinstance(raw, str) else ""
            if self.gen_opts.debug:
                print("\n----- Judge raw reply -----\n", reply, "\n---------------------------\n")
            try:
                data = json.loads(_extract_json_block(reply)) if reply.strip() else {"total": 0}
            except json.JSONDecodeError:
                data = {"total": 0, "correction": "Malformed judge output"}
            votes.append(data)
            if data.get("total", 0) >= self.pass_threshold:
                passes += 1
        majority_pass = passes > self.size // 2
        return majority_pass, votes


# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------


def generate_valid_blueprint(
    llm_cfg: LLMConfig,
    tools_schema: Dict[str, Any],
    personas: List[str],
    max_attempts: int = 5,
    prompt_kwargs: Optional[Dict[str, Any]] = None,
) -> Blueprint:
    """Generate-validate-review loop until a blueprint is accepted or we give up."""

    # Use centralized generator options (enable debug for visibility)
    generator_opts = BLUEPRINT_GENERATION_OPTIONS.model_copy(update={"debug": True})
    # Use centralized committee options for blueprint review (enable debug for visibility)
    committee_opts = BLUEPRINT_COMMITTEE_OPTIONS.model_copy(update={"debug": True})

    generator = BlueprintGenerator(llm_cfg, gen_opts=generator_opts)
    validator = BlueprintValidator(tools_schema)
    committee = ReviewCommittee(llm_cfg, gen_opts=committee_opts, tools_schema=tools_schema)

    prev_feedback = ""  # aggregated feedback from validator & committee

    for attempt in range(1, max_attempts + 1):
        persona = random.choice(personas)
        try:
            bp = generator.generate(
                persona,
                tools_schema,
                **{**(prompt_kwargs or {}), "prev_feedback": prev_feedback},
            )
        except ValueError as e:
            print(f"[Attempt {attempt}] ❌ Generation failed: {e}")
            # pass the error message as feedback to nudge next attempt
            prev_feedback = str(e)
            continue

        ok_struct, errs = validator.validate(bp)
        if not ok_struct:
            print(f"[Attempt {attempt}] ❌ Validation failed: {errs}")
            # join validation errors as feedback
            prev_feedback = "; ".join(errs)
            continue

        ok_review, votes = committee.review(bp)
        if ok_review:
            print(f"[Attempt {attempt}] ✅ Blueprint accepted")
            return bp

        # collect corrections from committee votes for next round
        corrections = [v.get("correction", "") for v in votes if v.get("correction")]
        prev_feedback = " | ".join(corrections)
        print(f"[Attempt {attempt}] ❌ Committee rejected. Feedback: {prev_feedback}")

    raise RuntimeError(
        f"Failed to generate a valid blueprint after {max_attempts} attempts. Last feedback: {prev_feedback}"
    )


 