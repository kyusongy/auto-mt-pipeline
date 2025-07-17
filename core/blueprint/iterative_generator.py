"""Iterative blueprint generation with action validation.

This module implements the new iterative blueprint generation process:
1. LLM generates intent + actions, validated by original checks + review committee
2. ActionExecutor executes the validated actions  
3. ExecutionReviewer reviews results and either modifies actions OR approves completely
4. Loop until approved, then proceed to trajectory collection
"""

from __future__ import annotations

import json
import random
import textwrap
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletionMessageParam

from business_components.workflow import Context, Workflow, WorkflowConfig
from agent_types.common import Plan
from agent_types.common import ToolCalling as AgentCortexToolCalling
from config import (LLMConfig, GenerationOptions, BLUEPRINT_GENERATION_OPTIONS, agentcortex_config)
from core.llm_client import sync_request_llm
from core.mcp_client import MCPClient
from core.models import ToolCalling

from .action_modifier import ActionModifier
from .execution_reviewer import ExecutionReviewer, ReviewDecision
from .output_generator import OutputGenerator
from .pipeline import (
    Blueprint, BlueprintValidator, ReviewCommittee, _extract_json_block, _extract_thought_block
)


@dataclass
class ExecutionResult:
    """Container for tool execution results - compatible with blueprint pipeline."""
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class ActionExecutionSummary:
    """Summary of all action executions - compatible with blueprint pipeline."""
    results: List[ExecutionResult]
    total_actions: int
    successful_actions: int
    failed_actions: int
    dependency_chain: List[str]  # Order of execution


class Stage1Generator:
    """Generates ONLY intent + actions (no outputs) for Stage 1 validation."""

    def __init__(self, llm_config: LLMConfig, gen_opts: Optional[GenerationOptions] = None):
        self.llm_config = llm_config
        self.gen_opts = gen_opts or BLUEPRINT_GENERATION_OPTIONS

    def _build_stage1_prompt(
        self,
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
        """Create prompt for Stage 1: intent + actions only (NO outputs)."""

        prompt = textwrap.dedent(
            f"""
        ## Instructions
        Generate a task instruction that mimics realistic human users and their intentions, such as with different personality and goals. The task instruction should be
        followed by `actions` which is a list of the tool_calls to be taken to solve this task. Think step by step to come up with the action(s) and the corresponding 
        tool_call(s) translating this thought that would be necessary to fulfil the user's request or solve their intentions. Focus on common Lenovo customer scenarios 
        following the provided task instruction guidelines.

        ## Persona
        {persona}

        ## Guidelines for Generating Task Instruction
        {domain_rules}
        
        ## Guidelines for generating Groundtruth Actions
        1.  The main focus is to generate actions that help users with Lenovo products and services.
        2.  For actions that provide information requests, use appropriate tools like product_recommend, product_knowledge_retrieval, etc.
        3.  Include multiple tool calls when the scenario requires comprehensive assistance.
        4.  Ensure all actions adhere to Lenovo service policies and help users make informed decisions.
        
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
            """).strip()

        return [{"role": "user", "content": prompt}]

    def generate(
        self,
        persona: str,
        tools_schema: Dict[str, Any],
        **prompt_kwargs: Any,
    ) -> Blueprint:
        """Generate a blueprint with intent + actions only (no outputs)."""

        messages = self._build_stage1_prompt(
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
        
        raw_content = completion.choices[0].message.content
        if self.gen_opts.debug:
            print("\n----- Stage1 Generator raw output -----\n", raw_content, "\n-------------------------------\n")
        if not isinstance(raw_content, str):
            raise ValueError("Expected string content from LLM response")

        # Extract content from <answer> tag
        json_content = _extract_json_block(raw_content)
        thought_process = _extract_thought_block(raw_content)

        # Parse JSON
        try:
            bp_dict = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Stage 1 blueprint is not valid JSON: {e}\nRaw: {json_content}") from e

        intent = bp_dict.get("intent", "")
        if not intent:
            raise ValueError("No intent found in Stage 1 response")
            
        actions_raw: List[Dict[str, Any]] = bp_dict.get("actions", [])
        if not actions_raw:
            raise ValueError("No actions found in Stage 1 response")

        actions: List[ToolCalling] = []
        for a_raw in actions_raw:
            tool_name = a_raw.get("name")
            tool_args = a_raw.get("arguments")
            if tool_name and tool_args is not None:
                actions.append(ToolCalling(name=tool_name, arguments=tool_args))

        if not actions:
            raise ValueError("No valid actions parsed from Stage 1 response")

        # Return blueprint with empty outputs (will be filled by Stage 2)
        return Blueprint(intent, actions, expected_outputs=[], raw_response=json_content, thought_process=thought_process)


class IterativeBlueprintGenerator:
    """Generates validated blueprints through the new two-stage validation process."""

    def __init__(
        self, 
        llm_config: LLMConfig, 
        mcp_client: MCPClient,
        tools_schema: Dict[str, Any],
        gen_opts: Optional[GenerationOptions] = None,
        debug: bool = False
    ):
        self.llm_config = llm_config
        self.mcp_client = mcp_client
        self.tools_schema = tools_schema
        self.gen_opts = gen_opts or BLUEPRINT_GENERATION_OPTIONS.model_copy(update={"debug": debug})
        self.debug = debug
        
        # Initialize components - use Stage1Generator instead of original BlueprintGenerator
        self.stage1_generator = Stage1Generator(llm_config, gen_opts)
        self.validator = BlueprintValidator(tools_schema)
        self.committee = ReviewCommittee(llm_config, gen_opts=gen_opts, tools_schema=tools_schema)
        
        # Initialize AgentCortex Workflow
        if agentcortex_config:
            wf_config = WorkflowConfig(
                session_memory_url=agentcortex_config.session_memory_url,
                system_memory_url=agentcortex_config.system_memory_url,
                intent_url=agentcortex_config.intent_url,
                planning_url=agentcortex_config.planning_url,
                execution_url=agentcortex_config.execution_url,
                summarization_url=agentcortex_config.summarization_url,
                max_iterations=agentcortex_config.max_iterations,
                extract_mentions_url=agentcortex_config.extract_mentions_url,
                personalization_url=agentcortex_config.personalization_url,
            )
            self.workflow = Workflow(wf_config)
        else:
            raise RuntimeError("AgentCortex config is not available, cannot initialize Workflow.")

        self.execution_reviewer = ExecutionReviewer(llm_config, gen_opts)
        self.action_modifier = ActionModifier(llm_config, gen_opts or BLUEPRINT_GENERATION_OPTIONS)
        self.output_generator = OutputGenerator(llm_config, gen_opts or BLUEPRINT_GENERATION_OPTIONS)

    def generate_validated_blueprint(
        self,
        persona: str,
        max_generation_attempts: int = 5,
        max_execution_iterations: int = 3,
        **prompt_kwargs: Any
    ) -> Blueprint:
        """Generate a validated blueprint through the two-stage process.
        
        Stage 1: Generate intent + actions with original validation
        Stage 2: Execute actions and iteratively refine until approved
        
        Args:
            persona: User persona for blueprint generation
            max_generation_attempts: Max attempts for Stage 1 (intent+actions generation)
            max_execution_iterations: Max iterations for Stage 2 (execution validation)
            **prompt_kwargs: Additional prompt context (domain_rules, api_dependencies, etc.)
            
        Returns:
            Blueprint with validated actions and execution-based outputs
        """
        if self.debug:
            print(f"🔄 IterativeBlueprintGenerator: Two-stage validation process")
        
        # Store original context for execution review
        original_context = {
            **prompt_kwargs,
            "tools_schema": self.tools_schema,
            "persona": persona
        }
        
        # ========================================================================
        # Stage 1: Generate and validate intent + actions (original method)
        # ========================================================================
        
        if self.debug:
            print(f"\n📝 Stage 1: Intent + Actions Generation (max {max_generation_attempts} attempts)")
            print("=" * 60)
        
        prev_feedback = ""
        validated_intent = None
        validated_actions = None
        validated_thought_process = None
        
        for attempt in range(1, max_generation_attempts + 1):
            if self.debug:
                print(f"\n🔄 Generation attempt {attempt}/{max_generation_attempts}")
            
            try:
                # Generate initial blueprint (intent + actions ONLY, no outputs)
                blueprint = self.stage1_generator.generate(
                    persona,
                    self.tools_schema,
                    prev_feedback=prev_feedback,
                    **prompt_kwargs
                )
                
                if self.debug:
                    print(f"📝 Generated intent: {blueprint.user_intent}")
                    print(f"🛠️  Generated {len(blueprint.actions)} actions: {[a.name for a in blueprint.actions]}")
                
            except ValueError as e:
                if self.debug:
                    print(f"❌ Generation failed: {e}")
                prev_feedback = str(e)
                continue

            # Validate structure
            ok_struct, errs = self.validator.validate(blueprint)
            if not ok_struct:
                if self.debug:
                    print(f"❌ Validation failed: {errs}")
                prev_feedback = "; ".join(errs)
                continue

            # Review with committee
            ok_review, votes = self.committee.review(blueprint)
            if ok_review:
                if self.debug:
                    print(f"✅ Intent + Actions approved by committee")
                validated_intent = blueprint.user_intent
                validated_actions = blueprint.actions
                validated_thought_process = blueprint.thought_process or ""
                break
            else:
                # Collect corrections from committee votes for next round
                corrections = [v.get("correction", "") for v in votes if v.get("correction")]
                prev_feedback = " | ".join(corrections)
                if self.debug:
                    print(f"❌ Committee rejected. Feedback: {prev_feedback}")

        if validated_intent is None or validated_actions is None:
            raise RuntimeError(
                f"Failed to generate valid intent + actions after {max_generation_attempts} attempts. "
                f"Last feedback: {prev_feedback}"
            )
        
        # ========================================================================
        # Stage 2: Execute actions and iteratively refine until approved
        # ========================================================================
        
        if self.debug:
            print(f"\n🔧 Stage 2: Action Execution Validation (max {max_execution_iterations} iterations)")
            print("=" * 60)
        
        current_actions = validated_actions
        
        for iteration in range(1, max_execution_iterations + 1):
            if self.debug:
                print(f"\n🔄 Execution iteration {iteration}/{max_execution_iterations}")

            # Execute current actions using the agentcortex-lsa workflow
            try:
                # Treat each execution iteration as a new session
                session_id = str(uuid.uuid4())

                # Create context with default arguments, similar to agentcortex-lsa's test runner
                default_args = {
                    "user_info": {"uid": "10208390957", "user_identity": 1, "available_num": 0.0, "current_amount": "0", "enterprise_name": "", "future_expire_num": 0.0, "level_name": "", "entry_source": "shop", "user_province": "北京"},
                    "trace_id": session_id,
                    "uid": "10208390957",
                    "terminal": "1",
                    "latitude": "23.89447712420573",
                    "longitude": "106.6172117534938",
                    "device_ip": "117.183.16.69",
                    "get_position_permission": "agree",
                    "event": "",
                    "bind_mobile_id": 0,
                }
                context = Context(session_id=session_id, query=validated_intent, tools=self.workflow.tools, default_args=default_args)

                # Initialize context with proper workflow initialization steps
                # This mimics the initialization phase from pipeline.py and workflow.py
                # These steps are crucial for proper tool execution as they:
                # 1. Load session memory for context awareness
                # 2. Rewrite query to improve understanding and tool parameter extraction
                # 3. Extract mentions (product IDs, categories, etc.) for tool parameters
                # 4. Extract user preferences for personalized responses
                if self.debug:
                    print(f"🔧 Initializing context for execution...")
                
                # Step 1: Read session memory
                self.workflow.read_session_memory(context)
                
                # Step 2: Rewrite query for better understanding
                self.workflow.rewrite_query(context)
                
                # Step 3: Extract mentions from the rewritten query
                self.workflow.read_mentions(context)
                
                # Step 4: Extract session preferences
                self.workflow.read_session_preference(context)
                
                if self.debug:
                    print(f"🔧 Context initialized with:")
                    print(f"   - Rewritten query: {context.rewrite_query}")
                    print(f"   - Mentions: {len(context.session_memory.mentions) if context.session_memory and context.session_memory.mentions else 0}")
                    print(f"   - Session preference: {context.session_memory.session_preference if context.session_memory else None}")

                # Convert actions to agentcortex Plan
                agentcortex_actions = [
                    AgentCortexToolCalling(name=a.name, arguments=a.arguments or {})
                    for a in current_actions
                ]
                context.plan = Plan(tool_callings=agentcortex_actions, content="")

                # Execute the plan via the workflow
                self.workflow.execute(context)

                # Convert observations back to ActionExecutionSummary
                results = []
                dependency_chain = []
                for i, status in enumerate(context.observations[0].status):
                    action = current_actions[i]
                    if status.error:
                        result = ExecutionResult(
                            tool_name=action.name,
                            arguments=action.arguments or {},
                            success=False,
                            result=None,
                            error=status.error.message,
                        )
                    else:
                        result = ExecutionResult(
                            tool_name=action.name,
                            arguments=action.arguments or {},
                            success=True,
                            result=status.result,
                            error=None,
                        )
                    results.append(result)
                    dependency_chain.append(action.name)
                
                successful = sum(1 for r in results if r.success)
                execution_summary = ActionExecutionSummary(
                    results=results,
                    total_actions=len(current_actions),
                    successful_actions=successful,
                    failed_actions=len(current_actions) - successful,
                    dependency_chain=dependency_chain,
                )

                if self.debug:
                    print(f"🔧 Execution completed: {execution_summary.successful_actions}/{execution_summary.total_actions} successful")

            except Exception as e:
                if self.debug:
                    print(f"❌ Execution failed: {e}")
                if iteration == max_execution_iterations:
                    raise RuntimeError(f"Action execution failed: {str(e)}")
                continue
            
            # Review execution results and decide: modify actions OR approve
            try:
                review_decision = self._review_execution_with_modification_option(
                    validated_intent, current_actions, validated_thought_process or "", execution_summary, original_context
                )
                
                if self.debug:
                    print(f"📋 Review decision: {'✅ APPROVED' if review_decision.approved else '🔄 MODIFY ACTIONS'}")
                    print(f"📊 Confidence: {review_decision.confidence_score:.2f}")
                
                if review_decision.approved:
                    # Generate final outputs using OutputGenerator
                    try:
                        final_outputs = self.output_generator.generate_outputs(
                            validated_intent, current_actions, validated_thought_process or "", 
                            execution_summary, original_context
                        )
                        
                        if self.debug:
                            print(f"📋 Generated {len(final_outputs)} final outputs")
                        
                    except Exception as e:
                        if self.debug:
                            print(f"❌ Output generation failed: {e}")
                        final_outputs = ["Execution completed successfully"]
                    
                    # Success! Create final blueprint with generated outputs
                    final_blueprint = Blueprint(
                        user_intent=validated_intent,
                        actions=current_actions,
                        expected_outputs=final_outputs,
                        thought_process=validated_thought_process
                    )
                    
                    if self.debug:
                        print(f"🎉 Blueprint fully approved after {iteration} execution iteration(s)")
                    
                    return final_blueprint
                else:
                    # Generate modified actions and thought process using ActionModifier
                    modified_actions, modified_thought_process = self.action_modifier.modify_actions_and_thought(
                        validated_intent, current_actions, validated_thought_process or "", 
                        review_decision.feedback, execution_summary, original_context
                    )
                    
                    current_actions = modified_actions
                    validated_thought_process = modified_thought_process
                    
                    if self.debug:
                        print(f"🔄 Modified actions for next iteration: {[a.name for a in current_actions]}")
                        print(f"🔄 Updated thought process: {validated_thought_process[:100]}...")
                        
            except Exception as e:
                if self.debug:
                    print(f"❌ Review failed: {e}")
                if iteration == max_execution_iterations:
                    raise RuntimeError(f"Execution review failed: {str(e)}")
                continue
        
        # Max iterations reached without approval
        raise RuntimeError(
            f"Failed to approve blueprint after {max_execution_iterations} execution iterations"
        )
    
    def _review_execution_with_modification_option(
        self,
        user_intent: str,
        actions: List[ToolCalling],
        thought_process: str,
        execution_summary,
        original_context: Dict[str, Any]
    ) -> ReviewDecision:
        """Review execution results and decide whether to approve or modify (simplified reviewer)."""
        
        # Format execution results for review
        execution_details = self.execution_reviewer._format_execution_results(execution_summary)
        
        # Format current actions
        actions_json = json.dumps([a.model_dump() for a in actions], indent=2, ensure_ascii=False)
        
        prompt = textwrap.dedent(f"""
        ## Task: Review Action Execution Results
        
        You are reviewing the execution results of blueprint actions. Your ONLY job is to decide whether to approve or modify, with clear reasoning.
        
        ## Current Blueprint
        
        **User Intent:**
        {user_intent}
        
        **Thought Process:**
        {thought_process}
        
        **Executed Actions:**
        {actions_json}
        
        ## Execution Results
        
        {execution_details}
        
        ## Decision Guidelines
        
        **APPROVE** if:
        - All executions were successful without wrong parameters or wrong tools
        - Results correctly address the user intent
        - Data quality is realistic and useful
        - Thought process aligns with execution results and the actions
        
        **MODIFY** if:
        - Execution failed with wrong tools or wrong parameters provided
        - Tool sequence needs adjustment for dependencies
        - Execution results don't properly address the user intent
        - Thought process contradicts execution results and the actions
        
        ## Response Format
        
        Provide ONLY your decision in this JSON format:
        
        ```json
        {{
            "approved": true/false,
            "confidence": float (0.0-1.0),
            "reasoning": "Clear explanation of why you approve or need modifications"
        }}
        ```
        
        Make your decision:
        """).strip()
        
        messages = [{"role": "user", "content": prompt}]
        completion = sync_request_llm(self.llm_config, messages, generation_config=self.gen_opts)
        
        if self.gen_opts.debug:
            print("\n----- Execution Review LLM Response -----")
            print(completion.choices[0].message.content)
            print("------------------------------------------\n")
        
        return self._parse_simple_review_response(completion.choices[0].message.content or "")
    
    def _parse_simple_review_response(self, response: str) -> ReviewDecision:
        """Parse the simplified review response (approve/modify only)."""
        json_text = _extract_json_block(response)
        
        try:
            data = json.loads(json_text)
            
            approved = data.get("approved", False)
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "")
            
            return ReviewDecision(
                approved=approved,
                feedback=reasoning,
                confidence_score=confidence
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            return ReviewDecision(
                approved=False,
                feedback=f"Failed to parse review response: {str(e)}",
                confidence_score=0.0
            )
    

def generate_validated_blueprint(
    llm_cfg: LLMConfig,
    tools_schema: Dict[str, Any], 
    personas: List[str],
    mcp_client: MCPClient,
    max_attempts: int = 5,
    max_generation_attempts_per_persona: int = 3,
    max_execution_iterations: int = 3,
    prompt_kwargs: Optional[Dict[str, Any]] = None
) -> Blueprint:
    """Generate a validated blueprint using the new two-stage process.
    
    Stage 1: Generate intent + actions with original validation
    Stage 2: Execute actions and iteratively refine until approved
    
    Args:
        llm_cfg: LLM configuration
        tools_schema: Available tools schema
        personas: List of user personas
        mcp_client: MCP client for tool execution
        max_attempts: Maximum number of overall attempts (different personas)
        max_generation_attempts_per_persona: Max attempts for Stage 1 per persona
        max_execution_iterations: Max iterations for Stage 2 per persona
        prompt_kwargs: Additional prompt context
        
    Returns:
        Blueprint with validated actions and execution-based outputs
    """
    generator = IterativeBlueprintGenerator(llm_cfg, mcp_client, tools_schema, debug=True)
    
    for attempt in range(1, max_attempts + 1):
        persona = random.choice(personas)
        
        try:
            blueprint = generator.generate_validated_blueprint(
                persona=persona,
                max_generation_attempts=max_generation_attempts_per_persona,
                max_execution_iterations=max_execution_iterations,
                **(prompt_kwargs or {})
            )
            
            print(f"✅ Validated blueprint generated on attempt {attempt}")
            return blueprint
            
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {str(e)}")
            if attempt == max_attempts:
                raise RuntimeError(f"Failed to generate validated blueprint after {max_attempts} attempts")
    
    # Should never reach here
    raise RuntimeError("Unexpected error in blueprint generation") 