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
from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletionMessageParam

from config import LLMConfig, GenerationOptions, BLUEPRINT_GENERATION_OPTIONS
from core.models import ToolCalling
from core.llm_client import sync_request_llm
from core.mcp_client import MCPClient
from .pipeline import Blueprint, BlueprintValidator, ReviewCommittee, _extract_json_block
from .action_executor import ActionExecutor
from .execution_reviewer import ExecutionReviewer, ReviewDecision


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
            Generate a realistic task instruction and corresponding actions for a Lenovo customer scenario. 
            **Important**: Generate only the user intent and actions - do NOT generate expected outputs yet. 
            The outputs will be created based on actual tool execution results in Stage 2.

            ## Guidelines for Generating Task Instruction (intent)
            {domain_rules}
            
            ## Guidelines for Generating Actions
            1. Focus on generating actions that help users with Lenovo products and services.
            2. For actions that provide information requests, use appropriate tools like product_recommend, product_knowledge_retrieval, etc.
            3. Include multiple tool calls when the scenario requires comprehensive assistance (e.g., product recommendation + parameter comparison).
            4. Provide precise tool calls with all necessary parameters for each action.
            5. Ensure all actions adhere to Lenovo service policies and help users make informed decisions.
            6. **Tool Chaining & Dependencies**: Some tools require outputs from previous tools as inputs:
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
            Generate your response according to the following format. Enclose the thought process within `<thought></thought>` tags, 
            and the final structured response within `<answer></answer>` tags. The structured response should be in strict JSON format.

            **JSON Format:**
            {{
                "intent": "User intent description",
                "actions": [
                    {{
                        "name": "tool_name",
                        "arguments": {{
                            "param1": "value1",
                            "param2": "value2"
                        }}
                    }}
                ]
            }}

            ## Example Tasks (for reference only)
            {examples}

            ## User Context
            **User Details:** {sampled_user_details}
            **Order Context:** {sampled_orders}

            ## Feedback from Previous Attempt
            {prev_feedback}

            Do not directly copy instruction and action patterns from the examples. Ground the generation from the provided data.
            Generate the task now focusing on realistic user scenarios.
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
        return Blueprint(intent, actions, expected_outputs=[], raw_response=json_content)


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
        self.action_executor = ActionExecutor(mcp_client, debug=debug)
        self.execution_reviewer = ExecutionReviewer(llm_config, gen_opts)

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
            
            # Execute current actions
            try:
                execution_summary = self.action_executor.execute_actions(current_actions, validated_intent)
                
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
                    validated_intent, current_actions, execution_summary, original_context
                )
                
                if self.debug:
                    print(f"📋 Review decision: {'✅ APPROVED' if review_decision.approved else '🔄 MODIFY ACTIONS'}")
                    print(f"📊 Confidence: {review_decision.confidence_score:.2f}")
                
                if review_decision.approved:
                    # Success! Create final blueprint with execution-based outputs
                    final_blueprint = Blueprint(
                        user_intent=validated_intent,
                        actions=current_actions,
                        expected_outputs=review_decision.final_outputs or ["Execution completed successfully"]
                    )
                    
                    if self.debug:
                        print(f"🎉 Blueprint fully approved after {iteration} execution iteration(s)")
                    
                    return final_blueprint
                else:
                    # Extract modified actions for next iteration
                    if hasattr(review_decision, 'modified_actions') and review_decision.modified_actions:
                        current_actions = review_decision.modified_actions
                        if self.debug:
                            print(f"🔄 Using modified actions for next iteration: {[a.name for a in current_actions]}")
                    else:
                        if self.debug:
                            print(f"❌ No modified actions provided, using feedback for refinement")
                        # Fallback: generate new actions based on feedback
                        current_actions = self._regenerate_actions_from_feedback(
                            validated_intent, review_decision.feedback, original_context
                        )
                        
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
        execution_summary,
        original_context: Dict[str, Any]
    ) -> ReviewDecision:
        """Review execution results with option to modify actions OR approve completely."""
        
        # Format execution results for review
        execution_details = self.execution_reviewer._format_execution_results(execution_summary)
        
        # Format current actions
        actions_json = json.dumps([a.model_dump() for a in actions], indent=2, ensure_ascii=False)
        
        prompt = textwrap.dedent(f"""
        ## Task: Review Action Execution Results
        
        You are reviewing the execution results of blueprint actions. Based on the results, you must decide:
        1. **MODIFY ACTIONS**: If execution needs improvement, provide modified actions with the same intent
        2. **APPROVE**: If execution results are satisfactory, provide final outputs for trajectory collection
        
        ## Original Context (from initial generation)
        
        **Domain Rules:**
        {original_context.get('domain_rules', 'None provided')}
        
        **API Dependencies:**
        {original_context.get('api_dependencies', 'None provided')}
        
        **Available Tools:**
        {json.dumps(original_context.get('tools_schema', {}), indent=2, ensure_ascii=False)[:2000]}...
        
        **Example Tasks:**
        {original_context.get('examples', 'None provided')}
        
        ## Current Blueprint
        
        **User Intent (DO NOT MODIFY):**
        {user_intent}
        
        **Current Actions:**
        {actions_json}
        
        ## Execution Results
        
        {execution_details}
        
        ## Decision Guidelines
        
        **MODIFY ACTIONS** if:
        - Execution failed or produced poor results
        - Wrong tools were used or wrong parameters provided
        - Tool sequence needs adjustment for dependencies
        - Results don't properly address the user intent
        
        **APPROVE** if:
        - All executions were successful with good results
        - Results correctly address the user intent
        - Data quality is realistic and useful
        - Tool usage follows proper dependencies
        
        ## Response Format
        
        Provide your decision in this JSON format:
        
        **For MODIFY ACTIONS:**
        ```json
        {{
            "approved": false,
            "confidence": float (0.0-1.0),
            "reasoning": "Why actions need modification",
            "modified_actions": [
                {{
                    "name": "tool_name",
                    "arguments": {{"param": "value"}}
                }}
            ]
        }}
        ```
        
        **For APPROVE:**
        ```json
        {{
            "approved": true,
            "confidence": float (0.0-1.0),
            "reasoning": "Why execution is satisfactory",
            "final_outputs": ["Summary 1", "Summary 2", ...] // Based on execution results
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
        
        return self._parse_execution_review_response(completion.choices[0].message.content or "")
    
    def _parse_execution_review_response(self, response: str) -> ReviewDecision:
        """Parse the execution review response with support for action modification."""
        json_text = _extract_json_block(response)
        
        try:
            data = json.loads(json_text)
            
            approved = data.get("approved", False)
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "")
            
            if approved:
                final_outputs = data.get("final_outputs", [])
                if not final_outputs:
                    final_outputs = ["Action execution completed successfully"]
                
                decision = ReviewDecision(
                    approved=True,
                    feedback=reasoning,
                    final_outputs=final_outputs,
                    confidence_score=confidence
                )
            else:
                # Parse modified actions
                modified_actions_data = data.get("modified_actions", [])
                modified_actions = []
                
                for action_data in modified_actions_data:
                    if isinstance(action_data, dict):
                        name = action_data.get("name", "")
                        arguments = action_data.get("arguments", {})
                        if name:
                            modified_actions.append(ToolCalling(name=name, arguments=arguments))
                
                decision = ReviewDecision(
                    approved=False,
                    feedback=reasoning,
                    confidence_score=confidence
                )
                # Add modified actions as custom attribute
                decision.modified_actions = modified_actions
                
            return decision
            
        except (json.JSONDecodeError, ValueError) as e:
            return ReviewDecision(
                approved=False,
                feedback=f"Failed to parse execution review response: {str(e)}",
                confidence_score=0.0
            )
    
    def _regenerate_actions_from_feedback(
        self, 
        intent: str, 
        feedback: str, 
        original_context: Dict[str, Any]
    ) -> List[ToolCalling]:
        """Fallback: regenerate actions based on feedback if no modified actions provided."""
        
        prompt = textwrap.dedent(f"""
        ## Task: Modify Actions Based on Execution Feedback
        
        Generate modified actions for the following intent based on execution feedback.
        Keep the intent unchanged, only modify the actions.
        
        **Intent (DO NOT CHANGE):**
        {intent}
        
        **Execution Feedback:**
        {feedback}
        
        **Available Tools:**
        {json.dumps(original_context.get('tools_schema', {}), indent=2, ensure_ascii=False)[:1000]}...
        
        **Domain Rules:**
        {original_context.get('domain_rules', 'None provided')}
        
        ## Response Format
        
        Provide only the modified actions in JSON format:
        
        ```json
        {{
            "actions": [
                {{
                    "name": "tool_name",
                    "arguments": {{"param": "value"}}
                }}
            ]
        }}
        ```
        """).strip()
        
        messages = [{"role": "user", "content": prompt}]
        completion = sync_request_llm(self.llm_config, messages, generation_config=self.gen_opts)
        response = completion.choices[0].message.content or ""
        
        # Parse response
        json_text = _extract_json_block(response)
        try:
            data = json.loads(json_text)
            actions_data = data.get("actions", [])
            
            actions = []
            for action_data in actions_data:
                if isinstance(action_data, dict):
                    name = action_data.get("name", "")
                    arguments = action_data.get("arguments", {})
                    if name:
                        actions.append(ToolCalling(name=name, arguments=arguments))
            
            return actions if actions else [ToolCalling(name="fallback", arguments={})]
            
        except (json.JSONDecodeError, ValueError):
            # Ultimate fallback
            return [ToolCalling(name="general_knowledge_retrieval", arguments={"query": intent})]


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