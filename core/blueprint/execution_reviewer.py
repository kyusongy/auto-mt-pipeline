"""Execution review for iterative blueprint validation.

This module provides the ExecutionReviewer class that reviews action execution
results along with the original generation context to decide whether to approve
the blueprint or request action refinements.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam

from config import LLMConfig, GenerationOptions, BLUEPRINT_GENERATION_OPTIONS
from core.models import ToolCalling
from core.llm_client import sync_request_llm
from .action_executor import ActionExecutionSummary


@dataclass
class ReviewDecision:
    """Result of execution review."""
    approved: bool
    feedback: str  # For refinement if not approved
    final_outputs: Optional[List[str]] = None  # Summary outputs if approved
    confidence_score: float = 0.0  # 0-1 confidence in the decision


class ExecutionReviewer:
    """Reviews action execution results and decides approval or refinement."""
    
    def __init__(self, llm_config: LLMConfig, gen_opts: Optional[GenerationOptions] = None):
        self.llm_config = llm_config
        self.gen_opts = gen_opts or BLUEPRINT_GENERATION_OPTIONS.model_copy(update={"debug": True})
    
    def review_execution(
        self,
        user_intent: str,
        actions: List[ToolCalling],
        execution_summary: ActionExecutionSummary,
        original_context: Dict[str, Any]
    ) -> ReviewDecision:
        """Review execution results and decide whether to approve or refine.
        
        Args:
            user_intent: The generated user intent
            actions: The actions that were executed
            execution_summary: Results of action execution
            original_context: Original generation context (domain_rules, api_dependencies, etc.)
            
        Returns:
            ReviewDecision with approval status and feedback
        """
        messages = self._build_review_prompt(
            user_intent, actions, execution_summary, original_context
        )
        
        completion = sync_request_llm(self.llm_config, messages, generation_config=self.gen_opts)
        
        if self.gen_opts.debug:
            print("\n----- ExecutionReviewer LLM Response -----")
            print(completion.choices[0].message.content)
            print("------------------------------------------\n")
        
        return self._parse_review_response(completion.choices[0].message.content or "")
    
    def _build_review_prompt(
        self,
        user_intent: str,
        actions: List[ToolCalling],
        execution_summary: ActionExecutionSummary,
        original_context: Dict[str, Any]
    ) -> List[ChatCompletionMessageParam]:
        """Build the review prompt with full context."""
        
        # Format execution results for review
        execution_details = self._format_execution_results(execution_summary)
        
        # Format original actions
        actions_json = json.dumps([a.model_dump() for a in actions], indent=2, ensure_ascii=False)
        
        prompt = textwrap.dedent(f"""
        ## Task: Review Blueprint Action Execution
        
        You are reviewing the execution results of blueprint actions to determine if they should be approved or refined.
        
        ## Original Context
        You generated a blueprint based on these requirements:
        
        **Domain Rules:**
        {original_context.get('domain_rules', 'None provided')}
        
        **API Dependencies:**
        {original_context.get('api_dependencies', 'None provided')}
        
        **Available Tools:**
        {json.dumps(original_context.get('tools_schema', {}), indent=2, ensure_ascii=False)[:2000]}...
        
        **Example Tasks:**
        {original_context.get('examples', 'None provided')}
        
        **User Details Context:**
        {original_context.get('sampled_user_details', 'None provided')}
        
        **Order Context:**
        {original_context.get('sampled_orders', 'None provided')}
        
        ## Generated Blueprint
        
        **User Intent:**
        {user_intent}
        
        **Actions:**
        {actions_json}
        
        ## Execution Results
        
        {execution_details}
        
        ## Review Guidelines
        
        Evaluate whether the execution results meet the following criteria:
        
        1. **Correctness**: Do the execution results correctly address the user intent?
        2. **Completeness**: Are all necessary steps completed successfully?
        3. **Domain Compliance**: Do the results follow the domain rules and policies?
        4. **Data Quality**: Are the execution results realistic and useful?
        5. **Tool Usage**: Were the right tools used in the correct sequence?
        
        ## Decision Options
        
        **APPROVE**: If execution results are satisfactory:
        - Generate final output summaries based on the execution results
        - These summaries will be used for trajectory validation
        - Keep summaries concise but informative
        
        **REFINE**: If execution needs improvement:
        - Provide specific feedback on what needs to be changed
        - Focus on actionable improvements to the action sequence
        - Consider tool dependencies and parameter corrections
        
        ## Response Format
        
        Provide your decision in this JSON format:
        
        ```json
        {{
            "approved": boolean,
            "confidence": float (0.0-1.0),
            "reasoning": "Detailed explanation of your decision",
            "feedback": "Specific feedback for refinement (if not approved)",
            "final_outputs": ["Summary 1", "Summary 2", ...] // Only if approved
        }}
        ```
        
        Review the execution and make your decision:
        """).strip()
        
        return [{"role": "user", "content": prompt}]
    
    def _format_execution_results(self, execution_summary: ActionExecutionSummary) -> str:
        """Format execution results for display in the review prompt."""
        lines = []
        
        lines.append(f"**Execution Summary:**")
        lines.append(f"- Total Actions: {execution_summary.total_actions}")
        lines.append(f"- Successful: {execution_summary.successful_actions}")
        lines.append(f"- Failed: {execution_summary.failed_actions}")
        lines.append(f"- Execution Order: {' → '.join(execution_summary.dependency_chain)}")
        lines.append("")
        
        lines.append("**Detailed Results:**")
        for i, result in enumerate(execution_summary.results, 1):
            lines.append(f"\n{i}. **{result.tool_name}**")
            lines.append(f"   - Arguments: {json.dumps(result.arguments, ensure_ascii=False)}")
            lines.append(f"   - Status: {'✅ Success' if result.success else '❌ Failed'}")
            
            if result.success:
                lines.append(f"   - Result: {json.dumps(result.result, ensure_ascii=False, indent=2)[:1000]}...")
            else:
                lines.append(f"   - Error: {result.error}")
                
            if result.execution_time:
                lines.append(f"   - Time: {result.execution_time:.2f}s")
        
        return "\n".join(lines)
    
    def _parse_review_response(self, response: str) -> ReviewDecision:
        """Parse the LLM review response into a ReviewDecision."""
        # Extract JSON from response
        json_text = self._extract_json_block(response)
        
        try:
            data = json.loads(json_text)
            
            approved = data.get("approved", False)
            confidence = float(data.get("confidence", 0.0))
            feedback = data.get("feedback", "")
            
            if approved:
                final_outputs = data.get("final_outputs", [])
                if not final_outputs:
                    # Fallback: generate basic summary if none provided
                    final_outputs = ["Action execution completed successfully"]
            else:
                final_outputs = None
                
            return ReviewDecision(
                approved=approved,
                feedback=feedback,
                final_outputs=final_outputs,
                confidence_score=confidence
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback for parsing errors
            return ReviewDecision(
                approved=False,
                feedback=f"Failed to parse review response: {str(e)}",
                confidence_score=0.0
            )
    
    def _extract_json_block(self, text: str) -> str:
        """Extract JSON block from LLM response."""
        cleaned = text.strip()
        
        # Remove markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        
        # Find JSON object
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            return brace_match.group(0)
            
        return cleaned.strip() 