"""Trajectory validation for Phase 2 of the auto-mt-pipeline.

This module implements trajectory validation using a separate LLM to evaluate
whether the collected conversation between simulated human and sales agent
follows the ground truth actions, outputs, and thought process from Phase 1.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletionMessageParam

from config import LLMConfig, GenerationOptions, TRAJECTORY_JUDGE_OPTIONS
from core.llm_client import sync_request_llm
from core.blueprint.pipeline import Blueprint
from .models import Trajectory, Turn, ValidationResult


class TrajectoryValidator:
    """Validates collected trajectories against Phase 1 blueprints."""
    
    def __init__(
        self, 
        llm_config: LLMConfig, 
        generation_opts: Optional[GenerationOptions] = None,
        debug: bool = False
    ):
        self.llm_config = llm_config
        self.gen_opts = generation_opts or TRAJECTORY_JUDGE_OPTIONS
        self.debug = debug

    def _build_validation_prompt(
        self,
        blueprint: Blueprint,
        trajectory: Trajectory,
    ) -> List[ChatCompletionMessageParam]:
        """Build the validation prompt for the LLM."""
        
        # Format the blueprint information
        blueprint_info = f"""
## Phase 1 Blueprint Information

**User Intent:**
{blueprint.user_intent}

**Planned Actions:**
{json.dumps([{"name": action.name, "arguments": action.arguments} for action in blueprint.actions], indent=2, ensure_ascii=False)}

**Expected Outputs:**
{json.dumps(blueprint.expected_outputs, indent=2, ensure_ascii=False)}

**Thought Process:**
{blueprint.thought_process}
"""

        # Format the conversation using the existing history and tool_calls
        conversation_lines = []
        
        # Add conversation turns (excluding system messages that contain the intent)
        for turn in trajectory.turns:
            if turn.role == "system":
                continue  # Skip system messages that contain the hidden intent
            elif turn.role == "user":
                conversation_lines.append(f"USER: {turn.content}")
            elif turn.role == "assistant":
                conversation_lines.append(f"ASSISTANT: {turn.content}")
            elif turn.role == "function_call":
                try:
                    call_data = json.loads(turn.content)
                    conversation_lines.append(f"ASSISTANT: [CALLS {call_data.get('name', 'unknown')}]")
                except:
                    conversation_lines.append(f"ASSISTANT: [CALLS function]")
            elif turn.role == "observation":
                conversation_lines.append(f"SYSTEM: [TOOL RESULT] {turn.content[:100]}...")
        
        # Add tool calls summary for validation reference
        if trajectory.tool_calls:
            conversation_lines.append(f"\nTOOL CALLS EXECUTED:")
            for i, tool_call in enumerate(trajectory.tool_calls, 1):
                conversation_lines.append(f"  {i}. {tool_call.name}: {json.dumps(tool_call.arguments, ensure_ascii=False)}")
        
        conversation_text = "\n".join(conversation_lines)

        # Build the validation prompt
        prompt = textwrap.dedent(f"""
You are an expert evaluator for conversation quality and task completion. You need to validate whether a collected conversation between a simulated human and a lenovo agent properly follows the blueprint generated in Phase 1.

{blueprint_info}

## Collected Conversation

{conversation_text}

## Validation Criteria

Please evaluate this conversation based on the following criteria:

1. **Intent Expression (0-1 points):**
   - Does the simulated human fully express their intent from the blueprint (0/0.5)?
   - Are all key details from the intent properly communicated during the conversation? (0/0.5)

2. **Action Flow Adherence (0-5 points):**
   - Does the conversation flow follow the planned actions from the blueprint (subtract 1 point for each missing actions from the blueprint. 0-3)?
   - Are the tool calls made in a logical sequence that matches the blueprint (0/1)?
   - If agent made extra tool call(s) due to the gradually revealed information, does it follow the blueprint (0/1)?


3. **Output Achievement (0-1 points):**
   - Are the expected outputs from the blueprint achieved in the conversation (0/0.5)?
   - Is the user's goal satisfied by the end of the conversation (0/0.5)?

4. **Thought Process Alignment (0-1 points):**
   - Does the conversation reflect the reasoning and thought process from the blueprint (0/1)?


## Scoring Guidelines

- **Score 8 (APPROVED):** The conversation excellently follows the blueprint, achieves all goals, and feels natural
- **Score 6-7 (APPROVED with minor issues):** The conversation mostly follows the blueprint with minor deviations
- **Score 0-5 (REJECTED with major issues):** Significant deviations from the blueprint or poor conversation quality

## Response Format

Please provide your evaluation in the following JSON format:

{{
  "score": <0-8 integer>,
  "is_approved": <true/false>,
  "issues": ["<list of specific issues found>"],
  "strengths": ["<list of specific strengths>"]
}}

Focus on whether the conversation successfully implements the blueprint's intent, actions, and expected outputs while maintaining natural dialogue flow.
""").strip()

        return [{"role": "user", "content": prompt}]

    def validate_trajectory(
        self, 
        blueprint: Blueprint, 
        trajectory: Trajectory
    ) -> ValidationResult:
        """Validate a collected trajectory against the Phase 1 blueprint."""
        
        if self.debug:
            print(f"\n🔍 Validating trajectory ...")
        
        # Build validation prompt
        messages = self._build_validation_prompt(blueprint, trajectory)
        
        # Get validation response from LLM
        completion = sync_request_llm(
            self.llm_config,
            messages,
            generation_config=self.gen_opts,
        )
        
        raw_content = completion.choices[0].message.content
        if self.debug:
            print(f"\n----- Trajectory Validator raw output -----\n{raw_content}\n-------------------------------\n")
        
        if not isinstance(raw_content, str):
            raise ValueError("Expected string content from validation LLM response")
        
        # Parse the JSON response
        try:
            # Try to extract JSON from the response
            json_start = raw_content.find('{')
            json_end = raw_content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = raw_content[json_start:json_end]
                validation_data = json.loads(json_content)
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            if self.debug:
                print(f"Failed to parse validation response as JSON: {e}")
                print(f"Raw response: {raw_content}")
            
            # Fallback: try to extract score and approval from text
            return self._parse_fallback_response(raw_content)
        
        # Extract validation result
        score = validation_data.get("score", 0)
        is_approved = validation_data.get("is_approved", False)
        issues = validation_data.get("issues", [])
        strengths = validation_data.get("strengths", [])
        
        result = ValidationResult(
            is_approved=is_approved,
            score=score,
            issues=issues,
            strengths=strengths
        )
        
        if self.debug:
            print(f"✅ Validation complete: Score={score}, Approved={is_approved}")
            if issues:
                print(f"❌ Issues: {issues}")
            if strengths:
                print(f"✅ Strengths: {strengths}")
        
        return result
    
    def _parse_fallback_response(self, raw_content: str) -> ValidationResult:
        """Fallback parsing when JSON parsing fails."""
        import re
        
        # Look for score patterns
        score_match = re.search(r'score["\s]*:["\s]*(\d+)', raw_content, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 4
        
        # Look for approval patterns
        approved_match = re.search(r'approved["\s]*:["\s]*(true|false)', raw_content, re.IGNORECASE)
        is_approved = approved_match.group(1).lower() == "true" if approved_match else score >= 6
        
        return ValidationResult(
            is_approved=is_approved,
            score=score,
            issues=["Failed to parse validation response properly"],
            strengths=["Validation attempted"]
        ) 