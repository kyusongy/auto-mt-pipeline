"""Action and thought process modification based on execution feedback.

This module provides the ActionModifier class that generates modified actions
and thought processes when execution results need improvement.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List, Tuple

from openai.types.chat import ChatCompletionMessageParam

from config import LLMConfig, GenerationOptions
from core.models import ToolCalling
from core.llm_client import sync_request_llm
from .pipeline import _extract_json_block


class ActionModifier:
    """Modifies actions and thought processes based on execution feedback."""

    def __init__(self, llm_config: LLMConfig, gen_opts: GenerationOptions):
        self.llm_config = llm_config
        self.gen_opts = gen_opts

    def modify_actions_and_thought(
        self,
        user_intent: str,
        current_actions: List[ToolCalling],
        current_thought_process: str,
        execution_feedback: str,
        execution_summary,
        original_context: Dict[str, Any]
    ) -> Tuple[List[ToolCalling], str]:
        """Generate modified actions and thought process based on execution feedback.
        
        Args:
            user_intent: The user's intent (unchanged)
            current_actions: Current actions that need modification
            current_thought_process: Current thought process that may need updating
            execution_feedback: Reasoning from reviewer about why modification is needed
            execution_summary: Actual execution results showing what happened
            original_context: Original generation context
            
        Returns:
            Tuple of (modified_actions, modified_thought_process)
        """
        
        # Format execution results
        from .execution_reviewer import ExecutionReviewer
        execution_details = ExecutionReviewer._format_execution_results(execution_summary)
        
        # Format current actions
        actions_json = json.dumps([a.model_dump() for a in current_actions], indent=2, ensure_ascii=False)
        
        prompt = textwrap.dedent(f"""
        ## Task: Modify Actions and Thought Process
        
        Based on execution feedback, generate improved actions and thought process while keeping the user intent unchanged.
        
        ## Original Context
        
        **Domain Rules:**
        {original_context.get('domain_rules', 'None provided')}
        
        **API Dependencies:**
        {original_context.get('api_dependencies', 'None provided')}
        
        **Available Tools:**
        {json.dumps(original_context.get('tools_schema', {}), indent=2, ensure_ascii=False)[:2000]}...
        
        ## Current Blueprint
        
        **User Intent (DO NOT MODIFY):**
        {user_intent}
        
        **Current Thought Process:**
        {current_thought_process}
        
        **Current Actions:**
        {actions_json}
        
        ## Execution Results
        
        {execution_details}
        
        ## Reviewer Feedback
        
        **Why modification is needed:**
        {execution_feedback}
        
        ## Instructions
        
        Based on the actual execution results and reviewer feedback, generate improved actions and thought process.
        
        1. **Keep the user intent exactly the same**
        2. **Update the thought process** to reflect a realistic reasoning that:
           - Explains the new strategy for the modified actions
           - Shows realistic problem-solving thinking
        3. **Generate new actions** that:
           - Fix the specific problems shown in execution results
           - Address the issues identified in the reviewer feedback
           - Follow proper tool dependencies and sequences
           - Use appropriate parameters for each tool
           - Are more likely to succeed based on what we learned from execution
        
        ## Response Format
        
        Provide your response in JSON format:
        
        ```json
        {{
            "modified_thought_process": "Updated realistic thought process that explains the reasoning for the new actions",
            "modified_actions": [
                {{
                    "name": "tool_name",
                    "arguments": {{"param": "value"}}
                }}
            ]
        }}
        ```
        
        Generate the modifications:
        """).strip()
        
        messages = [{"role": "user", "content": prompt}]
        completion = sync_request_llm(self.llm_config, messages, generation_config=self.gen_opts)
        
        if self.gen_opts.debug:
            print("\n----- Action Modifier LLM Response -----")
            print(completion.choices[0].message.content)
            print("------------------------------------------\n")
        
        return self._parse_modification_response(completion.choices[0].message.content or "")
    
    def _parse_modification_response(self, response: str) -> Tuple[List[ToolCalling], str]:
        """Parse the modification response and extract actions and thought process."""
        json_text = _extract_json_block(response)
        
        try:
            data = json.loads(json_text)
            
            modified_thought_process = data.get("modified_thought_process", "")
            modified_actions_data = data.get("modified_actions", [])
            
            modified_actions = []
            for action_data in modified_actions_data:
                if isinstance(action_data, dict):
                    name = action_data.get("name", "")
                    arguments = action_data.get("arguments", {})
                    if name:
                        modified_actions.append(ToolCalling(name=name, arguments=arguments))
            
            return modified_actions, modified_thought_process
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse action modification response: {str(e)}") 