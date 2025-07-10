"""Output generation based on execution results.

This module provides the OutputGenerator class that generates final outputs
for approved blueprints based on actual execution results.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List

from openai.types.chat import ChatCompletionMessageParam

from config import LLMConfig, GenerationOptions
from core.models import ToolCalling
from core.llm_client import sync_request_llm
from .pipeline import _extract_json_block


class OutputGenerator:
    """Generates final outputs based on execution results."""

    def __init__(self, llm_config: LLMConfig, gen_opts: GenerationOptions):
        self.llm_config = llm_config
        self.gen_opts = gen_opts

    def generate_outputs(
        self,
        user_intent: str,
        actions: List[ToolCalling],
        thought_process: str,
        execution_summary,
        original_context: Dict[str, Any]
    ) -> List[str]:
        """Generate final outputs based on successful execution results.
        
        Args:
            user_intent: The user's intent
            actions: The executed actions
            thought_process: The thought process that guided the actions
            execution_summary: Summary of execution results
            original_context: Original generation context
            
        Returns:
            List of output strings for trajectory collection
        """
        
        # Format execution results
        from .execution_reviewer import ExecutionReviewer
        execution_details = ExecutionReviewer._format_execution_results(execution_summary)
        
        # Format actions
        actions_json = json.dumps([a.model_dump() for a in actions], indent=2, ensure_ascii=False)
        
        prompt = textwrap.dedent(f"""
        ## Task: Generate Final Outputs
        
        Based on successful execution results, generate final outputs that summarize what was accomplished for the user.
        
        ## Original Context
        
        **Domain Rules:**
        {original_context.get('domain_rules', 'None provided')}
        
        ## Blueprint Information
        
        **User Intent:**
        {user_intent}
        
        **Thought Process:**
        {thought_process}
        
        **Executed Actions:**
        {actions_json}
        
        ## Execution Results
        
        {execution_details}
        
        ## Instructions
        
        Generate final outputs that:
        1. **Summarize what was accomplished** for the user's intent
        2. **Include key information** from the execution results
        3. **Are realistic and accurate** based on actual execution data
        4. **Are useful for trajectory collection** - showing what the actions achieved
        5. **Match the user's original request** and intent
        
        The outputs should be clear, concise summaries that someone could use to understand:
        - What the user wanted
        - What actions were taken
        - What results were obtained
        
        ## Response Format
        
        Provide your response in JSON format:
        
        ```json
        {{
            "outputs": [
                "Summary of first key result or accomplishment",
                "Summary of second key result or accomplishment",
                "Additional relevant information or outcomes"
            ]
        }}
        ```
        
        Generate the outputs:
        """).strip()
        
        messages = [{"role": "user", "content": prompt}]
        completion = sync_request_llm(self.llm_config, messages, generation_config=self.gen_opts)
        
        if self.gen_opts.debug:
            print("\n----- Output Generator LLM Response -----")
            print(completion.choices[0].message.content)
            print("------------------------------------------\n")
        
        return self._parse_output_response(completion.choices[0].message.content or "")
    
    def _parse_output_response(self, response: str) -> List[str]:
        """Parse the output generation response and extract outputs."""
        json_text = _extract_json_block(response)
        
        try:
            data = json.loads(json_text)
            outputs = data.get("outputs", [])
            
            if not outputs:
                return ["Execution completed successfully"]
            
            return outputs
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to default output
            return ["Execution completed successfully"] 