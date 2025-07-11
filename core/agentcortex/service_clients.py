#!/usr/bin/env python3

"""AgentCortex Service Clients

HTTP clients for all AgentCortex services following the original workflow.py pattern.
These clients make HTTP calls to the actual AgentCortex microservices.
"""

import json
import httpx
from typing import Dict, Any, Optional
from config.agentcortex_config import agentcortex_config

from agent_types.planning import PlanningRequest, PlanningResponse
from agent_types.execution import ToolExecutingRequest, ToolExecutingResponse
from agent_types.intent import RewritingRequest, RewritingResponse
from agent_types.memory.session import ReadSessionMemoryRequest, ReadSessionMemoryResponse, WriteChatHistoryRequest
from agent_types.memory.session import ExtractMentionsRequest, ExtractMentionsResponse
from agent_types.personalization import ExtractUserPreferenceRequest, ExtractUserPreferenceResponse


class HTTPOptions:
    """HTTP options for service calls."""
    def __init__(self, timeout: int = 60):
        self.timeout = timeout


class APIClient:
    """Generic API client following the original agentcortex pattern."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    def post(self, endpoint: str, data: Dict[str, Any], options: Optional[HTTPOptions] = None) -> Dict[str, Any]:
        """Make a POST request to the service."""
        if not self.base_url:
            raise ValueError(f"Service URL not configured for endpoint {endpoint}")
        
        url = f"{self.base_url}{endpoint}"
        timeout = options.timeout if options else 60
        
        try:
            with httpx.Client(verify=False, timeout=timeout) as client:
                response = client.post(
                    url,
                    json=data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            raise RuntimeError(f"Failed to call {url}: {str(e)}")


class AgentCortexServiceClients:
    """Collection of all AgentCortex service clients following workflow.py pattern."""
    
    def __init__(self):
        """Initialize all service clients."""
        self.session_memory = APIClient(agentcortex_config.session_memory_url)
        self.intent = APIClient(agentcortex_config.intent_url)
        self.planner = APIClient(agentcortex_config.planning_url)
        self.executor = APIClient(agentcortex_config.execution_url)
        self.summarizer = APIClient(agentcortex_config.summarization_url)
        self.extract_mentions = APIClient(agentcortex_config.extract_mentions_url)
        self.personalization_service = APIClient(agentcortex_config.personalization_url)
        
    def is_available(self) -> bool:
        """Check if all required services are available."""
        return (agentcortex_config.execution_url != "" and 
                agentcortex_config.planning_url != "" and
                agentcortex_config.session_memory_url != "")
    
    # Planning Service
    def plan(self, request: PlanningRequest) -> PlanningResponse:
        """Call planning service."""
        response_data = self.planner.post("/plan", request.model_dump(), HTTPOptions(timeout=60))
        return PlanningResponse.model_validate(response_data)
    
    # Execution Service  
    def execute_tools(self, request: ToolExecutingRequest) -> ToolExecutingResponse:
        """Call execution service."""
        response_data = self.executor.post("/execute_tools", request.model_dump())
        
        # Handle AgentCortex response format - it wraps the response in "result"
        if "result" in response_data and "observation" in response_data["result"]:
            # AgentCortex wraps the response in "result"
            actual_response = response_data["result"]
        else:
            actual_response = response_data
        
        return ToolExecutingResponse.model_validate(actual_response)
    
    def list_tools(self) -> Dict[str, Any]:
        """List tools from execution service."""
        return self.executor.post("/list_tools", {})
    
    # Intent Service
    def rewrite_query(self, request: RewritingRequest) -> RewritingResponse:
        """Call intent rewriting service."""
        response_data = self.intent.post("/rewrite_query", request.model_dump())
        return RewritingResponse.model_validate(response_data)
    
    # Session Memory Service
    def read_session_memory(self, request: ReadSessionMemoryRequest) -> ReadSessionMemoryResponse:
        """Read session memory."""
        response_data = self.session_memory.post("/read_session_memory", request.model_dump())
        return ReadSessionMemoryResponse.model_validate(response_data)
    
    def write_chat_history(self, request: WriteChatHistoryRequest) -> None:
        """Write chat history to session memory."""
        self.session_memory.post("/write_chat_history", request.model_dump())
    
    # Mention Extraction Service
    def extract_mentions(self, request: ExtractMentionsRequest) -> ExtractMentionsResponse:
        """Extract mentions from query."""
        response_data = self.extract_mentions.post("/extract_mentions", request.model_dump())
        return ExtractMentionsResponse.model_validate(response_data)
    
    # Personalization Service
    def extract_session_preference(self, request: ExtractUserPreferenceRequest) -> ExtractUserPreferenceResponse:
        """Extract session preferences."""
        response_data = self.personalization_service.post("/extract_session_preference", request.model_dump())
        return ExtractUserPreferenceResponse.model_validate(response_data)


# Global service clients instance
service_clients = AgentCortexServiceClients() 