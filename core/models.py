from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolCalling(BaseModel):
    name: str = Field(..., description="The name of the tool to be called.")
    arguments: Optional[Dict[str, Any]] = Field(
        ..., description="A dictionary of arguments to be passed to the tool."
    )