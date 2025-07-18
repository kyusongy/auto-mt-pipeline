"""Shared data models for trajectory collection and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.models import ToolCalling


@dataclass
class Turn:
    """A single turn in a conversation."""
    role: str  # "user" / "assistant" / "system" / "function_call" / "observation"
    content: str


@dataclass
class Trajectory:
    """A complete conversation trajectory with tool calls."""
    turns: List[Turn]
    tool_calls: List[ToolCalling]


@dataclass
class ValidationResult:
    """Result of trajectory validation."""
    is_approved: bool
    score: int  # 0-8 score based on the updated scoring system
    issues: List[str]
    strengths: List[str] 