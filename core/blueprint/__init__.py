"""Blueprint generation package.

Contains the prototype implementation for Phase-1 of APIGen-MT (task blueprint
creation & validation).  See `pipeline.py` for the entry-point helpers.

Now includes iterative validation through action execution:
- IterativeBlueprintGenerator: Main class for iterative generation
- generate_validated_blueprint: Drop-in replacement with execution validation
"""

from .pipeline import generate_valid_blueprint, Blueprint, BlueprintGenerator, BlueprintValidator
from .iterative_generator import IterativeBlueprintGenerator, generate_validated_blueprint
from .action_executor import ActionExecutor, ExecutionResult, ActionExecutionSummary
from .execution_reviewer import ExecutionReviewer, ReviewDecision

__all__ = [
    # Original components
    "generate_valid_blueprint", 
    "Blueprint", 
    "BlueprintGenerator", 
    "BlueprintValidator",
    # New iterative components
    "IterativeBlueprintGenerator",
    "generate_validated_blueprint", 
    "ActionExecutor",
    "ExecutionResult",
    "ActionExecutionSummary",
    "ExecutionReviewer",
    "ReviewDecision"
] 