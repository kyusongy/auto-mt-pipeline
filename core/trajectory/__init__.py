"""Phase-2: Simulated human–agent interaction (trajectory collection).

This package contains a prototype implementation that follows Section 4.2 of
*APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated
Agent-Human Interplay*.

High-level flow implemented in `pipeline.py`:
1.  A validated task *blueprint* (intent + ground-truth actions + outputs) is
    passed in.
2.  `SimulatedHuman` (LLM) reveals the intent gradually using the *Trajectory
    Collection Prompt* (Fig. 11 of the paper) until the agent satisfies the
    goal.
3.  `TestAgent` (LLM in function-calling mode) tries to solve the task by
    invoking tool calls.  Each step is executed by the *environment stub* and
    traced.
4.  When the conversation ends, a validator checks that the produced sequence
    of tool calls and final assistant message match the blueprint.  Successful
    trajectories are returned.

All prompts are transcribed verbatim from the paper and exposed via helper
functions so you can easily swap in your own data.
"""

from .pipeline import TrajectoryCollector, SimulatedHuman, QwenTestAgent, Turn, Trajectory

__all__ = ["TrajectoryCollector", "SimulatedHuman", "QwenTestAgent", "Turn", "Trajectory"] 