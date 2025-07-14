#!/usr/bin/env python3
"""Quick script to run ONLY the trajectory collection phase using a pre-generated
blueprint JSON (e.g. data/blueprint.json).

Usage:
    python test_trajectory.py --blueprint-file data/blueprint.json

Environment variables:
    BLUEPRINT_FILE – alternative way to specify the blueprint file path.

The rest of the pipeline (MCP tool schemas, AgentCortex hybrid execution, etc.)
relies on the same configuration used by `run_pipeline.py`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

# Import shared pipeline pieces (no side-effects)
from run_pipeline import get_tool_schemas, llm_config, pipeline_config
from config import is_agentcortex_enabled, agentcortex_config
from core.trajectory.pipeline import TrajectoryCollector
from core.models import ToolCalling
from core.blueprint.pipeline import Blueprint


def load_blueprint(path: Path) -> tuple[Blueprint, dict]:
    """Load blueprint.json and return Blueprint object + raw dict."""
    if not path.exists():
        raise FileNotFoundError(f"Blueprint file not found: {path}")

    blueprint_json = json.loads(path.read_text(encoding="utf-8"))
    bp = Blueprint(
        user_intent=blueprint_json["intent"],
        actions=[ToolCalling(**a) for a in blueprint_json["actions"]],
        expected_outputs=blueprint_json.get("outputs", []),
    )
    return bp, blueprint_json


def main():
    parser = argparse.ArgumentParser("Trajectory-only runner")
    parser.add_argument(
        "--blueprint-file",
        type=str,
        default=os.getenv("BLUEPRINT_FILE", "data/blueprint.json"),
        help="Path to existing blueprint.json",
    )
    args = parser.parse_args()

    blueprint_path = Path(args.blueprint_file)
    blueprint, blueprint_data = load_blueprint(blueprint_path)

    print("=" * 60)
    print("🎯 Trajectory-only Test")
    print("=" * 60)
    print(f"Loaded blueprint from: {blueprint_path}\n")

    # Retrieve tools (MCP or dummy) and optional mcp_client
    tools_schema, mcp_client = get_tool_schemas()

    # If we do have MCP, register wrapper so Qwen sees cleaned schemas
    if mcp_client:
        from core.trajectory.qwen_tool_wrappers import initialize_mcp_client
        initialize_mcp_client()
    else:
        print("📦 Using dummy tools for trajectory collection")

    # Show AgentCortex integration status
    if is_agentcortex_enabled():
        print("🧠 AgentCortex integration ENABLED → Hybrid execution will run")
    else:
        print("⚠️  AgentCortex integration DISABLED – tool calls will not execute")

    collector = TrajectoryCollector(
        llm_config,  # human simulator
        llm_config,  # assistant planning LLM (Qwen)
        tools_schema=tools_schema,
        debug=pipeline_config.debug,
        bon_n=pipeline_config.bon_n,
        use_plan_execute_agent=True,  # Qwen planning + AgentCortex execute
    )

    trajectory = collector.collect(blueprint)

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    if trajectory:
        print(f"\n✅ Collected trajectory with {len(trajectory.turns)} turns")
        traj_data = {
            "blueprint": blueprint_data,
            "trajectory": {
                "turns": [{"role": t.role, "content": t.content} for t in trajectory.turns],
                "tool_calls": [tc.model_dump() for tc in trajectory.tool_calls],
            },
            "metadata": {
                "total_turns": len(trajectory.turns),
                "tool_calls_made": len(trajectory.tool_calls),
                "used_mcp": mcp_client is not None,
            },
        }
        sessions_path = output_dir / "collected_sessions.json"
        if sessions_path.exists():
            try:
                existing = json.loads(sessions_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    existing.append(traj_data)
                else:
                    existing = [traj_data]
            except Exception:
                existing = [traj_data]
        else:
            existing = [traj_data]
        sessions_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        print(f"💾 Saved trajectory session to {sessions_path}")
    else:
        print("\n❌ Failed to collect a valid trajectory")
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
