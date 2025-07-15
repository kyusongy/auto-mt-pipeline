#!/usr/bin/env python3
"""Debug script for Phase-2 trajectory collection.

This helper bypasses Phase-1 blueprint generation entirely and loads a cached
`data/blueprint.json` file produced by an earlier run.  Only the *intent* (and
optionally pre-computed tool calls) is fed to the simulated human; this allows
rapid iteration on Phase-2 without incurring LLM cost or waiting for
blueprint validation.

Usage::

    python debug_trajectory.py

Make sure you have a valid `data/blueprint.json` file.  If you want to use a
custom location, set the env var `BLUEPRINT_PATH`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

# Local application imports – reuse existing helpers
from config import DEFAULT_LLM_CONFIG, DEFAULT_PIPELINE_CONFIG
from run_pipeline import get_tool_schemas
from core.blueprint.pipeline import Blueprint
from core.models import ToolCalling
from core.trajectory.pipeline import TrajectoryCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_blueprint(path: Path) -> Blueprint:
    """Load on-disk JSON and convert to `Blueprint` dataclass."""
    if not path.exists():
        raise FileNotFoundError(f"Blueprint file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    # Backwards-compatibility with older key names
    intent: str = data.get("intent") or data.get("user_intent") or ""
    if not intent:
        raise ValueError("Blueprint JSON is missing required 'intent' field")

    # Convert raw dicts into `ToolCalling` objects (optional)
    raw_actions: List[dict] = data.get("actions", [])
    actions: List[ToolCalling] = [
        ToolCalling(name=a["name"], arguments=a.get("arguments", {})) for a in raw_actions
    ]

    expected_outputs = data.get("outputs", [])

    return Blueprint(
        user_intent=intent,
        actions=actions,
        expected_outputs=expected_outputs,
        raw_response=None,
        thought_process=None,
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    blueprint_path = Path(os.getenv("BLUEPRINT_PATH", "data/blueprint.json"))
    blueprint = _load_blueprint(blueprint_path)

    # Verify AgentCortex integration is enabled (same behaviour as run_pipeline)
    from config import is_agentcortex_enabled, agentcortex_config
    if not is_agentcortex_enabled():
        print("❌ AgentCortex Integration: DISABLED")
        print("   To use this debug script, set AGENTCORTEX_ENABLED=true and provide valid service URLs")
        return

    print("=" * 60)
    print("🚀  Phase-2 Debug Script – Using Cached Blueprint (Plan+Execute Agent)")
    print("=" * 60)
    print(f"✔️  Loaded blueprint from: {blueprint_path}")
    print("→ Intent:", blueprint.user_intent)
    print(f"→ Pre-computed actions: {len(blueprint.actions)}\n")

    # Fetch tool schemas (MCP service or fallback)
    tools_schema, _ = get_tool_schemas()

    # Instantiate trajectory collector (same settings as main pipeline)
    collector = TrajectoryCollector(
        DEFAULT_LLM_CONFIG,          # Simulated human LLM cfg
        DEFAULT_LLM_CONFIG,          # Assistant LLM cfg
        tools_schema=tools_schema,
        debug=DEFAULT_PIPELINE_CONFIG.debug,
        bon_n=DEFAULT_PIPELINE_CONFIG.bon_n,
        use_plan_execute_agent=True,  # Match run_pipeline behaviour
    )

    print("💬 Starting trajectory collection…\n")
    trajectory = collector.collect(blueprint)

    if not trajectory:
        print("❌  Trajectory collection failed or was rejected.")
        return

    print(f"\n✅  Collected trajectory with {len(trajectory.turns)} turns")
    print("-" * 40)
    for t in trajectory.turns:
        if t.role != "system":  # Hide internal system messages
            tag = t.role.upper()
            print(f"[{tag}] {t.content}")

    print("\n🎉  Debug run complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user – exiting…")
