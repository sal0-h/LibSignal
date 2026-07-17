#!/usr/bin/env python3
"""
Generate crossing_proxy_lanes.json for a SUMO network.

Real-world NEMA concurrent pedestrian service (see docs/CROSSING_PROXY.md):
  - Phase 0 = N-S through green  + concurrent E-W crosswalk walk
  - Phase 4 = E-W through green  + concurrent N-S crosswalk walk

Each through-phase entry in the JSON lists **conflict_lanes**: incoming lanes whose
movement is on the crosswalk street AND has a non-red signal (G/g/s) during that
phase. Runtime always applies **actuated phase extension** (hold green) for T
seconds on ped call; conflict lanes are halted only when non-empty (permissive turns).

Phase indices are LibSignal **green phase actions** 0..7 (same as mplight.yml).

Usage:
    export SUMO_HOME=/path/to/sumo
    python extras/gen_crossing_proxy_lanes.py --network grid4x4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# grid4x4 / cologne3 NEMA: phase 0 = [1,7] NT,ST; phase 4 = [4,10] ET,WT
THROUGH_PHASES = ("0", "4")

THROUGH_PHASE_NOTES = {
    "0": "NS_through + concurrent E-W walk — extension + E-W conflict lanes (G/g/s)",
    "4": "EW_through + concurrent N-S walk — extension + N-S conflict lanes (G/g/s)",
}

NETWORKS = {
    "grid4x4": {
        "net": REPO / "data/raw_data/grid4x4/grid4x4.net.xml",
        "rou": REPO / "data/raw_data/grid4x4/grid4x4.rou.xml",
        "out": REPO / "data/raw_data/grid4x4/crossing_proxy_lanes.json",
        "through_phase_axis": {
            "0": "EW",  # ped crosses east-west street
            "4": "NS",
        },
    },
}


def _movement_axis(from_lane: str, cx: float, cy: float, lane_shape_fn) -> str:
    """Dominant travel axis of an approach lane toward the junction."""
    shape = lane_shape_fn(from_lane)
    fx, fy = shape[0]
    dx, dy = cx - fx, cy - fy
    if abs(dx) >= abs(dy):
        return "EW"
    return "NS"


def _green_phase_states(tls_id: str, eng) -> list[str]:
    """Green-only phase state strings in SUMO program order (matches action index)."""
    logic = eng.trafficlight.getAllProgramLogics(tls_id)[0]
    states = []
    for phase in logic.phases:
        state = phase.state
        if "y" in state:
            continue
        if state.count("r") + state.count("s") == len(state):
            continue
        states.append(state)
    return states


def build_lane_map(net_path: Path, rou_path: Path, through_phase_axis: dict) -> dict:
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError(
            "SUMO_HOME must be set (e.g. export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo)"
        )
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

    import sumolib
    import traci

    net = sumolib.net.readNet(str(net_path))
    sumo_bin = sumolib.checkBinary("sumo")
    traci.start(
        [
            sumo_bin,
            "-n",
            str(net_path),
            "-r",
            str(rou_path),
            "--no-warnings",
            "true",
            "--end",
            "1",
        ]
    )

    intersections = {}
    try:
        for tls_id in traci.trafficlight.getIDList():
            node = net.getNode(tls_id)
            cx, cy = node.getCoord()
            links = traci.trafficlight.getControlledLinks(tls_id)
            green_states = _green_phase_states(tls_id, traci)

            phase_map = {}
            for phase_key in THROUGH_PHASES:
                idx = int(phase_key)
                if idx >= len(green_states):
                    phase_map[phase_key] = []
                    continue
                state = green_states[idx]
                cross_axis = through_phase_axis[phase_key]
                conflict = set()
                for link_idx, link in enumerate(links):
                    if not link:
                        continue
                    if link_idx >= len(state):
                        break
                    if state[link_idx] not in ("G", "g", "s"):
                        continue
                    from_lane = link[0][0]
                    if from_lane.startswith(":"):
                        continue
                    if _movement_axis(from_lane, cx, cy, traci.lane.getShape) == cross_axis:
                        conflict.add(from_lane)
                phase_map[phase_key] = sorted(conflict)
            intersections[tls_id] = phase_map
    finally:
        traci.close()

    return {
        "network": net_path.stem.replace(".net", ""),
        "mechanism": "actuated_phase_extension_plus_crosswalk_conflicts",
        "through_phases": list(THROUGH_PHASES),
        "phase_semantics": THROUGH_PHASE_NOTES,
        "intersections": intersections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        default="grid4x4",
        choices=list(NETWORKS.keys()),
    )
    args = parser.parse_args()
    spec = NETWORKS[args.network]
    data = build_lane_map(spec["net"], spec["rou"], spec["through_phase_axis"])
    spec["out"].parent.mkdir(parents=True, exist_ok=True)
    with open(spec["out"], "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    n_tls = len(data["intersections"])
    sample = data["intersections"][next(iter(data["intersections"]))]
    n0 = len(sample.get("0", []))
    n4 = len(sample.get("4", []))
    print(
        f"Wrote {spec['out']} — {n_tls} TLS, "
        f"conflict lanes sample phase0={n0} phase4={n4} (extension always on call)"
    )


if __name__ == "__main__":
    main()
