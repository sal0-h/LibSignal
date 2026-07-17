#!/usr/bin/env python3
"""
Generate crossing_proxy_lanes.json for a SUMO network.

Maps each traffic-light id to incoming lane IDs that should lose service during
concurrent-walk proxy events on through-green phases (NEMA-style):
  - phase 0 (NS through): halt E + W incoming lanes (ped crosses E-W street)
  - phase 4 (EW through): halt N + S incoming lanes (ped crosses N-S street)

Phase indices follow mplight.yml signal_config.phase_pairs for grid4x4 / cologne3.

Usage:
    source .venv/bin/activate
    python extras/gen_crossing_proxy_lanes.py
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

NETWORKS = {
    "grid4x4": {
        "net": REPO / "data/raw_data/grid4x4/grid4x4.net.xml",
        "rou": REPO / "data/raw_data/grid4x4/grid4x4.rou.xml",
        "out": REPO / "data/raw_data/grid4x4/crossing_proxy_lanes.json",
        "through_phases": {
            "0": ("E", "W"),
            "4": ("N", "S"),
        },
    },
}

THROUGH_PHASE_NOTES = {
    "0": "NS_through_green — proxy halt on E/W incoming (E-W crosswalk service)",
    "4": "EW_through_green — proxy halt on N/S incoming (N-S crosswalk service)",
}


def _cardinal(dx: float, dy: float) -> str:
    ang = math.degrees(math.atan2(dy, dx))
    if ang >= 45.0:
        return "N"
    if ang >= -45.0:
        return "E"
    if ang >= -135.0:
        return "S"
    return "W"


def _approach_direction(lane_id: str, cx: float, cy: float, lane_shape_fn) -> str:
    shape = lane_shape_fn(lane_id)
    fx, fy = shape[0]
    return _cardinal(cx - fx, cy - fy)


def build_lane_map(net_path: Path, rou_path: Path, through_phases: dict) -> dict:
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
            by_dir = {"N": set(), "E": set(), "S": set(), "W": set()}

            for links in traci.trafficlight.getControlledLinks(tls_id):
                if not links:
                    continue
                from_lane = links[0][0]
                edge_id = from_lane.rsplit("_", 1)[0]
                if edge_id.startswith(":"):
                    continue
                direction = _approach_direction(
                    from_lane, cx, cy, traci.lane.getShape
                )
                by_dir[direction].add(from_lane)

            phase_map = {}
            for phase_key, dirs in through_phases.items():
                lanes = sorted(
                    lane
                    for d in dirs
                    for lane in by_dir[d]
                )
                phase_map[phase_key] = lanes
            intersections[tls_id] = phase_map
    finally:
        traci.close()

    return {
        "network": net_path.stem.replace(".net", ""),
        "through_phases": list(through_phases.keys()),
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
    data = build_lane_map(spec["net"], spec["rou"], spec["through_phases"])
    spec["out"].parent.mkdir(parents=True, exist_ok=True)
    with open(spec["out"], "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    n_tls = len(data["intersections"])
    sample = next(iter(data["intersections"].values()))
    n_lanes = len(sample.get("0", [])) + len(sample.get("4", []))
    print(f"Wrote {spec['out']} — {n_tls} TLS, ~{n_lanes} halt lanes per junction (both phases)")


if __name__ == "__main__":
    main()
