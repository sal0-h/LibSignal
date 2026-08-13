#!/usr/bin/env python3
"""grid4x4 lane-id parsing must match the rsplit helper (single-digit indices)."""
from pathlib import Path
import os
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("SUMO_HOME", os.environ.get("SUMO_HOME", ""))

from world.world_sumo import sumo_edge_id_from_lane, sumo_lane_index


def legacy_edge(lane_id):
    return lane_id[:-2]


def legacy_index(lane_id):
    return int(lane_id[-1])


def main():
    net = REPO / "data/raw_data/grid4x4/grid4x4.net.xml"
    text = net.read_text()
    # crude: lane id="..."
    import re
    ids = re.findall(r'<lane id="([^"]+)"', text)
    assert ids, "no lanes in grid4x4.net.xml"
    mismatches = []
    for lid in ids:
        if sumo_edge_id_from_lane(lid) != legacy_edge(lid) or sumo_lane_index(lid) != legacy_index(lid):
            mismatches.append(lid)
    if mismatches:
        raise SystemExit(f"FAIL {len(mismatches)} grid4x4 lanes differ, e.g. {mismatches[:5]}")
    print(f"PASS {len(ids)} grid4x4 lanes: rsplit == [:-2] / int(id[-1])")


if __name__ == "__main__":
    main()
