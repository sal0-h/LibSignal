#!/usr/bin/env python3
"""
Hub-centric synthetic OD demand for the Doha Corniche SUMO network.

Builds TAZ from the real topology (fringe / internal / Corniche hub), then uses
the same od2trips + duarouter pipeline as extras/gen_od_hub_demand_grid4x4.py.

Outputs:
  data/raw_data/doha_corniche/od_hubs/taz.xml
  data/raw_data/doha_corniche/od_hubs/demand_set/{fixed_1800,train_*,hold_*}.rou.xml
  data/raw_data/doha_corniche/doha_corniche.rou.xml   (3600 s default / smoke)

Usage:
    python extras/gen_od_hub_demand_doha.py
    python extras/gen_od_hub_demand_doha.py --validate-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NET = REPO / "data/raw_data/doha_corniche/doha_corniche.net.xml"
TAZ = REPO / "data/raw_data/doha_corniche/od_hubs/taz.xml"
OUT_DIR = REPO / "data/raw_data/doha_corniche/od_hubs/demand_set"
REL_PREFIX = "raw_data/doha_corniche/od_hubs/demand_set"
SMOKE_ROU = REPO / "data/raw_data/doha_corniche/doha_corniche.rou.xml"

HUB_NAME_SUBSTR = "الكورنيش"  # Al Corniche Street
MIN_EDGE_M = 25.0
MIN_ROUTE_EDGES = 3
MIN_ROUTE_M = 400.0
BOUNDARY_MARGIN_M = 120.0
N_TRAIN = 10
N_HOLD = 3
TARGET_VEHICLES_1800 = 800
TARGET_VEHICLES_3600 = 1400
TRAIN_SEED_BASE = 2000
HOLD_SEED_BASE = 9000
FIXED_SEED = 42
SMOKE_SEED = 7

TIMELINE_1800 = [
    (0, 300, 0.08),
    (300, 600, 0.12),
    (600, 900, 0.22),
    (900, 1200, 0.28),
    (1200, 1500, 0.18),
    (1500, 1800, 0.12),
]
TIMELINE_3600 = [
    (0, 300, 0.05),
    (300, 600, 0.06),
    (600, 900, 0.08),
    (900, 1200, 0.10),
    (1200, 1500, 0.12),
    (1500, 1800, 0.14),
    (1800, 2100, 0.12),
    (2100, 2400, 0.10),
    (2400, 2700, 0.08),
    (2700, 3000, 0.06),
    (3000, 3300, 0.05),
    (3300, 3600, 0.04),
]


def _load_grid4x4_mod():
    path = REPO / "extras" / "gen_od_hub_demand_grid4x4.py"
    spec = importlib.util.spec_from_file_location("od_hub_grid4x4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_sumolib():
    home = os.environ.get("SUMO_HOME")
    if home:
        tools = str(Path(home) / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
    import sumolib
    return sumolib


def _edge_mid(e):
    shape = e.getShape()
    xs = [p[0] for p in shape]
    ys = [p[1] for p in shape]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _passenger_edges(net):
    out = []
    for e in net.getEdges():
        if e.getID().startswith(":"):
            continue
        if e.getLength() < MIN_EDGE_M:
            continue
        if not e.allows("passenger"):
            continue
        out.append(e)
    return out


def _is_fringe(e, bbox, margin):
    xmin, ymin, xmax, ymax = bbox
    x, y = _edge_mid(e)
    if x <= xmin + margin or x >= xmax - margin or y <= ymin + margin or y >= ymax - margin:
        return True
    incoming = [x for x in e.getFromNode().getIncoming() if not x.getID().startswith(":")]
    outgoing = [x for x in e.getToNode().getOutgoing() if not x.getID().startswith(":")]
    return (len(incoming) == 0) or (len(outgoing) == 0)


def _compass(x, y, cx, cy):
    ang = math.degrees(math.atan2(x - cx, y - cy))  # 0 = north, +east
    if ang < 0:
        ang += 360
    if ang < 45 or ang >= 315:
        return "north"
    if ang < 135:
        return "east"
    if ang < 225:
        return "south"
    return "west"


def _internal_cell(x, y, bbox, nx=2, ny=2):
    xmin, ymin, xmax, ymax = bbox
    xi = min(nx - 1, max(0, int((x - xmin) / (xmax - xmin) * nx)))
    yi = min(ny - 1, max(0, int((y - ymin) / (ymax - ymin) * ny)))
    names_x = ["W", "E"]
    names_y = ["S", "N"]
    return f"internal_{names_y[yi]}{names_x[xi]}"


def build_taz(net_path: Path, taz_path: Path):
    sumolib = _ensure_sumolib()
    net = sumolib.net.readNet(str(net_path))
    bb = net.getBBoxXY()
    bbox = (bb[0][0], bb[0][1], bb[1][0], bb[1][1])
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    edges = _passenger_edges(net)

    zones = defaultdict(lambda: {"sources": [], "sinks": []})
    hub_ids = set()
    for e in edges:
        name = e.getName() if hasattr(e, "getName") else ""
        if name and HUB_NAME_SUBSTR in name:
            hub_ids.add(e.getID())
            zones["hub_corniche"]["sources"].append(e.getID())
            zones["hub_corniche"]["sinks"].append(e.getID())

    if len(zones["hub_corniche"]["sources"]) < 8:
        raise SystemExit(
            f"Corniche hub too small ({len(zones['hub_corniche']['sources'])} edges). "
            "Check street names in the net."
        )

    for e in edges:
        eid = e.getID()
        if eid in hub_ids:
            continue
        x, y = _edge_mid(e)
        incoming = [x for x in e.getFromNode().getIncoming() if not x.getID().startswith(":")]
        outgoing = [x for x in e.getToNode().getOutgoing() if not x.getID().startswith(":")]
        if _is_fringe(e, bbox, BOUNDARY_MARGIN_M):
            zid = "fringe_" + _compass(x, y, cx, cy)
            if len(incoming) == 0:
                zones[zid]["sources"].append(eid)
            if len(outgoing) == 0:
                zones[zid]["sinks"].append(eid)
            # bbox-margin edges can be both
            if incoming and outgoing:
                zones[zid]["sources"].append(eid)
                zones[zid]["sinks"].append(eid)
        else:
            zid = _internal_cell(x, y, bbox)
            zones[zid]["sources"].append(eid)
            zones[zid]["sinks"].append(eid)

    # Dedup while preserving order
    for z in zones.values():
        z["sources"] = list(dict.fromkeys(z["sources"]))
        z["sinks"] = list(dict.fromkeys(z["sinks"]))

    # Drop empty zones
    zones = {k: v for k, v in zones.items() if v["sources"] and v["sinks"]}
    fringe = [k for k in zones if k.startswith("fringe_")]
    internal = [k for k in zones if k.startswith("internal_")]
    if not fringe or not internal or "hub_corniche" not in zones:
        raise SystemExit(f"TAZ incomplete: keys={sorted(zones)}")

    taz_path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("tazs")
    for zid in sorted(zones):
        taz = ET.SubElement(root, "taz", id=zid)
        for eid in zones[zid]["sources"]:
            ET.SubElement(taz, "tazSource", id=eid, weight="1.0")
        for eid in zones[zid]["sinks"]:
            ET.SubElement(taz, "tazSink", id=eid, weight="1.0")
    ET.ElementTree(root).write(taz_path, encoding="utf-8", xml_declaration=True)
    summary = {k: {"n_src": len(v["sources"]), "n_snk": len(v["sinks"])} for k, v in zones.items()}
    print(f"wrote {taz_path} zones={summary}")
    return zones


def _filter_short_routes(rou_path: Path, net) -> int:
    tree = ET.parse(rou_path)
    root = tree.getroot()
    kept = []
    edge_len = {e.getID(): e.getLength() for e in net.getEdges()}
    for veh in list(root.findall("vehicle")):
        route = veh.find("route")
        if route is None or not route.get("edges"):
            root.remove(veh)
            continue
        edges = route.get("edges").split()
        length = sum(edge_len.get(e, 0.0) for e in edges)
        if len(edges) < MIN_ROUTE_EDGES or length < MIN_ROUTE_M:
            root.remove(veh)
            continue
        kept.append(veh)
    for i, veh in enumerate(kept):
        veh.set("id", str(i))
    tree.write(rou_path, encoding="utf-8", xml_declaration=True)
    return len(kept)


def _generate_with(mod, seed, out_path, target, timeline):
    mod.NET = NET
    mod.TAZ = TAZ
    mod.TARGET_VEHICLES = target
    mod.TIMELINE_BINS = timeline
    mod.HUB_ID = "hub_corniche"
    od2trips = mod._find_sumo_binary("od2trips")
    duarouter = mod._find_sumo_binary("duarouter")
    zones = mod._load_taz(TAZ)
    centroids = mod._zone_xy(NET, zones)
    stats = mod.generate_one(seed, out_path, zones, centroids, od2trips, duarouter)
    sumolib = _ensure_sumolib()
    net = sumolib.net.readNet(str(NET))
    kept = _filter_short_routes(out_path, net)
    stats["routed_vehicles"] = kept
    stats["seed"] = seed
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--taz-only", action="store_true")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-hold", type=int, default=N_HOLD)
    args = parser.parse_args()

    mod = _load_grid4x4_mod()
    man_path = OUT_DIR / "manifest.json"

    if args.validate_only:
        if not man_path.exists():
            raise SystemExit(f"missing {man_path}")
        # Patch module paths used by validator
        mod.TAZ = TAZ
        mod.OUT_DIR = OUT_DIR
        ok = mod.validate_demand_set(json.loads(man_path.read_text()))
        raise SystemExit(0 if ok else 1)

    if not NET.exists():
        raise SystemExit(f"missing {NET}; run extras/build_doha_network.py first")

    build_taz(NET, TAZ)
    if args.taz_only:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_stats = []

    print(f"generating 3600s smoke/default seed={SMOKE_SEED}...")
    st = _generate_with(mod, SMOKE_SEED, SMOKE_ROU, TARGET_VEHICLES_3600, TIMELINE_3600)
    st["role"] = "smoke"
    st["file"] = "raw_data/doha_corniche/doha_corniche.rou.xml"
    file_stats.append(st)
    print(f"  -> {st['routed_vehicles']} veh hub_touch={st['hub_touch_frac']:.2f}")

    print(f"generating fixed_1800 seed={FIXED_SEED}...")
    fixed_path = OUT_DIR / "fixed_1800.rou.xml"
    st = _generate_with(mod, FIXED_SEED, fixed_path, TARGET_VEHICLES_1800, TIMELINE_1800)
    st["role"] = "fixed"
    st["file"] = f"{REL_PREFIX}/fixed_1800.rou.xml"
    file_stats.append(st)
    print(f"  -> {st['routed_vehicles']} veh hub_touch={st['hub_touch_frac']:.2f}")

    train_files = []
    for i in range(args.n_train):
        seed = TRAIN_SEED_BASE + i
        path = OUT_DIR / f"train_{i:02d}.rou.xml"
        print(f"generating train_{i:02d} seed={seed}...")
        st = _generate_with(mod, seed, path, TARGET_VEHICLES_1800, TIMELINE_1800)
        st["role"] = "train"
        st["file"] = f"{REL_PREFIX}/train_{i:02d}.rou.xml"
        file_stats.append(st)
        train_files.append(st["file"])
        print(f"  -> {st['routed_vehicles']} veh hub_touch={st['hub_touch_frac']:.2f}")

    hold_files = []
    for i in range(args.n_hold):
        seed = HOLD_SEED_BASE + i
        path = OUT_DIR / f"hold_{i:02d}.rou.xml"
        print(f"generating hold_{i:02d} seed={seed}...")
        st = _generate_with(mod, seed, path, TARGET_VEHICLES_1800, TIMELINE_1800)
        st["role"] = "hold"
        st["file"] = f"{REL_PREFIX}/hold_{i:02d}.rou.xml"
        file_stats.append(st)
        hold_files.append(st["file"])
        print(f"  -> {st['routed_vehicles']} veh hub_touch={st['hub_touch_frac']:.2f}")

    manifest = {
        "network": "sumo_doha",
        "episode_len": 1800,
        "taz": "raw_data/doha_corniche/od_hubs/taz.xml",
        "fixed": f"{REL_PREFIX}/fixed_1800.rou.xml",
        "train_set": train_files,
        "heldout": hold_files,
        "smoke": "raw_data/doha_corniche/doha_corniche.rou.xml",
        "routing": "od2trips + duarouter_shortest_path",
        "synthetic": True,
        "note": "Synthetic demand on a real OSM topology. Not calibrated Doha counts.",
        "file_stats": file_stats,
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {man_path}")
    mod.TAZ = TAZ
    mod.validate_demand_set(manifest)


if __name__ == "__main__":
    main()
