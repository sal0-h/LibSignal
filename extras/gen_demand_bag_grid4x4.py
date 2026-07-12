#!/usr/bin/env python3
"""
Build a Grid4x4 demand bag for short-episode (1200 s) generalization experiments.

From the baseline hour-long movie (grid4x4.rou.xml):
  - fixed_1200.rou.xml  — vehicles with depart < 1200 (same first 20 min)
  - train_00..train_09  — same intensity (#vehicles + depart times), resampled routes
  - hold_00..hold_02    — held-out variants (disjoint seeds)

Route resampling keeps the temporal demand profile identical while changing
which OD paths appear — so agents cannot memorize one traffic movie.

Usage:
    python extras/gen_demand_bag_grid4x4.py
"""
from __future__ import annotations

import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/raw_data/grid4x4/grid4x4.rou.xml"
OUT_DIR = REPO / "data/raw_data/grid4x4/demand_bag"
EPISODE_LEN = 1200
N_TRAIN = 10
N_HOLD = 3
TRAIN_SEED_BASE = 1000
HOLD_SEED_BASE = 9000


def _parse_vehicles(src: Path):
    tree = ET.parse(src)
    root = tree.getroot()
    vtype = root.find("vType")
    vehicles = []
    for veh in root.findall("vehicle"):
        route = veh.find("route")
        vehicles.append(
            {
                "depart": int(float(veh.get("depart"))),
                "edges": route.get("edges") if route is not None else "",
            }
        )
    return vtype, vehicles


def _write_rou(path: Path, vtype_el, vehicles):
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set(
        "xsi:noNamespaceSchemaLocation",
        "http://sumo.dlr.de/xsd/routes_file.xsd",
    )
    if vtype_el is not None:
        vt = ET.SubElement(root, "vType")
        for k, v in vtype_el.attrib.items():
            vt.set(k, v)
    for i, veh in enumerate(vehicles):
        v = ET.SubElement(root, "vehicle")
        v.set("id", str(i))
        v.set("depart", str(veh["depart"]))
        r = ET.SubElement(v, "route")
        r.set("edges", veh["edges"])
    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _make_variant(base_window, route_pool, seed: int):
    """Same depart schedule as base_window; routes sampled from full pool."""
    rng = random.Random(seed)
    routes = [v["edges"] for v in route_pool]
    out = []
    for veh in base_window:
        out.append({"depart": veh["depart"], "edges": rng.choice(routes)})
    return out


def main():
    if not SRC.exists():
        raise SystemExit(f"missing baseline routes: {SRC}")

    vtype, all_vehicles = _parse_vehicles(SRC)
    window = [v for v in all_vehicles if v["depart"] < EPISODE_LEN]
    window = sorted(window, key=lambda v: v["depart"])
    print(f"baseline vehicles={len(all_vehicles)}  window<{EPISODE_LEN}s={len(window)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed_path = OUT_DIR / "fixed_1200.rou.xml"
    _write_rou(fixed_path, vtype, window)
    print(f"wrote {fixed_path.relative_to(REPO)} ({len(window)} vehicles)")

    train_files = []
    for i in range(N_TRAIN):
        seed = TRAIN_SEED_BASE + i
        path = OUT_DIR / f"train_{i:02d}.rou.xml"
        _write_rou(path, vtype, _make_variant(window, all_vehicles, seed))
        train_files.append(f"raw_data/grid4x4/demand_bag/{path.name}")
        print(f"wrote {path.relative_to(REPO)} seed={seed}")

    hold_files = []
    for i in range(N_HOLD):
        seed = HOLD_SEED_BASE + i
        path = OUT_DIR / f"hold_{i:02d}.rou.xml"
        _write_rou(path, vtype, _make_variant(window, all_vehicles, seed))
        hold_files.append(f"raw_data/grid4x4/demand_bag/{path.name}")
        print(f"wrote {path.relative_to(REPO)} seed={seed}")

    manifest = {
        "network": "sumo4x4",
        "episode_len_s": EPISODE_LEN,
        "n_vehicles_per_file": len(window),
        "fixed": "raw_data/grid4x4/demand_bag/fixed_1200.rou.xml",
        "train_bag": train_files,
        "heldout": hold_files,
        "note": (
            "fixed = first 20 min of baseline movie; "
            "train/hold keep the same depart times but resample routes from the full hour pool."
        ),
    }
    man_path = OUT_DIR / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {man_path.relative_to(REPO)}")
    print("Done.")


if __name__ == "__main__":
    main()
