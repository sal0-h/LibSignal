#!/usr/bin/env python3
"""
Network statistics for a SUMO .net.xml (used for the Doha Corniche benchmark).

Usage:
    python extras/doha_network_stats.py
    python extras/doha_network_stats.py --net data/raw_data/doha_corniche/doha_corniche.net.xml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_NET = REPO / "data" / "raw_data" / "doha_corniche" / "doha_corniche.net.xml"
DEFAULT_SOURCE = REPO / "data" / "raw_data" / "doha_corniche" / "SOURCE.json"
DEFAULT_OUT = REPO / "data" / "raw_data" / "doha_corniche" / "STATS.json"


def _ensure_sumolib():
    home = os.environ.get("SUMO_HOME")
    if home:
        tools = str(Path(home) / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
    import sumolib  # noqa: F401
    return sumolib


def _green_states(phases):
    greens = []
    for ph in phases:
        state = ph.get("state") or ""
        if "y" in state:
            continue
        if state.count("r") + state.count("s") == len(state):
            continue
        greens.append(state)
    return greens


def _weak_components(net):
    adj = defaultdict(set)
    nodes = set()
    for e in net.getEdges():
        if e.getID().startswith(":"):
            continue
        a, b = e.getFromNode().getID(), e.getToNode().getID()
        adj[a].add(b)
        adj[b].add(a)
        nodes.add(a)
        nodes.add(b)
    seen = set()
    sizes = []
    for n in nodes:
        if n in seen:
            continue
        q = deque([n])
        seen.add(n)
        sz = 0
        while q:
            u = q.popleft()
            sz += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        sizes.append(sz)
    return sizes


def _median(xs):
    if not xs:
        return None
    return float(statistics.median(xs))


def collect(net_path: Path) -> dict:
    sumolib = _ensure_sumolib()
    net = sumolib.net.readNet(str(net_path), withInternal=True)
    loc = net.getLocationOffset() if hasattr(net, "getLocationOffset") else None
    bbox_xy = net.getBBoxXY()
    conv = [bbox_xy[0][0], bbox_xy[0][1], bbox_xy[1][0], bbox_xy[1][1]]

    # origBoundary from XML
    root = ET.parse(net_path).getroot()
    loc_el = root.find("location")
    orig = loc_el.get("origBoundary") if loc_el is not None else None
    conv_b = loc_el.get("convBoundary") if loc_el is not None else None
    proj = loc_el.get("projParameter") if loc_el is not None else None

    junctions = [n for n in net.getNodes() if n.getType() != "internal"]
    edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
    lanes = [ln for e in edges for ln in e.getLanes()]
    lane_len = sum(ln.getLength() for ln in lanes)
    n_oneway = 0
    for e in edges:
        eid = e.getID()
        rev = eid[1:] if eid.startswith("-") else "-" + eid
        try:
            net.getEdge(rev)
        except Exception:
            n_oneway += 1

    src = snk = 0
    boundary = []
    for e in edges:
        incoming = [x for x in e.getFromNode().getIncoming() if not x.getID().startswith(":")]
        outgoing = [x for x in e.getToNode().getOutgoing() if not x.getID().startswith(":")]
        is_src = len(incoming) == 0
        is_snk = len(outgoing) == 0
        if is_src:
            src += 1
        if is_snk:
            snk += 1
        if is_src or is_snk:
            boundary.append(e.getID())

    comps = _weak_components(net)

    tls_xml = {}
    for logic in root.findall("tlLogic"):
        tid = logic.get("id")
        phases = list(logic.findall("phase"))
        greens = _green_states(phases)
        tls_xml[tid] = {
            "type": logic.get("type"),
            "n_phases_raw": len(phases),
            "n_green": len(greens),
            "green_lengths": [len(s) for s in greens],
            "min_duration": min(float(p.get("duration")) for p in phases) if phases else None,
            "joined": tid.startswith("joined") or tid.startswith("GS_cluster") or tid.startswith("cluster_"),
            "guessed": tid.startswith("GS_"),
        }

    # incoming/outgoing lanes from connections in net.xml
    in_by_tl = defaultdict(set)
    out_by_tl = defaultdict(set)
    for conn in root.findall("connection"):
        tl = conn.get("tl")
        if not tl:
            continue
        fr, to_ = conn.get("from"), conn.get("to")
        if fr:
            in_by_tl[tl].add(f"{fr}_{conn.get('fromLane')}")
        if to_:
            out_by_tl[tl].add(f"{to_}_{conn.get('toLane')}")

    in_counts = []
    out_counts = []
    green_counts = []
    for tid, info in tls_xml.items():
        ic = len(in_by_tl.get(tid, ()))
        oc = len(out_by_tl.get(tid, ()))
        info["n_in_lanes"] = ic
        info["n_out_lanes"] = oc
        in_counts.append(ic)
        out_counts.append(oc)
        green_counts.append(info["n_green"])

    jtypes = Counter(n.getType() for n in junctions)

    stats = {
        "net": str(net_path.relative_to(REPO)),
        "conv_boundary_xy_m": conv_b,
        "orig_boundary_lonlat": orig,
        "proj": proj,
        "extent_m": {
            "width": round(conv[2] - conv[0], 1),
            "height": round(conv[3] - conv[1], 1),
        },
        "n_junctions_noninternal": len(junctions),
        "junction_types": dict(jtypes),
        "n_traffic_lights": len(tls_xml),
        "n_edges": len(edges),
        "n_lanes": len(lanes),
        "total_lane_length_km": round(lane_len / 1000.0, 3),
        "n_oneway_edges": n_oneway,
        "weak_connected_components": {
            "count": len(comps),
            "sizes": sorted(comps, reverse=True)[:10],
        },
        "source_edges": src,
        "sink_edges": snk,
        "n_boundary_edges": len(set(boundary)),
        "tls_green_phase_histogram": dict(Counter(green_counts)),
        "tls_green_phases": {
            "min": min(green_counts) if green_counts else None,
            "median": _median(green_counts),
            "max": max(green_counts) if green_counts else None,
        },
        "tls_incoming_lanes": {
            "min": min(in_counts) if in_counts else None,
            "median": _median(in_counts),
            "max": max(in_counts) if in_counts else None,
        },
        "tls_outgoing_lanes": {
            "min": min(out_counts) if out_counts else None,
            "median": _median(out_counts),
            "max": max(out_counts) if out_counts else None,
        },
        "n_joined_or_cluster_tls": sum(1 for v in tls_xml.values() if v["joined"]),
        "n_guessed_tls": sum(1 for v in tls_xml.values() if v["guessed"]),
        "traffic_lights": tls_xml,
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=Path, default=DEFAULT_NET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    stats = collect(args.net)
    if DEFAULT_SOURCE.exists():
        src = json.loads(DEFAULT_SOURCE.read_text())
        stats["bbox_west_south_east_north"] = src.get("bbox_west_south_east_north")
        stats["sumo_version"] = src.get("sumo_version")
        stats["acquisition_date_utc"] = src.get("acquisition_date_utc")
    args.out.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: stats[k] for k in stats if k != "traffic_lights"}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
