#!/usr/bin/env python3
"""Fail loudly if a SUMO network cannot be represented as LibSignal TSC agents.

Checks match world_sumo.World.generate_valid_phase:
  drop states containing 'y'; drop all-red / all-s; require >= 1 remaining green
  and >= 1 controlled connection.

Usage:
    python extras/validate_sumo_tls.py --net data/raw_data/doha_corniche/doha_corniche.net.xml
    python extras/validate_sumo_tls.py --net data/raw_data/grid4x4/grid4x4.net.xml
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def libsignal_green_states(phases):
    seen = []
    for ph in phases:
        state = ph.get("state") or ""
        if state not in seen:
            seen.append(state)
    greens = []
    for state in seen:
        if "y" in state:
            continue
        if state.count("r") + state.count("s") == len(state):
            continue
        greens.append(state)
    return greens


def validate_net_xml(net_path: Path) -> list[str]:
    errors = []
    warnings = []
    root = ET.parse(net_path).getroot()
    logics = list(root.findall("tlLogic"))
    if not logics:
        return ["no <tlLogic> elements in network"]

    conns_by_tl = Counter()
    for conn in root.findall("connection"):
        tl = conn.get("tl")
        if tl:
            conns_by_tl[tl] += 1

    green_hist = Counter()
    for logic in logics:
        tid = logic.get("id")
        phases = list(logic.findall("phase"))
        if not phases:
            errors.append(f"{tid}: empty tlLogic")
            continue
        lengths = {len(p.get("state") or "") for p in phases}
        if len(lengths) != 1:
            errors.append(f"{tid}: mixed phase state lengths {lengths}")
        greens = libsignal_green_states(phases)
        green_hist[len(greens)] += 1
        if not greens:
            errors.append(
                f"{tid}: 0 green phases after LibSignal yellow/all-red filter"
            )
        if conns_by_tl.get(tid, 0) == 0:
            errors.append(f"{tid}: no controlled <connection tl=...> entries")
        durations = [float(p.get("duration")) for p in phases]
        if durations and min(durations) < 2:
            warnings.append(
                f"{tid}: min phase duration {min(durations):g}s "
                f"(LibSignal yellow_phase_time = min(durations); known legacy heuristic)"
            )
        for i, st in enumerate(greens):
            if "G" not in st and "s" not in st:
                warnings.append(
                    f"{tid} green[{i}] has no G/s (only permissive g). "
                    "MaxPressure maps G|s only — existing semantics, empty lanelinks for this phase."
                )

    print(f"tls={len(logics)} green_phase_histogram={dict(sorted(green_hist.items()))}")
    n_joined = sum(
        1
        for logic in logics
        if logic.get("id", "").startswith(("joined", "cluster", "GS_cluster"))
    )
    print(f"joined_or_cluster_tls={n_joined}")
    if len(green_hist) > 1:
        warnings.append(
            f"heterogeneous action counts {dict(green_hist)}. "
            "FixedTime/MaxPressure/DQN/PressLight use per-intersection Discrete(n). "
            "CoLight uses Discrete(max n) without masking — invalid actions possible."
        )
    for w in warnings:
        print(f"WARN {w}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", type=Path, required=True)
    args = parser.parse_args()
    if not args.net.exists():
        raise SystemExit(f"missing {args.net}")
    errors = validate_net_xml(args.net)
    if errors:
        print("FAIL")
        for err in errors:
            print(f"  {err}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
