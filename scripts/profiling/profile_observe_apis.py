#!/usr/bin/env python3
"""Micro-profile Intersection.observe sub-calls (getNextTLS, getSpeed, etc.)."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def main():
    sys.argv = [
        "run.py",
        "--agent",
        "maxpressure",
        "--world",
        "sumo",
        "--network",
        "sumo1x1",
        "--seed",
        "42",
        "--ngpu",
        "-1",
        "--prefix",
        "obs_micro",
    ]
    from run import Runner, parser
    from common.registry import Registry
    import world.world_sumo as ws

    args = parser.parse_args()
    runner = Runner(args)
    from utils.logger import setup_logging

    logger = setup_logging(30)
    trainer = Registry.mapping["trainer_mapping"]["tsc"](logger)
    world = trainer.world
    env = trainer.env
    agents = trainer.agents

    totals = defaultdict(float)
    counts = defaultdict(int)
    eng = world.eng

    # Monkeypatch eng.vehicle methods used in observe
    orig = {
        "getWaitingTime": eng.vehicle.getWaitingTime,
        "getSpeed": eng.vehicle.getSpeed,
        "getLanePosition": eng.vehicle.getLanePosition,
        "getNextTLS": eng.vehicle.getNextTLS,
        "getLastStepVehicleIDs": eng.lane.getLastStepVehicleIDs,
        "simulationStep": eng.simulationStep,
    }

    def wrap(name, fn):
        def wrapped(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                totals[name] += time.perf_counter() - t0
                counts[name] += 1

        return wrapped

    eng.vehicle.getWaitingTime = wrap("veh.getWaitingTime", orig["getWaitingTime"])
    eng.vehicle.getSpeed = wrap("veh.getSpeed", orig["getSpeed"])
    eng.vehicle.getLanePosition = wrap("veh.getLanePosition", orig["getLanePosition"])
    eng.vehicle.getNextTLS = wrap("veh.getNextTLS", orig["getNextTLS"])
    eng.lane.getLastStepVehicleIDs = wrap(
        "lane.getLastStepVehicleIDs", orig["getLastStepVehicleIDs"]
    )
    # Also wrap trajectory-related
    orig_allowed = eng.vehicle.getAllowedSpeed
    eng.vehicle.getAllowedSpeed = wrap("veh.getAllowedSpeed", orig_allowed)
    orig_vidlist = eng.vehicle.getIDList
    eng.vehicle.getIDList = wrap("veh.getIDList", orig_vidlist)

    # Run one episode of test
    trainer.steps = 3600
    trainer.test_steps = 3600
    t0 = time.perf_counter()
    trainer.test()
    wall = time.perf_counter() - t0

    rows = []
    for k, v in sorted(totals.items(), key=lambda x: -x[1]):
        rows.append(
            {
                "name": k,
                "seconds": round(v, 4),
                "calls": counts[k],
                "us_per_call": round(1e6 * v / counts[k], 2) if counts[k] else 0,
            }
        )
    out = {"wall_seconds": round(wall, 4), "api_calls": rows}
    path = "extras/efficiency_audit/observe_api_micro.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
