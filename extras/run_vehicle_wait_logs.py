#!/usr/bin/env python3
"""
Run a LibSignal TSC agent on SUMO and export per-vehicle waiting-time logs.

Analysis CSV/JSON go under extras/output/ (gitignored), not data/output_data/.

Usage (from repo root):
  python extras/run_vehicle_wait_logs.py --agent maxpressure --network sumo1x1 --seed 42
  python extras/run_vehicle_wait_logs.py --agent maxpressure --network sumo4x4 --seed 42
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

_saved_argv = sys.argv[:]
sys.argv = [_saved_argv[0]]
from run import Runner  # noqa: E402
sys.argv = _saved_argv

import task  # noqa: F401,E402
import trainer  # noqa: F401,E402
import agent  # noqa: F401,E402
import dataset  # noqa: F401,E402

from common.registry import Registry
from utils.logger import setup_logging

EXTRAS_DIR = os.path.join(PROJECT_ROOT, "extras")
DEFAULT_OUTPUT_ROOT = os.path.join(EXTRAS_DIR, "output")

# Match SUMO's near-stop threshold for getAccumulatedWaitingTime.
STOP_SPEED_THRESHOLD = 0.1


def resolve_output_dir(agent, network, seed, test_steps, run_name=None, output_root=None):
    """extras/output/<agent>/<network>/<run_name>/"""
    root = output_root or DEFAULT_OUTPUT_ROOT
    folder = run_name or f"seed{seed}_steps{test_steps}"
    return os.path.join(root, agent, network, folder)


def _default_interface(requested):
    if requested != "libsumo":
        return requested
    try:
        import libsumo  # noqa: F401
        return "libsumo"
    except ImportError:
        print("[Info] libsumo not available — using traci instead.")
        return "traci"


def _simulation_id_list(world, method_name):
    method = getattr(world.eng.simulation, method_name, None)
    if method is None:
        return []
    try:
        return list(method())
    except Exception:
        return []


def _free_flow_route_time(world, veh_id):
    """Minimum travel time if the vehicle drove at each edge's speed limit."""
    try:
        route = world.eng.vehicle.getRoute(veh_id)
        total = 0.0
        for edge_id in route:
            length = world.eng.edge.getLength(edge_id)
            lane_id = f"{edge_id}_0"
            speed = world.eng.lane.getMaxSpeed(lane_id)
            if speed > 0:
                total += length / speed
        return total
    except Exception:
        return None


def _snapshot_vehicle_metrics(world):
    """Capture per-vehicle SUMO metrics before env.step() removes departures."""
    pre_sumo_wait = {}
    pre_time_loss = {}
    pre_speed = {}
    for veh_id in world.eng.vehicle.getIDList():
        pre_sumo_wait[veh_id] = float(world.eng.vehicle.getAccumulatedWaitingTime(veh_id))
        try:
            pre_time_loss[veh_id] = float(world.eng.vehicle.getTimeLoss(veh_id))
        except Exception:
            pre_time_loss[veh_id] = 0.0
        pre_speed[veh_id] = float(world.eng.vehicle.getSpeed(veh_id))
    return pre_sumo_wait, pre_time_loss, pre_speed


def _update_custom_wait(custom_wait, pre_speed, dt):
    for veh_id, speed in pre_speed.items():
        if speed < STOP_SPEED_THRESHOLD:
            custom_wait[veh_id] = custom_wait.get(veh_id, 0.0) + dt


def _total_metric(per_step, cumulative, veh_id):
    return float(per_step.get(veh_id, 0.0)) + float(cumulative.get(veh_id, 0.0))


def _make_record(
    veh_id,
    enter_time,
    departure_time,
    travel_time,
    sumo_wait,
    custom_wait,
    time_loss,
    free_flow_time,
    trip_status,
    vehicle_type="",
):
    delay = None
    if free_flow_time is not None and travel_time > 0:
        delay = max(0.0, travel_time - free_flow_time)
    primary_wait = custom_wait
    return {
        "vehicle_id": veh_id,
        "vehicle_type": vehicle_type,
        "enter_time_s": round(enter_time, 3),
        "departure_time_s": round(departure_time, 3) if departure_time is not None else None,
        "travel_time_s": round(travel_time, 3),
        "accumulated_waiting_time_s": round(sumo_wait, 3),
        "custom_wait_s": round(custom_wait, 3),
        "time_loss_s": round(time_loss, 3),
        "free_flow_time_s": round(free_flow_time, 3) if free_flow_time is not None else "",
        "delay_s": round(delay, 3) if delay is not None else "",
        "waiting_fraction": round(primary_wait / travel_time, 4) if travel_time > 0 else 0.0,
        "completed_trip": trip_status == "completed",
        "trip_status": trip_status,
    }


def _vehicle_enter_time(world, veh_id, sim_time):
    return float(world.inside_vehicles.get(veh_id, sim_time))


def _vehicle_type(world, veh_id):
    """Return SUMO vType id for a vehicle, or '' if unavailable (hetero=false or departed)."""
    try:
        return str(world.eng.vehicle.getTypeID(veh_id))
    except Exception:
        return ""


def _collect_departed_records(
    world,
    pre_sumo_wait,
    pre_time_loss,
    custom_wait,
    cumulative_sumo_wait,
    cumulative_time_loss,
    free_flow_times,
    seen_logged,
    sim_time,
):
    records = []
    newly_completed = set(world.vehicles.keys()) - seen_logged
    for veh_id in sorted(newly_completed):
        travel_time = float(world.vehicles[veh_id])
        sumo_wait = _total_metric(pre_sumo_wait, cumulative_sumo_wait, veh_id)
        time_loss = _total_metric(pre_time_loss, cumulative_time_loss, veh_id)
        wait_custom = float(custom_wait.get(veh_id, 0.0))
        cumulative_sumo_wait.pop(veh_id, None)
        cumulative_time_loss.pop(veh_id, None)
        custom_wait.pop(veh_id, None)
        departure_time = sim_time
        enter_time = departure_time - travel_time
        records.append(
            _make_record(
                veh_id,
                enter_time,
                departure_time,
                travel_time,
                sumo_wait,
                wait_custom,
                time_loss,
                free_flow_times.pop(veh_id, None),
                "completed",
                vehicle_type=_vehicle_type(world, veh_id),
            )
        )
    return records, seen_logged | newly_completed


def _collect_removed_records(
    world,
    pre_sumo_wait,
    pre_time_loss,
    custom_wait,
    cumulative_sumo_wait,
    cumulative_time_loss,
    free_flow_times,
    seen_logged,
    sim_time,
    veh_ids,
):
    records = []
    newly_logged = set()
    for veh_id in sorted(set(veh_ids) - seen_logged):
        sumo_wait = _total_metric(pre_sumo_wait, cumulative_sumo_wait, veh_id)
        time_loss = _total_metric(pre_time_loss, cumulative_time_loss, veh_id)
        wait_custom = float(custom_wait.get(veh_id, 0.0))
        cumulative_sumo_wait.pop(veh_id, None)
        cumulative_time_loss.pop(veh_id, None)
        custom_wait.pop(veh_id, None)
        enter_time = _vehicle_enter_time(world, veh_id, sim_time)
        travel_time = sim_time - enter_time
        records.append(
            _make_record(
                veh_id,
                enter_time,
                sim_time,
                travel_time,
                sumo_wait,
                wait_custom,
                time_loss,
                free_flow_times.pop(veh_id, None),
                "removed",
                vehicle_type=_vehicle_type(world, veh_id),
            )
        )
        newly_logged.add(veh_id)
    return records, seen_logged | newly_logged


def _accumulate_teleport_metrics(
    teleported_ids,
    pre_sumo_wait,
    pre_time_loss,
    cumulative_sumo_wait,
    cumulative_time_loss,
):
    for veh_id in teleported_ids:
        cumulative_sumo_wait[veh_id] = cumulative_sumo_wait.get(veh_id, 0.0) + float(
            pre_sumo_wait.get(veh_id, 0.0)
        )
        cumulative_time_loss[veh_id] = cumulative_time_loss.get(veh_id, 0.0) + float(
            pre_time_loss.get(veh_id, 0.0)
        )


def _refresh_free_flow_times(world, free_flow_times):
    for veh_id in world.eng.vehicle.getIDList():
        fft = _free_flow_route_time(world, veh_id)
        if fft is not None:
            free_flow_times[veh_id] = fft


def _collect_remaining_records(
    world,
    custom_wait,
    cumulative_sumo_wait,
    cumulative_time_loss,
    free_flow_times,
    seen_logged,
    sim_time,
):
    records = []
    for veh_id in world.eng.vehicle.getIDList():
        if veh_id in seen_logged:
            continue
        sumo_wait = float(world.eng.vehicle.getAccumulatedWaitingTime(veh_id))
        sumo_wait += float(cumulative_sumo_wait.get(veh_id, 0.0))
        try:
            time_loss = float(world.eng.vehicle.getTimeLoss(veh_id))
        except Exception:
            time_loss = 0.0
        time_loss += float(cumulative_time_loss.get(veh_id, 0.0))
        wait_custom = float(custom_wait.get(veh_id, 0.0))
        enter_time = _vehicle_enter_time(world, veh_id, sim_time)
        travel_time = sim_time - enter_time
        records.append(
            _make_record(
                veh_id,
                enter_time,
                None,
                travel_time,
                sumo_wait,
                wait_custom,
                time_loss,
                free_flow_times.get(veh_id),
                "on_map_at_end",
                vehicle_type=_vehicle_type(world, veh_id),
            )
        )
    return records


def _metric_stats(records, field):
    values = [r[field] for r in records if r.get(field) not in (None, "")]
    if not values:
        return None
    values = [float(v) for v in values]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def run_with_vehicle_logs(runner, output_dir):
    """Run test episode and write per-vehicle waiting-time CSV."""
    trainer_obj = runner.trainer
    env = trainer_obj.env
    world = trainer_obj.world

    test_steps = Registry.mapping["trainer_mapping"]["setting"].param["test_steps"]
    action_interval = Registry.mapping["trainer_mapping"]["setting"].param["action_interval"]
    dt = float(world.step_length)

    trainer_obj.metric.clear()
    obs = env.reset()
    for ag in trainer_obj.agents:
        ag.reset()

    custom_wait = {}
    cumulative_sumo_wait = {}
    cumulative_time_loss = {}
    free_flow_times = {}
    seen_logged = set()
    vehicle_records = []
    dones = [False]

    i = 0
    while i < test_steps:
        if i % action_interval == 0:
            phases = np.stack([ag.get_phase() for ag in trainer_obj.agents])
            actions = np.stack(
                [
                    ag.get_action(obs[idx], phases[idx], test=True)
                    for idx, ag in enumerate(trainer_obj.agents)
                ]
            )

            rewards_list = []
            for _ in range(action_interval):
                _refresh_free_flow_times(world, free_flow_times)
                pre_sumo_wait, pre_time_loss, pre_speed = _snapshot_vehicle_metrics(world)
                _update_custom_wait(custom_wait, pre_speed, dt)

                obs, rewards, dones, _ = env.step(actions.flatten())

                sim_time = world.eng.simulation.getTime()

                step_records, seen_logged = _collect_departed_records(
                    world,
                    pre_sumo_wait,
                    pre_time_loss,
                    custom_wait,
                    cumulative_sumo_wait,
                    cumulative_time_loss,
                    free_flow_times,
                    seen_logged,
                    sim_time,
                )
                vehicle_records.extend(step_records)

                teleported = _simulation_id_list(world, "getTeleportingIDList")
                _accumulate_teleport_metrics(
                    teleported,
                    pre_sumo_wait,
                    pre_time_loss,
                    cumulative_sumo_wait,
                    cumulative_time_loss,
                )

                removed = _simulation_id_list(world, "getRemovedIDList")
                step_records, seen_logged = _collect_removed_records(
                    world,
                    pre_sumo_wait,
                    pre_time_loss,
                    custom_wait,
                    cumulative_sumo_wait,
                    cumulative_time_loss,
                    free_flow_times,
                    seen_logged,
                    sim_time,
                    removed,
                )
                vehicle_records.extend(step_records)

                i += 1
                rewards_list.append(np.stack(rewards))
                if i >= test_steps:
                    break

            if rewards_list:
                trainer_obj.metric.update(np.mean(rewards_list, axis=0))

        if all(dones):
            break

    sim_time = world.eng.simulation.getTime()
    vehicle_records.extend(
        _collect_remaining_records(
            world,
            custom_wait,
            cumulative_sumo_wait,
            cumulative_time_loss,
            free_flow_times,
            seen_logged,
            sim_time,
        )
    )

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "vehicle_waiting_times.csv")
    fieldnames = [
        "vehicle_id",
        "vehicle_type",
        "enter_time_s",
        "departure_time_s",
        "travel_time_s",
        "accumulated_waiting_time_s",
        "custom_wait_s",
        "time_loss_s",
        "free_flow_time_s",
        "delay_s",
        "waiting_fraction",
        "completed_trip",
        "trip_status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in vehicle_records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    completed = [r for r in vehicle_records if r.get("trip_status") == "completed"]
    incomplete = [r for r in vehicle_records if r.get("trip_status") != "completed"]

    status_counts = {}
    for rec in vehicle_records:
        status = rec.get("trip_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    meta = {
        "agent": runner.config["command"]["agent"],
        "world": runner.config["command"]["world"],
        "network": runner.config["command"]["network"],
        "prefix": runner.config["command"]["prefix"],
        "seed": runner.config["command"]["seed"],
        "test_steps": test_steps,
        "action_interval": action_interval,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_vehicles_logged": len(vehicle_records),
        "n_completed_trips": len(completed),
        "n_incomplete_trips": len(incomplete),
        "trip_status_counts": status_counts,
        "primary_fairness_metric": "custom_wait_s",
        "sumo_accumulated_wait_note": (
            "accumulated_waiting_time_s uses SUMO getAccumulatedWaitingTime(), which is "
            "capped by default --waiting-time-memory=100s. Do NOT use it for tail/fairness "
            "plots. custom_wait_s is uncapped (per-step speed < 0.1 m/s bookkeeping)."
        ),
        "waiting_time_stats_completed": _metric_stats(completed, "accumulated_waiting_time_s"),
        "waiting_time_stats_incomplete": _metric_stats(incomplete, "accumulated_waiting_time_s"),
        "waiting_time_stats_all": _metric_stats(vehicle_records, "accumulated_waiting_time_s"),
        "custom_wait_stats_completed": _metric_stats(completed, "custom_wait_s"),
        "custom_wait_stats_incomplete": _metric_stats(incomplete, "custom_wait_s"),
        "custom_wait_stats_all": _metric_stats(vehicle_records, "custom_wait_s"),
        "delay_stats_completed": _metric_stats(completed, "delay_s"),
        "delay_stats_incomplete": _metric_stats(incomplete, "delay_s"),
        "delay_stats_all": _metric_stats(vehicle_records, "delay_s"),
        "time_loss_stats_all": _metric_stats(vehicle_records, "time_loss_s"),
        "mean_travel_time_s": float(np.mean([r["travel_time_s"] for r in completed]))
        if completed
        else None,
        "avg_travel_time_metric": float(trainer_obj.metric.real_average_travel_time()),
        "throughput": int(trainer_obj.metric.throughput()),
        "output_dir": output_dir,
        "csv_path": csv_path,
        "vehicle_type_counts": {
            t: sum(1 for r in vehicle_records if r.get("vehicle_type") == t)
            for t in set(r.get("vehicle_type", "") for r in vehicle_records)
        },
        "custom_wait_stats_by_type": {
            t: _metric_stats(
                [r for r in vehicle_records if r.get("vehicle_type") == t],
                "custom_wait_s",
            )
            for t in set(r.get("vehicle_type", "") for r in vehicle_records)
            if t
        },
        "notes": (
            "Use custom_wait_s for all fairness / outlier plots (no 100s SUMO cap). "
            "accumulated_waiting_time_s is kept for reference only. custom_wait_s increments "
            f"each step when speed < {STOP_SPEED_THRESHOLD} m/s, snapshotted before env.step(). "
            "Include trip_status=on_map_at_end for stranded vehicles."
        ),
    }
    meta_path = os.path.join(output_dir, "vehicle_waiting_times_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    custom_all = meta.get("custom_wait_stats_all") or {}
    trainer_obj.logger.info(
        "Saved %d vehicle records to %s (custom_wait mean %.2fs, p95 %.2fs, max %.2fs)",
        len(vehicle_records),
        csv_path,
        custom_all.get("mean", 0.0),
        custom_all.get("p95", 0.0),
        custom_all.get("max", 0.0),
    )
    print(f"\nVehicle waiting-time log: {csv_path}")
    print(f"Metadata:               {meta_path}")
    if custom_all:
        print(
            f"custom_wait_s (all vehicles): mean={custom_all['mean']:.1f}s "
            f"p95={custom_all['p95']:.1f}s max={custom_all['max']:.1f}s"
        )
    return csv_path, meta_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a TSC agent on SUMO and log per-vehicle waiting times"
    )
    parser.add_argument(
        "--agent", "-a", type=str, default="maxpressure",
        help="Agent name (e.g. maxpressure, fixedtime, sotl)",
    )
    parser.add_argument("--network", "-n", type=str, default="sumo1x1")
    parser.add_argument(
        "--prefix", type=str, default="analysis",
        help="LibSignal run prefix (for data/output_data only; analysis CSV uses extras/output/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subfolder under extras/output/<agent>/<network>/ (default: seed<N>_steps<T>)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root (default: extras/output)",
    )
    parser.add_argument(
        "--interface", type=str, default="libsumo", choices=["libsumo", "traci"]
    )
    parser.add_argument(
        "--test-steps", type=int, default=None, help="Override trainer.test_steps"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    interface = _default_interface(args.interface)

    run_args = argparse.Namespace(
        thread_num=4,
        ngpu="-1",
        prefix=args.prefix,
        seed=args.seed,
        debug=False,
        interface=interface,
        delay_type="apx",
        task="tsc",
        agent=args.agent,
        world="sumo",
        network=args.network,
        dataset="onfly",
    )

    runner = Runner(run_args)
    test_steps = args.test_steps
    if test_steps is not None:
        Registry.mapping["trainer_mapping"]["setting"].param["test_steps"] = test_steps
        Registry.mapping["trainer_mapping"]["setting"].param["steps"] = test_steps
    else:
        test_steps = Registry.mapping["trainer_mapping"]["setting"].param["test_steps"]

    output_dir = resolve_output_dir(
        agent=args.agent,
        network=args.network,
        seed=args.seed,
        test_steps=test_steps,
        run_name=args.run_name,
        output_root=args.output_root,
    )

    logger = setup_logging(logging.INFO)
    runner.trainer = Registry.mapping["trainer_mapping"][
        Registry.mapping["command_mapping"]["setting"].param["task"]
    ](logger)

    return run_with_vehicle_logs(runner, output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
