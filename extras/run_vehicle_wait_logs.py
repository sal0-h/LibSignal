#!/usr/bin/env python3
"""
Run a LibSignal TSC agent on SUMO and export per-vehicle trip metrics (CSV + JSON).

Outputs: extras/output/<agent>/<network>/<run_name>/
  vehicle_trip_metrics.csv   — one row per vehicle (completed + censored)
  vehicle_trip_metrics_meta.json — summary stats + LibSignal ATT

Usage (from repo root):
  python extras/run_vehicle_wait_logs.py --agent maxpressure --network sumo4x4 --seed 42
  python extras/run_vehicle_wait_logs.py --agent dqn --network sumo4x4 --seed 42 --train
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


def _vehicle_route(world, veh_id):
    try:
        return list(world.eng.vehicle.getRoute(veh_id))
    except Exception:
        return None


def _route_distance_m(world, veh_id):
    """Planned route length (sum of edge lengths)."""
    route = _vehicle_route(world, veh_id)
    if not route:
        return None
    try:
        return float(sum(world.eng.edge.getLength(edge_id) for edge_id in route))
    except Exception:
        return None


def _free_flow_route_time(world, veh_id):
    """Minimum travel time if the vehicle drove at each edge's speed limit."""
    route = _vehicle_route(world, veh_id)
    if not route:
        return None
    try:
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


def _distance_traveled_m(world, veh_id):
    """Odometer distance along the route (SUMO getDistance)."""
    try:
        return float(world.eng.vehicle.getDistance(veh_id))
    except Exception:
        return None


def _snapshot_vehicle_metrics(world):
    """Capture per-vehicle SUMO metrics before env.step() removes departures."""
    pre_sumo_wait = {}
    pre_time_loss = {}
    pre_speed = {}
    pre_distance = {}
    for veh_id in world.eng.vehicle.getIDList():
        pre_sumo_wait[veh_id] = float(world.eng.vehicle.getAccumulatedWaitingTime(veh_id))
        try:
            pre_time_loss[veh_id] = float(world.eng.vehicle.getTimeLoss(veh_id))
        except Exception:
            pre_time_loss[veh_id] = 0.0
        pre_speed[veh_id] = float(world.eng.vehicle.getSpeed(veh_id))
        dist = _distance_traveled_m(world, veh_id)
        if dist is not None:
            pre_distance[veh_id] = dist
    return pre_sumo_wait, pre_time_loss, pre_speed, pre_distance


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
    route_distance_m=None,
    distance_traveled_m=None,
):
    delay = None
    schedule_efficiency = None
    moving_time = max(0.0, travel_time - custom_wait) if travel_time > 0 else 0.0
    waiting_fraction = custom_wait / travel_time if travel_time > 0 else 0.0
    moving_fraction = 1.0 - waiting_fraction if travel_time > 0 else 0.0
    if free_flow_time is not None and travel_time > 0:
        delay = max(0.0, travel_time - free_flow_time)
        schedule_efficiency = min(1.0, max(0.0, free_flow_time / travel_time))
    return {
        "vehicle_id": veh_id,
        "vehicle_type": vehicle_type,
        "enter_time_s": round(enter_time, 3),
        "departure_time_s": round(departure_time, 3) if departure_time is not None else None,
        "travel_time_s": round(travel_time, 3),
        "route_distance_m": round(route_distance_m, 3) if route_distance_m is not None else "",
        "distance_traveled_m": round(distance_traveled_m, 3) if distance_traveled_m is not None else "",
        "accumulated_waiting_time_s": round(sumo_wait, 3),
        "custom_wait_s": round(custom_wait, 3),
        "moving_time_s": round(moving_time, 3),
        "time_loss_s": round(time_loss, 3),
        "free_flow_time_s": round(free_flow_time, 3) if free_flow_time is not None else "",
        "delay_s": round(delay, 3) if delay is not None else "",
        "waiting_fraction": round(waiting_fraction, 4),
        "moving_fraction": round(moving_fraction, 4),
        "schedule_efficiency": round(schedule_efficiency, 4) if schedule_efficiency is not None else "",
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
    pre_distance,
    custom_wait,
    cumulative_sumo_wait,
    cumulative_time_loss,
    free_flow_times,
    route_distances,
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
                route_distance_m=route_distances.pop(veh_id, None),
                distance_traveled_m=pre_distance.get(veh_id),
            )
        )
    return records, seen_logged | newly_completed


def _collect_removed_records(
    world,
    pre_sumo_wait,
    pre_time_loss,
    pre_distance,
    custom_wait,
    cumulative_sumo_wait,
    cumulative_time_loss,
    free_flow_times,
    route_distances,
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
                route_distance_m=route_distances.pop(veh_id, None),
                distance_traveled_m=pre_distance.get(veh_id),
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


def _refresh_route_metrics(world, free_flow_times, route_distances):
    for veh_id in world.eng.vehicle.getIDList():
        fft = _free_flow_route_time(world, veh_id)
        if fft is not None:
            free_flow_times[veh_id] = fft
        rd = _route_distance_m(world, veh_id)
        if rd is not None:
            route_distances[veh_id] = rd


def _collect_remaining_records(
    world,
    custom_wait,
    cumulative_sumo_wait,
    cumulative_time_loss,
    free_flow_times,
    route_distances,
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
                route_distance_m=route_distances.get(veh_id),
                distance_traveled_m=_distance_traveled_m(world, veh_id),
            )
        )
    return records


def _system_idle_share(records):
    total_time = sum(float(r["travel_time_s"]) for r in records if r.get("travel_time_s"))
    total_idle = sum(float(r["custom_wait_s"]) for r in records if r.get("custom_wait_s") is not None)
    if total_time <= 0:
        return None
    return float(total_idle / total_time)


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
    route_distances = {}
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
                _refresh_route_metrics(world, free_flow_times, route_distances)
                pre_sumo_wait, pre_time_loss, pre_speed, pre_distance = _snapshot_vehicle_metrics(world)
                _update_custom_wait(custom_wait, pre_speed, dt)

                obs, rewards, dones, _ = env.step(actions.flatten())

                sim_time = world.eng.simulation.getTime()

                step_records, seen_logged = _collect_departed_records(
                    world,
                    pre_sumo_wait,
                    pre_time_loss,
                    pre_distance,
                    custom_wait,
                    cumulative_sumo_wait,
                    cumulative_time_loss,
                    free_flow_times,
                    route_distances,
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
                    pre_distance,
                    custom_wait,
                    cumulative_sumo_wait,
                    cumulative_time_loss,
                    free_flow_times,
                    route_distances,
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
            route_distances,
            seen_logged,
            sim_time,
        )
    )

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "vehicle_trip_metrics.csv")
    fieldnames = [
        "vehicle_id",
        "vehicle_type",
        "enter_time_s",
        "departure_time_s",
        "travel_time_s",
        "route_distance_m",
        "distance_traveled_m",
        "accumulated_waiting_time_s",
        "custom_wait_s",
        "moving_time_s",
        "time_loss_s",
        "free_flow_time_s",
        "delay_s",
        "waiting_fraction",
        "moving_fraction",
        "schedule_efficiency",
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

    n_departed = len(vehicle_records)
    completion_rate = len(completed) / n_departed if n_departed else None

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
        "completion_rate": completion_rate,
        "trip_status_counts": status_counts,
        "primary_idle_metric": "custom_wait_s",
        "primary_efficiency_metric": "schedule_efficiency",
        "metric_definitions": {
            "travel_time_s": "Wall-clock time in network (LibSignal ATT uses mean over completed).",
            "route_distance_m": "Sum of edge lengths on assigned route.",
            "distance_traveled_m": "SUMO odometer at trip end (or at horizon for censored).",
            "custom_wait_s": f"Uncapped idle time (speed < {STOP_SPEED_THRESHOLD} m/s each step).",
            "waiting_fraction": "custom_wait_s / travel_time_s.",
            "moving_fraction": "1 - waiting_fraction.",
            "schedule_efficiency": "free_flow_time_s / travel_time_s in [0,1] (higher is better).",
            "system_idle_share": "sum(custom_wait_s) / sum(travel_time_s) over a record set.",
        },
        "sumo_accumulated_wait_note": (
            "accumulated_waiting_time_s uses SUMO getAccumulatedWaitingTime(), which is "
            "capped by default --waiting-time-memory=100s. Do NOT use it for tail/fairness "
            "plots. custom_wait_s is uncapped (per-step speed < 0.1 m/s bookkeeping)."
        ),
        "travel_time_stats_completed": _metric_stats(completed, "travel_time_s"),
        "travel_time_stats_incomplete": _metric_stats(incomplete, "travel_time_s"),
        "route_distance_stats_completed": _metric_stats(completed, "route_distance_m"),
        "distance_traveled_stats_completed": _metric_stats(completed, "distance_traveled_m"),
        "schedule_efficiency_stats_completed": _metric_stats(completed, "schedule_efficiency"),
        "moving_fraction_stats_completed": _metric_stats(completed, "moving_fraction"),
        "waiting_fraction_stats_completed": _metric_stats(completed, "waiting_fraction"),
        "custom_wait_stats_completed": _metric_stats(completed, "custom_wait_s"),
        "custom_wait_stats_incomplete": _metric_stats(incomplete, "custom_wait_s"),
        "custom_wait_stats_all": _metric_stats(vehicle_records, "custom_wait_s"),
        "delay_stats_completed": _metric_stats(completed, "delay_s"),
        "time_loss_stats_all": _metric_stats(vehicle_records, "time_loss_s"),
        "system_idle_share_completed": _system_idle_share(completed),
        "system_idle_share_all": _system_idle_share(vehicle_records),
        "mean_schedule_efficiency_completed": (
            float(np.mean([float(r["schedule_efficiency"]) for r in completed if r.get("schedule_efficiency") not in ("", None)]))
            if completed
            else None
        ),
        "mean_travel_time_s": float(np.mean([r["travel_time_s"] for r in completed]))
        if completed
        else None,
        "median_travel_time_s": float(np.median([r["travel_time_s"] for r in completed]))
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
        "notes": (
            "Plot histograms/ECDFs of travel_time_s and route_distance_m (completed first). "
            "Use schedule_efficiency and waiting_fraction for per-trip congestion shape. "
            "Report completion_rate alongside ATT. Censored trips: trip_status=on_map_at_end."
        ),
    }
    meta_path = os.path.join(output_dir, "vehicle_trip_metrics_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    trainer_obj.logger.info(
        "Saved %d vehicle records to %s (ATT=%.2fs, completion=%.1f%%)",
        len(vehicle_records),
        csv_path,
        meta.get("avg_travel_time_metric") or 0.0,
        (completion_rate or 0.0) * 100.0,
    )
    print(f"\nVehicle trip metrics CSV: {csv_path}")
    print(f"Metadata:                 {meta_path}")
    if meta.get("travel_time_stats_completed"):
        ts = meta["travel_time_stats_completed"]
        print(
            f"travel_time_s (completed): mean={ts['mean']:.1f}s median={ts['median']:.1f}s "
            f"p95={ts['p95']:.1f}s"
        )
    if meta.get("mean_schedule_efficiency_completed") is not None:
        print(f"mean schedule_efficiency (completed): {meta['mean_schedule_efficiency_completed']:.4f}")
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
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train RL agent first (e.g. dqn), then export trip metrics on greedy test rollout",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override trainer.episodes when --train is set",
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
    if args.episodes is not None:
        Registry.mapping["trainer_mapping"]["setting"].param["episodes"] = args.episodes

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

    if args.train:
        train_model = Registry.mapping["model_mapping"]["setting"].param.get("train_model", False)
        if not train_model:
            print(f"[Warning] --train set but agent '{args.agent}' has train_model=False; exporting only.")
        else:
            print(f"Training {args.agent} for {Registry.mapping['trainer_mapping']['setting'].param['episodes']} episodes...")
            runner.trainer.train()
            for ag in runner.trainer.agents:
                ag.load_model(runner.trainer.episodes)
            print("Training done; running trip-metrics export rollout...")

    return run_with_vehicle_logs(runner, output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
