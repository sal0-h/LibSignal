#!/usr/bin/env python3
"""
Profile LibSignal training hot path: SUMO physics vs observe vs trajectory
vs infos vs agent/RL.

Usage (from repo root, venv activated):
  python scripts/profiling/profile_training_hotpath.py --agent maxpressure --episodes 1
  python scripts/profiling/profile_training_hotpath.py --agent dqn --episodes 1 --skip-test
  python scripts/profiling/profile_training_hotpath.py --agent maxpressure --episodes 1 --skip-trajectory
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class Timer:
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def track(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def report(self) -> dict:
        total = sum(self.totals.values())
        rows = []
        for k, v in sorted(self.totals.items(), key=lambda x: -x[1]):
            rows.append(
                {
                    "name": k,
                    "seconds": round(v, 4),
                    "calls": self.counts[k],
                    "pct_of_summed": round(100.0 * v / total, 2) if total else 0.0,
                    "ms_per_call": round(1000.0 * v / self.counts[k], 4) if self.counts[k] else 0.0,
                }
            )
        return {"summed_seconds": round(total, 4), "breakdown": rows}


def install_world_hooks(world, timer: Timer, skip_trajectory: bool):
    import world.world_sumo as world_sumo

    # Patch Intersection.observe
    orig_observe = world_sumo.Intersection.observe

    def timed_observe(self, step_length, distance):
        with timer.track("observe"):
            return orig_observe(self, step_length, distance)

    world_sumo.Intersection.observe = timed_observe

    # Patch World methods used in step
    orig_step_sim = world.step_sim.__func__ if hasattr(world.step_sim, "__func__") else None
    # Bind to instance methods carefully
    orig_step_sim_bound = world.step_sim
    orig_update_infos = world._update_infos
    orig_get_traj = world.get_vehicle_trajectory
    orig_step = world.step
    orig_reset = world.reset

    def timed_step_sim():
        with timer.track("simulationStep"):
            return orig_step_sim_bound()

    def timed_update_infos():
        with timer.track("_update_infos"):
            return orig_update_infos()

    def timed_get_traj():
        with timer.track("get_vehicle_trajectory"):
            if skip_trajectory:
                return world.vehicle_trajectory, world.vehicle_maxspeed
            return orig_get_traj()

    def timed_reset():
        with timer.track("reset"):
            return orig_reset()

    def timed_step(action=None):
        # Fine-grained: replicate step with timers around major blocks
        if action is not None:
            with timer.track("pseudo_step"):
                for i, intersection in enumerate(world.intersections):
                    intersection.pseudo_step(action[i])
            timed_step_sim()
        # observe already timed via class patch
        for intsec in world.intersections:
            intsec.observe(world.step_length, world.max_distance)
        with timer.track("depart_arrive_bookkeeping"):
            entering_v = world.eng.simulation.getDepartedIDList()
            for v in entering_v:
                world.inside_vehicles.update({v: world.get_current_time()})
                if world.physics_mode == "ghost":
                    world._enforce_ghost_vehicle(v)
            if world.physics_mode == "ghost" and entering_v:
                world._collapse_ghost_lanes()
            exiting_v = world.eng.simulation.getArrivedIDList()
            for v in exiting_v:
                if v not in world.inside_vehicles:
                    continue
                world.vehicles.update(
                    {v: world.get_current_time() - world.inside_vehicles[v]}
                )
                del world.inside_vehicles[v]
        timed_update_infos()
        if skip_trajectory or (
            hasattr(world, "_trajectory_tracking_enabled")
            and not world._trajectory_tracking_enabled()
        ):
            with timer.track("get_vehicle_trajectory"):
                pass  # gated off (optimization path)
        else:
            world.vehicle_trajectory, world.vehicle_maxspeed = timed_get_traj()
        world.run += 1

    world.step_sim = timed_step_sim
    world._update_infos = timed_update_infos
    world.get_vehicle_trajectory = timed_get_traj
    world.step = timed_step
    world.reset = timed_reset
    return world_sumo.Intersection.observe, orig_observe


def install_agent_hooks(agents, timer: Timer):
    for ag in agents:
        if hasattr(ag, "get_ob"):
            orig = ag.get_ob

            def make_ob(o):
                def wrapped(*a, **k):
                    with timer.track("agent.get_ob"):
                        return o(*a, **k)

                return wrapped

            ag.get_ob = make_ob(orig)
        if hasattr(ag, "get_reward"):
            orig = ag.get_reward

            def make_rw(o):
                def wrapped(*a, **k):
                    with timer.track("agent.get_reward"):
                        return o(*a, **k)

                return wrapped

            ag.get_reward = make_rw(orig)
        if hasattr(ag, "get_action"):
            orig = ag.get_action

            def make_act(o):
                def wrapped(*a, **k):
                    with timer.track("agent.get_action"):
                        return o(*a, **k)

                return wrapped

            ag.get_action = make_act(orig)
        if hasattr(ag, "train"):
            orig = ag.train

            def make_tr(o):
                def wrapped(*a, **k):
                    with timer.track("agent.train"):
                        return o(*a, **k)

                return wrapped

            ag.train = make_tr(orig)
        if hasattr(ag, "remember"):
            orig = ag.remember

            def make_rm(o):
                def wrapped(*a, **k):
                    with timer.track("agent.remember"):
                        return o(*a, **k)

                return wrapped

            ag.remember = make_rm(orig)


def install_info_fn_hooks(world, timer: Timer):
    """Time each subscribed info function individually."""
    wrapped = {}
    for name, fn in list(world.info_functions.items()):
        if fn is None:
            continue

        def make(n, f):
            def wrapped_fn(*a, **k):
                with timer.track(f"info.{n}"):
                    return f(*a, **k)

            return wrapped_fn

        wrapped[name] = make(name, fn)
    world.info_functions.update(wrapped)


def build_runner(agent: str, network: str, seed: int, episodes: int, steps: int, skip_test: bool):
    # Mimic run.py argv so Registry/build_config works
    argv = [
        "profile_training_hotpath.py",
        "--agent",
        agent,
        "--world",
        "sumo",
        "--network",
        network,
        "--seed",
        str(seed),
        "--ngpu",
        "-1",
        "--prefix",
        "eff_audit",
    ]
    sys.argv = argv

    from common.utils import build_config, get_logger, get_logger_path
    from common.registry import Registry
    from run import Runner, parser as run_parser

    args = run_parser.parse_args()
    # Override trainer knobs via Registry after build
    cfg = build_config(args)
    # Force short runs
    cfg["trainer"]["episodes"] = episodes
    cfg["trainer"]["steps"] = steps
    cfg["trainer"]["test_steps"] = steps
    if skip_test:
        cfg["trainer"]["test_when_train"] = False
    # Also reduce learning_start for DQN profiling if needed
    if agent == "dqn":
        # Keep learning_start but we'll still see train() after buffer fills if long enough
        pass

    from common.interface import Interface
    from common.utils import get_output_file_path

    interface = Interface(cfg)
    interface.build()

    # Re-register settings after build_config mutations — Runner does this
    logging_level = int(os.environ.get("LIBSIGNAL_LOG", "20"))
    from utils.logger import setup_logging

    logger = setup_logging(logging_level)

    # Import side-effect registrations
    import agent  # noqa: F401
    import trainer  # noqa: F401
    import task  # noqa: F401
    import world  # noqa: F401

    # Rebuild config registry like Runner.__init__
    from run import Runner as R

    # Use Runner but patch task
    runner = R(args)
    # Override episodes/steps after Runner built config
    Registry.mapping["trainer_mapping"]["setting"].param["episodes"] = episodes
    Registry.mapping["trainer_mapping"]["setting"].param["steps"] = steps
    Registry.mapping["trainer_mapping"]["setting"].param["test_steps"] = steps
    if skip_test:
        Registry.mapping["trainer_mapping"]["setting"].param["test_when_train"] = False

    # Recreate trainer with updated params
    runner.trainer = Registry.mapping["trainer_mapping"][
        Registry.mapping["command_mapping"]["setting"].param["task"]
    ](logger)
    runner.task = Registry.mapping["task_mapping"][
        Registry.mapping["command_mapping"]["setting"].param["task"]
    ](runner.trainer)
    return runner, logger


def run_profiled(agent, network, seed, episodes, steps, skip_test, skip_trajectory, outfile):
    timer = Timer()
    wall0 = time.perf_counter()

    # Build via simplified path using run.py Runner
    sys.argv = [
        "run.py",
        "--agent",
        agent,
        "--world",
        "sumo",
        "--network",
        network,
        "--seed",
        str(seed),
        "--ngpu",
        "-1",
        "--prefix",
        "eff_audit",
    ]
    from run import Runner, parser

    args = parser.parse_args()
    runner = Runner(args)

    # Override after construction
    from common.registry import Registry

    Registry.mapping["trainer_mapping"]["setting"].param["episodes"] = episodes
    Registry.mapping["trainer_mapping"]["setting"].param["steps"] = steps
    Registry.mapping["trainer_mapping"]["setting"].param["test_steps"] = steps
    if skip_test:
        Registry.mapping["trainer_mapping"]["setting"].param["test_when_train"] = False

    # Rebuild trainer to pick up episode overrides
    from utils.logger import setup_logging

    logger = setup_logging(20)
    runner.trainer = Registry.mapping["trainer_mapping"][
        Registry.mapping["command_mapping"]["setting"].param["task"]
    ](logger)
    # Force attributes that were already copied in __init__
    runner.trainer.episodes = episodes
    runner.trainer.steps = steps
    runner.trainer.test_steps = steps
    runner.trainer.test_when_train = (not skip_test) and Registry.mapping["trainer_mapping"][
        "setting"
    ].param.get("test_when_train", True)

    world = runner.trainer.world
    agents = runner.trainer.agents

    install_world_hooks(world, timer, skip_trajectory=skip_trajectory)
    install_info_fn_hooks(world, timer)
    install_agent_hooks(agents, timer)

    # Time env.step wrapper
    env = runner.trainer.env
    orig_env_step = env.step

    def timed_env_step(actions, **kwargs):
        with timer.track("env.step_total"):
            return orig_env_step(actions, **kwargs)

    env.step = timed_env_step

    train_model = Registry.mapping["model_mapping"]["setting"].param["train_model"]
    with timer.track("task_run"):
        if train_model:
            runner.trainer.train()
        else:
            runner.trainer.test()

    wall = time.perf_counter() - wall0
    report = timer.report()
    report["wall_seconds"] = round(wall, 4)
    report["config"] = {
        "agent": agent,
        "network": network,
        "episodes": episodes,
        "steps": steps,
        "skip_test": skip_test,
        "skip_trajectory": skip_trajectory,
        "train_model": train_model,
        "n_intersections": len(world.intersections),
        "n_lanes": len(getattr(world, "all_lanes", [])),
        "subscribed_fns": list(getattr(world, "fns", [])),
    }
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="maxpressure")
    p.add_argument("--network", default="sumo1x1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--steps", type=int, default=3600)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--skip-trajectory", action="store_true")
    p.add_argument(
        "--outfile",
        default="extras/efficiency_audit/profile_latest.json",
    )
    args = p.parse_args()
    run_profiled(
        agent=args.agent,
        network=args.network,
        seed=args.seed,
        episodes=args.episodes,
        steps=args.steps,
        skip_test=args.skip_test,
        skip_trajectory=args.skip_trajectory,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()
