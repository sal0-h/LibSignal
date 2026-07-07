#!/usr/bin/env python3
"""
Verify ghost_stack_height invariants over a short SUMO run.

Checks on every ghost stack apply (post-step):
  1) No stack exceeds h vehicles.
  2) No stack with size < h has another stack behind it (partial stacks only at rear).

Usage:
  source .venv/bin/activate
  python extras/verify_ghost_stacks.py [--network sumo1x1] [--height 2] [--steps 300]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent  # noqa: F401
from common import interface
from common.registry import Registry
from utils.logger import build_config
from world.world_sumo import GHOST_STACK_GAP


def _normalize_height(h):
    if h is None or h <= 0:
        return None
    return int(h)


def check_lane(eng, lane, veh_ids, stack_height):
    """Return list of violation strings (empty if OK)."""
    violations = []
    if not veh_ids:
        return violations

    veh_ids = sorted(veh_ids, key=lambda v: (-eng.vehicle.getLanePosition(v), v))
    if stack_height is None:
        stacks = [veh_ids]
    else:
        stacks = [veh_ids[i:i + stack_height] for i in range(0, len(veh_ids), stack_height)]

    flat = [v for s in stacks for v in s]
    if flat != veh_ids:
        violations.append(f"{lane}: partition order mismatch")

    for i, stack in enumerate(stacks):
        n = len(stack)
        if stack_height is not None and (n < 1 or n > stack_height):
            violations.append(f"{lane}: stack[{i}] size {n} outside 1..{stack_height}")
        if stack_height is not None and i < len(stacks) - 1 and n < stack_height:
            violations.append(
                f"{lane}: stack[{i}] size {n} < {stack_height} with stack behind"
            )

    rep_positions = []
    rep_pos = None
    prev_lead = None
    for stack in stacks:
        lead = stack[0]
        own_pos = eng.vehicle.getLanePosition(lead)
        if rep_pos is None:
            rep_pos = own_pos
        else:
            spacing = eng.vehicle.getLength(prev_lead) + GHOST_STACK_GAP
            blocked_pos = max(0.0, rep_pos - spacing)
            rep_pos = min(own_pos, blocked_pos) if own_pos > blocked_pos else own_pos
        rep_positions.append(rep_pos)
        for v in stack:
            actual = eng.vehicle.getLanePosition(v)
            if abs(actual - rep_pos) > 0.1:
                violations.append(
                    f"{lane}: {v} at {actual:.2f} not at stack rep {rep_pos:.2f}"
                )
        prev_lead = lead

    for i in range(len(rep_positions) - 1):
        if rep_positions[i] < rep_positions[i + 1] - 0.1:
            violations.append(
                f"{lane}: stack[{i}]@{rep_positions[i]:.2f} behind stack[{i+1}]"
            )

    return violations


def inspect_world(world, stack_height, label):
    violations = []
    lanes_with_traffic = 0
    stack_size_hist = {}
    for lane_id in world.eng.lane.getIDList():
        if lane_id.startswith(':'):
            continue
        veh_ids = [v for v in world.eng.lane.getLastStepVehicleIDs(lane_id) if v]
        if not veh_ids:
            continue
        lanes_with_traffic += 1
        veh_ids = sorted(veh_ids, key=lambda v: (-world.eng.vehicle.getLanePosition(v), v))
        if stack_height is None:
            stacks = [veh_ids]
        else:
            stacks = [veh_ids[i:i + stack_height] for i in range(0, len(veh_ids), stack_height)]
        for s in stacks:
            stack_size_hist[len(s)] = stack_size_hist.get(len(s), 0) + 1
        violations.extend(check_lane(world.eng, lane_id, veh_ids, stack_height))

    print(f"[verify] {label}: lanes_with_traffic={lanes_with_traffic} stack_sizes={dict(sorted(stack_size_hist.items()))}")
    return violations


def main():
    parser = argparse.ArgumentParser(description="Verify ghost stack height invariants")
    parser.add_argument('--network', default='sumo1x1')
    parser.add_argument('--height', type=int, default=2)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--agent', default='maxpressure_stack2')
    args = parser.parse_args()

    ns = argparse.Namespace(
        thread_num=4,
        ngpu='-1',
        prefix='verify',
        seed=42,
        debug=False,
        interface='libsumo',
        delay_type='apx',
        task='tsc',
        agent=args.agent,
        world='sumo',
        network=args.network,
        dataset='onfly',
    )

    config, _ = build_config(ns)
    config['world']['ghost_debug'] = True
    if args.height != 2 or args.agent == 'maxpressure':
        config['world']['physics_mode'] = 'ghost'
        config['world']['ghost_stack_height'] = args.height

    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config)
    interface.Logger_path_Interface(config)
    log_root = Registry.mapping['logger_mapping']['path'].path
    os.makedirs(log_root, exist_ok=True)
    interface.ModelAgent_param_Interface(config)

    sim_cfg = os.path.join(os.getcwd(), 'configs/sim', args.network + '.cfg')
    world = Registry.mapping['world_mapping']['sumo'](
        sim_cfg, 0, interface='libsumo',
    )
    mp_agents = []
    model_name = Registry.mapping['model_mapping']['setting'].param['name']
    agent_cls = Registry.mapping['model_mapping'][model_name]
    mp_agents.append(agent_cls(world, 0))
    num_agent = int(len(world.intersections) / mp_agents[0].sub_agents)
    for i in range(1, num_agent):
        mp_agents.append(agent_cls(world, i))

    stack_height = _normalize_height(world.ghost_stack_height)
    total_violations = []
    world.reset()
    total_violations.extend(inspect_world(world, stack_height, 'after reset'))

    for step in range(args.steps):
        actions = []
        for ag in mp_agents:
            phase = ag.get_phase()
            ob = ag.get_ob()
            actions.append(ag.get_action(ob, phase, test=True))
        world.step(np.array(actions))
        if step % 50 == 49 or step == args.steps - 1:
            total_violations.extend(
                inspect_world(world, stack_height, f'step {step + 1}')
            )

    print()
    if total_violations:
        print(f"FAILED: {len(total_violations)} invariant violation(s):")
        for v in total_violations[:20]:
            print(f"  - {v}")
        if len(total_violations) > 20:
            print(f"  ... and {len(total_violations) - 20} more")
        sys.exit(1)

    print(f"PASSED: no invariant violations over {args.steps} steps "
          f"(network={args.network}, h={stack_height})")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
