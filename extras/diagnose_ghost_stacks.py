#!/usr/bin/env python3
"""Diagnose ghost stack positions on sumo1x1 at runtime."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent  # noqa
from common import interface
from common.registry import Registry
from utils.logger import build_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='maxpressure_stack2')
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--sample-every', type=int, default=10)
    args = parser.parse_args()

    import argparse as ap
    ns = ap.Namespace(
        thread_num=4, ngpu='-1', prefix='diag', seed=42, debug=False,
        interface='libsumo', delay_type='apx', task='tsc', agent=args.agent,
        world='sumo', network='sumo1x1', dataset='onfly',
    )
    config, _ = build_config(ns)
    if args.height is not None:
        config['world']['physics_mode'] = 'ghost'
        config['world']['ghost_stack_height'] = args.height

    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config)
    interface.Logger_path_Interface(config)
    interface.ModelAgent_param_Interface(config)
    os.makedirs(Registry.mapping['logger_mapping']['path'].path, exist_ok=True)

    sim_cfg = os.path.join(os.getcwd(), 'configs/sim', 'sumo1x1.cfg')
    world = Registry.mapping['world_mapping']['sumo'](sim_cfg, 0, interface='libsumo')
    mp = Registry.mapping['model_mapping']['maxpressure'](world, 0)

    print('physics_mode:', world.physics_mode)
    print('ghost_stack_height:', world.ghost_stack_height)
    print('normalized h:', world._normalize_ghost_stack_height(world.ghost_stack_height))

    world.reset()
    obs = [mp.get_ob()]
    max_lane_cars = 0
    worst_examples = []

    for step in range(args.steps):
        phase = mp.get_phase()
        ob = mp.get_ob()
        action = np.array([mp.get_action(ob, phase, test=True)])
        world.step(action)

        if step % args.sample_every != 0:
            continue

        t = world.eng.simulation.getTime()
        for lane_id in world.eng.lane.getIDList():
            if lane_id.startswith(':'):
                continue
            vehs = world.eng.lane.getLastStepVehicleIDs(lane_id)
            if len(vehs) < 3:
                continue
            positions = sorted(
                [(v, world.eng.vehicle.getLanePosition(v)) for v in vehs],
                key=lambda x: -x[1],
            )
            pos_vals = [p for _, p in positions]
            unique_pos = len({round(p, 1) for p in pos_vals})
            max_lane_cars = max(max_lane_cars, len(vehs))

            if len(vehs) >= 3:
                worst_examples.append({
                    't': t, 'lane': lane_id, 'n': len(vehs),
                    'unique_pos': unique_pos,
                    'positions': pos_vals[:8],
                    'ids': [v for v, _ in positions[:8]],
                })

    print(f'\nmax cars on one edge lane (sampled): {max_lane_cars}')
    print(f'lanes with 3+ cars (samples): {len(worst_examples)}')

    # Show worst cases: most cars, fewest unique positions
    worst_examples.sort(key=lambda x: (-x['n'], x['unique_pos']))
    for ex in worst_examples[:10]:
        print(
            f"  t={ex['t']:.0f} lane={ex['lane']} n={ex['n']} "
            f"unique_pos@0.1m={ex['unique_pos']} positions={['%.1f' % p for p in ex['positions']]}"
        )

    # Manual stack replay on busiest sample
    if worst_examples:
        ex = worst_examples[0]
        lane = ex['lane']
        vehs = world.eng.lane.getLastStepVehicleIDs(lane)
        h = world._normalize_ghost_stack_height(world.ghost_stack_height)
        sorted_v = world._sort_lane_vehicles_front_first(vehs)
        stacks = world._partition_ghost_stacks(sorted_v, h)
        print(f'\nReplay partition on lane {lane} (h={h}):')
        for i, st in enumerate(stacks):
            pos = [world.eng.vehicle.getLanePosition(v) for v in st]
            print(f'  stack[{i}] size={len(st)} positions={["%.2f" % p for p in pos]} ids={st}')


if __name__ == '__main__':
    main()
