#!/usr/bin/env python3
"""Run SUMO GUI briefly and capture a screenshot of ghost stacking (h=2)."""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent  # noqa
from common import interface
from common.registry import Registry
from utils.logger import build_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='maxpressure_stack2_gui')
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--screenshot-step', type=int, default=60)
    parser.add_argument('--output', default='/opt/cursor/artifacts/screenshots/ghost_stack_h2_gui.png')
    parser.add_argument('--delay-ms', type=int, default=80)
    args = parser.parse_args()

    os.environ.setdefault('DISPLAY', ':1')
    os.environ['SUMO_GUI_DELAY'] = str(args.delay_ms)

    import argparse as ap
    ns = ap.Namespace(
        thread_num=4, ngpu='-1', prefix='gui_cap', seed=42, debug=False,
        interface='libsumo', delay_type='apx', task='tsc', agent=args.agent,
        world='sumo', network='sumo1x1', dataset='onfly',
    )
    config, _ = build_config(ns)
    config['world']['gui'] = True

    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config)
    interface.Logger_path_Interface(config)
    interface.ModelAgent_param_Interface(config)
    os.makedirs(Registry.mapping['logger_mapping']['path'].path, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    sim_cfg = os.path.join(os.getcwd(), 'configs/sim', 'sumo1x1.cfg')
    world = Registry.mapping['world_mapping']['sumo'](sim_cfg, 0, interface='libsumo')
    mp = Registry.mapping['model_mapping']['maxpressure'](world, 0)

    print(f'ghost_stack_height={world.ghost_stack_height}, screenshot at step {args.screenshot_step}')
    world.reset()

    for step in range(1, args.steps + 1):
        phase = mp.get_phase()
        ob = mp.get_ob()
        action = np.array([mp.get_action(ob, phase, test=True)])
        world.step(action)
        if step == args.screenshot_step:
            time.sleep(0.5)
            subprocess.run(['scrot', '-o', args.output], check=True)
            print(f'screenshot saved: {args.output}')
            busiest = None
            for lane_id in world.eng.lane.getIDList():
                if lane_id.startswith(':'):
                    continue
                vehs = world.eng.lane.getLastStepVehicleIDs(lane_id)
                if busiest is None or len(vehs) > len(busiest[1]):
                    busiest = (lane_id, vehs)
            if busiest:
                lane, vehs = busiest
                positions = [round(world.eng.vehicle.getLanePosition(v), 1) for v in vehs[:12]]
                print(f'busiest lane {lane} ({len(vehs)} cars) first 12 positions: {positions}')

    time.sleep(1)
    if world.interface_flag:
        import libsumo
        libsumo.close()


if __name__ == '__main__':
    main()
