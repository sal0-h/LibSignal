from . import BaseAgent
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
import numpy as np
import gymnasium as gym


@Registry.register_model('maxpressure')
class MaxPressureAgent(BaseAgent):
    '''
    MaxPressureAgent — traffic-signal control by Max-Pressure (Varaiya 2013).

    Default ``mp_variant: varaiya`` follows the original policy as closely as this
    simulator interface allows:

      weight(l → m) = x(l) − x_down(m)
      pressure(phase) = Σ_{movements in phase} c · weight
      action = argmax_phase pressure

    where ``x`` is the lane waiting-vehicle count (queue proxy), ``x_down(m)`` is the
    total queue on the receiving link (0 for network exits), and ``c`` defaults to 1
    (equal saturation rates). See ``docs/MAXPRESSURE.md``.

    ``mp_variant: libsignal`` keeps the legacy LibSignal heuristic
    (lane_count[start] − lane_count[end] per lanelink) for reproducibility.
    '''
    def __init__(self, world, rank):
        super().__init__(world)
        self.world = world
        self.rank = rank
        self.model = None

        # get generator for each MaxPressure
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                     in_only=True, average='all', negative=True)
        
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)

        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True,
                                                     negative=False)
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        
        param = Registry.mapping['model_mapping']['setting'].param
        # the minimum duration of time of one phase (Varaiya §4 min-green variant)
        self.t_min = param['t_min']
        # 'varaiya' (default) | 'libsignal' (legacy lane_count heuristic)
        self.mp_variant = param.get('mp_variant', 'varaiya')
        # uniform saturation flow multiplier; set per-movement rates via config later if needed
        self.sat_flow = float(param.get('sat_flow', 1.0))
        self._exit_roads = self._compute_exit_roads()

    def _compute_exit_roads(self):
        '''
        Roads that are not an approach to any signalized intersection are network
        exits. Varaiya sets downstream queue weight to 0 on exit links.
        '''
        approach_roads = set()
        for inter in self.world.intersections:
            approach_roads.update(inter.in_roads)
        exit_roads = set()
        for inter in self.world.intersections:
            for road in inter.out_roads:
                if road not in approach_roads:
                    exit_roads.add(road)
        return exit_roads

    def _road_id_from_lane(self, lane_id):
        # Matches Intersection construction: lane IDs are treated as ``<road>_<digit>``.
        return lane_id[:-2]

    def _phase_pressure_varaiya(self, queues, phase_id):
        '''
        Varaiya-style phase pressure using waiting queues and exit-aware downstream.
        Without turn ratios, downstream is the total queue on receiving link m
        (Σ_p x(m,p) rather than Σ_p r(m,p) x(m,p)).
        '''
        pressure = 0.0
        for start, end in self.inter_obj.phase_available_lanelinks[phase_id]:
            upstream = queues.get(start, 0)
            end_road = self._road_id_from_lane(end)
            if end_road in self._exit_roads:
                downstream = 0.0
            else:
                lanes = self.inter_obj.road_lane_mapping.get(end_road, [end])
                downstream = sum(queues.get(lane, 0) for lane in lanes)
            pressure += self.sat_flow * (upstream - downstream)
        return pressure

    def _phase_pressure_libsignal(self, lane_counts, phase_id):
        '''Legacy LibSignal heuristic: Σ (lane_count[start] − lane_count[end]).'''
        return sum(
            lane_counts[start] - lane_counts[end]
            for start, end in self.inter_obj.phase_available_lanelinks[phase_id]
        )

    def reset(self):
        '''
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        '''
        # get generator for each MaxPressure
        inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[inter_id]
        self.ob_generator = self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj, ['lane_count'], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_count"],
                                                     in_only=True, average='all', negative=True)
        
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)

        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True,
                                                     negative=False)
        self._exit_roads = self._compute_exit_roads()

    def __repr__(self):
        return 'Maxpressure Agent has no Network model'

    def get_ob(self):
        '''
        get_ob
        Get observation from environment.

        :param: None
        :return x_obs: observation generated by ob_generator
        '''
        x_obs = []
        x_obs.append(self.ob_generator.generate())
        x_obs = np.array(x_obs, dtype=np.float32)
        return x_obs

    def get_reward(self):
        '''
        get_reward
        Get reward from environment.

        :param: None
        :return rewards: rewards generated by reward_generator
        '''
        rewards = []
        rewards.append(self.reward_generator.generate())
        rewards = np.squeeze(np.array(rewards)) * 12
        return rewards
    
    def get_phase(self):
        '''
        get_phase
        Get current phase of intersection(s) from environment.

        :param: None
        :return phase: current phase generated by phase_generator
        '''
        phase = []
        phase.append(self.phase_generator.generate())
        # phase = np.concatenate(phase, dtype=np.int8)
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase
    
    def get_action(self, ob, phase, test=True):
        '''
        get_action
        Generate action.

        :param ob: observation, the shape is (1,12)
        :param phase: current phase, the shape is (1,)
        :param test: boolean, decide whether is test process
        :return action: action that has the highest score
        '''
        if self.inter_obj.current_phase_time < self.t_min:
            return self.inter_obj.current_phase

        if self.mp_variant == 'libsignal':
            counts = self.world.get_info("lane_count")
            score_fn = lambda pid: self._phase_pressure_libsignal(counts, pid)
        else:
            queues = self.world.get_info("lane_waiting_count")
            score_fn = lambda pid: self._phase_pressure_varaiya(queues, pid)

        max_pressure = None
        action = -1
        for phase_id in range(len(self.inter_obj.phases)):
            pressure = score_fn(phase_id)
            if max_pressure is None or pressure > max_pressure:
                action = phase_id
                max_pressure = pressure

        return action

    def get_queue(self):
        '''
        get_queue
        Get queue length of intersection.

        :param: None
        :return: total queue length
        '''
        queue = []
        queue.append(self.queue.generate())
        queue = np.sum(np.squeeze(np.array(queue)))
        return queue

    def get_delay(self):
        '''
        get_delay
        Get delay of intersection.

        :param: None
        :return: total delay
        '''
        delay = []
        delay.append(self.delay.generate())
        delay = np.sum(np.squeeze(np.array(delay)))
        return delay
