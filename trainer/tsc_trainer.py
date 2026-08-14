import os
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer


@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # Demand-set / held-out eval (optional; default off for existing configs)
        world_param = Registry.mapping['world_mapping']['setting'].param
        trainer_param = Registry.mapping['trainer_mapping']['setting'].param
        self.demand_set = (
            world_param.get('demand_set')
            or world_param.get('demand_bag')
            or []
        )
        self.demand_heldout = world_param.get('demand_heldout') or []
        self.demand_train_file = world_param.get('demand_train_file')
        self.heldout_eval_every = int(trainer_param.get('heldout_eval_every') or 0)
        # Standard early-stop budget (min/max episodes + ATT plateau / paired cycle).
        # episodes is the hard max; missing keys keep legacy always-run-full-budget behavior.
        self.min_episodes = int(trainer_param.get('min_episodes') or 0)
        self.early_stop = bool(trainer_param.get('early_stop', False))
        self.early_stop_patience = int(trainer_param.get('early_stop_patience') or 0)
        self.early_stop_min_delta = float(trainer_param.get('early_stop_min_delta') or 0.0)
        self.early_stop_metric = str(trainer_param.get('early_stop_metric') or 'test').lower()
        self.early_stop_z = float(trainer_param.get('early_stop_z') or 1.0)
        self.early_stop_abs_floor = float(trainer_param.get('early_stop_abs_floor') or 0.0)
        self.early_stop_window = int(trainer_param.get('early_stop_window') or 0)
        mode = str(trainer_param.get('early_stop_mode') or 'auto').lower()
        cycle_len = len(self.demand_set) if len(self.demand_set) > 1 else (self.early_stop_window or 10)
        if mode in ('', 'auto'):
            if self.early_stop_metric == 'wait' and len(self.demand_set) > 1:
                mode = 'cycle_mean'
            elif self.early_stop_metric == 'train' and len(self.demand_set) > 1:
                mode = 'paired_cycle'
            else:
                mode = 'plateau'
        self.early_stop_mode = mode
        self.early_stop_cycle_len = cycle_len
        self._paired_stopper = None
        self._cycle_mean_stopper = None
        self.best_att = float('inf')
        self.best_episode = None
        self.best_metric_source = None
        self.episodes_without_improve = 0
        self.stop_episode = None  # last completed episode index (0-based) when train ends
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later
        
        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
            # real delay needs per-step vehicle trajectories; gate is off by default.
            if hasattr(self.world, 'update_vehicle_trajectory'):
                self.world.update_vehicle_trajectory = True
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents = []
        model_name = Registry.mapping['model_mapping']['setting'].param['name']
        agent = Registry.mapping['model_mapping'][model_name](self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][model_name](self.world, i))

        # pass device from trainer to all agents for GPU support
        for ag in self.agents:
            ag.to_device(self.device)
        print(f"[Device] Moved {len(self.agents)} agent(s) to {self.device}")

        # for magd agents should share information
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)
            # re-apply device after link_agents builds models
            for ag in self.agents:
                ag.to_device(self.device)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        '''
        train
        Train the agent(s). Stops at episodes (max) or earlier when early_stop
        is enabled (ATT plateau, paired-cycle train ATT, or cycle-mean wait).

        :param: None
        :return: None
        '''
        total_decision_num = 0
        flush = 0
        self.best_att = float('inf')
        self.best_episode = None
        self.best_metric_source = None
        self.episodes_without_improve = 0
        self.stop_episode = None
        self._paired_stopper = None
        self._cycle_mean_stopper = None
        if (self.early_stop_metric == 'wait' or self.early_stop_mode == 'cycle_mean') and hasattr(
                self.world, 'track_trip_wait'):
            self.world.track_trip_wait = True
        if self.early_stop_mode == 'cycle_mean':
            from common.early_stop import CycleMeanStopper
            self._cycle_mean_stopper = CycleMeanStopper(
                cycle_len=self.early_stop_cycle_len,
                patience=self.early_stop_patience,
                min_delta=self.early_stop_min_delta,
                abs_floor=self.early_stop_abs_floor,
                min_episodes=self.min_episodes,
            )
        elif self.early_stop_mode == 'paired_cycle':
            from common.early_stop import PairedCycleStopper
            self._paired_stopper = PairedCycleStopper(
                cycle_len=self.early_stop_cycle_len,
                patience=self.early_stop_patience,
                min_delta=self.early_stop_min_delta,
                z=self.early_stop_z,
                min_episodes=self.min_episodes,
            )
        if self.early_stop:
            self.logger.info(
                "early_stop enabled: mode={}, min_episodes={}, max_episodes={}, "
                "patience={}, min_delta={:.4f}, abs_floor={:.2f}, z={:.2f}, "
                "cycle_len={}, metric={}".format(
                    self.early_stop_mode, self.min_episodes, self.episodes,
                    self.early_stop_patience, self.early_stop_min_delta,
                    self.early_stop_abs_floor, self.early_stop_z,
                    self.early_stop_cycle_len, self.early_stop_metric)
            )
        for e in range(self.episodes):
            # TODO: check this reset agent
            self._select_train_demand(e)
            self.metric.clear()
            last_obs = self.env.reset()  # agent * [sub_agent, feature]

            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))                            
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])

                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for t in range(self.action_interval):
                        # Intermediate observations are discarded; only the last obs of
                        # the interval is stored in the replay buffer / next state.
                        obs, rewards, dones, _ = self.env.step(
                            actions.flatten(),
                            collect_obs=(t == self.action_interval - 1),
                        )
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                            obs[idx], cur_phase[idx], dones[idx], f'{e}_{i//self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1
                    last_obs = obs
                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:

                    cur_loss_q = np.stack([ag.train() for ag in self.agents])  # TODO: training

                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0

            train_att = self.metric.real_average_travel_time()
            self.writeLog("TRAIN", e, train_att,\
                mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput())
            self.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps,\
                mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
            if e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents]
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes, train_att))
            median_wait = None
            if hasattr(self.world, 'get_median_waiting_time'):
                median_wait = self.world.get_median_waiting_time()
                n_wait = len(getattr(self.world, 'departed_wait_s', {}) or {})
                if median_wait is not None:
                    self.logger.info(
                        "episode:{}/{}, median wait:{} (n={})".format(
                            e, self.episodes, median_wait, n_wait)
                    )
            for j in range(len(self.world.intersections)):
                self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j],\
                     self.metric.lane_queue()[j]))
            test_att = None
            if self.test_when_train:
                test_att = self.train_test(e)
            heldout_att = None
            if self.heldout_eval_every > 0 and self.demand_heldout and (e % self.heldout_eval_every == 0):
                heldout_att = self.heldout_eval(e)

            self.stop_episode = e
            if self.early_stop_mode == 'cycle_mean':
                if heldout_att is not None:
                    self._update_checkpoint(e, heldout_att, source='heldout')
                if self._step_cycle_mean_stop(e, median_wait, heldout_att):
                    break
            elif self.early_stop_mode == 'paired_cycle':
                if heldout_att is not None:
                    self._update_checkpoint(e, heldout_att, source='heldout')
                if self._step_paired_cycle_stop(e, train_att, heldout_att):
                    break
            else:
                stop_att = self._select_early_stop_att(train_att, test_att, heldout_att)
                if self._update_early_stop(e, stop_att):
                    self.logger.info(
                        "early_stop at episode {}/{} (best ATT={:.2f} at episode {}, "
                        "patience={})".format(
                            e, self.episodes, self.best_att, self.best_episode,
                            self.early_stop_patience)
                    )
                    break
        # Prefer best checkpoint for final test(); fall back to last weights.
        self._finalize_training_checkpoint()

    def _select_early_stop_att(self, train_att, test_att, heldout_att):
        '''Pick the ATT used for plateau detection (lower is better).

        Returns None when the configured metric was not measured this episode
        (e.g. heldout only every heldout_eval_every episodes) so patience is not
        advanced on missing samples.
        '''
        metric = self.early_stop_metric
        if metric == 'heldout':
            return float(heldout_att) if heldout_att is not None else None
        if metric == 'test':
            if test_att is not None:
                return float(test_att)
            # Fallback only when test_when_train is off.
            return float(train_att) if train_att is not None else None
        if metric == 'train':
            return float(train_att) if train_att is not None else None
        # Unknown metric name: prefer test, then train.
        if test_att is not None:
            return float(test_att)
        if train_att is not None:
            return float(train_att)
        return None

    def _update_checkpoint(self, e, att, source):
        '''Save weights when att is a new absolute best for model selection.'''
        if att is None or not np.isfinite(att):
            return False
        att = float(att)
        if self.best_att is not None and np.isfinite(self.best_att) and att >= self.best_att:
            return False
        self.best_att = att
        self.best_episode = e
        self.best_metric_source = source
        [ag.save_model(e=e) for ag in self.agents]
        [ag.save_model(e='best') for ag in self.agents]
        self.logger.info(
            "new best ATT={:.2f} at episode {} (source={})".format(
                self.best_att, self.best_episode, source)
        )
        return True

    def _step_paired_cycle_stop(self, e, train_att, heldout_att):
        '''Paired-cycle stop on train ATT. Returns True if training should stop.'''
        if self._paired_stopper is None:
            return False
        stop, decision = self._paired_stopper.add(train_att)
        if decision is not None:
            self.logger.info(
                "paired_cycle cycle={} mean_att={:.2f} mean_delta={:+.2f} se={:.2f} "
                "threshold={:.2f} improved={} patience={}/{}".format(
                    decision["cycle"], decision["mean_att"], decision["mean_delta"],
                    decision["se"], decision["threshold"], decision["improved"],
                    decision["cycles_without_improve"], self.early_stop_patience)
            )
            # If there is no held-out eval, keep the cycle-mean as the selection metric.
            if heldout_att is None:
                self._update_checkpoint(e, decision["mean_att"], source='train_cycle')
        if not self.early_stop:
            return False
        if stop:
            self.logger.info(
                "early_stop at episode {}/{} (paired_cycle, last mean_att={:.2f}, "
                "best ATT={:.2f} at episode {}, source={})".format(
                    e, self.episodes,
                    decision["mean_att"] if decision else float("nan"),
                    self.best_att if self.best_att is not None else float("nan"),
                    self.best_episode, self.best_metric_source)
            )
        return stop

    def _step_cycle_mean_stop(self, e, median_wait, heldout_att):
        '''Cycle-mean stop on episode median wait. Returns True to stop.'''
        if self._cycle_mean_stopper is None:
            return False
        stop, decision = self._cycle_mean_stopper.add(median_wait)
        if decision is not None:
            self.logger.info(
                "cycle_mean cycle={} m_prev={:.2f} m_curr={:.2f} rel={:+.4f} "
                "delta={:+.2f} flat={} improved={} patience={}/{}".format(
                    decision["cycle"], decision["mean_prev"], decision["mean_curr"],
                    decision["rel"], decision["delta"], decision["flat"],
                    decision["improved"], decision["cycles_without_improve"],
                    self.early_stop_patience)
            )
            if heldout_att is None:
                self._update_checkpoint(e, decision["mean_curr"], source='train_wait_cycle')
        if not self.early_stop:
            return False
        if stop:
            self.logger.info(
                "early_stop at episode {}/{} (cycle_mean wait, m={:.2f}, "
                "best ATT={:.2f} at episode {}, source={})".format(
                    e, self.episodes,
                    decision["mean_curr"] if decision else float("nan"),
                    self.best_att if self.best_att is not None else float("nan"),
                    self.best_episode, self.best_metric_source)
            )
        return stop

    def _update_early_stop(self, e, att):
        '''
        Plateau mode: compare this episode to the running best ATT.
        Returns True when training should stop after this episode.
        '''
        if att is None or not np.isfinite(att):
            return False

        # Relative improvement: new best if ATT drops by more than min_delta fraction.
        threshold = self.best_att * (1.0 - self.early_stop_min_delta) if np.isfinite(self.best_att) else float('inf')
        improved = att < threshold
        if improved or self.best_episode is None:
            self.episodes_without_improve = 0
            self._update_checkpoint(e, att, source=self.early_stop_metric)
        else:
            self.episodes_without_improve += 1

        if not self.early_stop:
            return False
        if (e + 1) < self.min_episodes:
            return False
        if self.early_stop_patience <= 0:
            return False
        return self.episodes_without_improve >= self.early_stop_patience

    def _finalize_training_checkpoint(self):
        '''Reload best weights (if any) and save under episodes for test().'''
        loaded_best = False
        if self.best_episode is not None:
            try:
                [ag.load_model('best') for ag in self.agents]
                loaded_best = True
            except Exception:
                try:
                    [ag.load_model(self.best_episode) for ag in self.agents]
                    loaded_best = True
                except Exception as exc:
                    self.logger.warning(
                        "could not reload best checkpoint (episode {}): {}".format(
                            self.best_episode, exc)
                    )
        [ag.save_model(e=self.episodes) for ag in self.agents]
        if loaded_best:
            self.logger.info(
                "saved final checkpoint from best episode {} (ATT={:.2f}) as episode {}".format(
                    self.best_episode, self.best_att, self.episodes)
            )
        elif self.stop_episode is not None:
            self.logger.info(
                "saved final checkpoint from last episode {} as episode {}".format(
                    self.stop_episode, self.episodes)
            )

    def _select_train_demand(self, e):
        '''Pick route file for this training episode (fixed or demand-set rotation).'''
        if not hasattr(self.world, 'set_route_file'):
            return
        if self.demand_set:
            route = self.demand_set[e % len(self.demand_set)]
            self.world.set_route_file(route)
            self.logger.info("episode:{}/{}, train_demand:{}".format(e, self.episodes, route))
        elif self.demand_train_file:
            self.world.set_route_file(self.demand_train_file)

    def _run_eval_episode(self):
        '''Greedy rollout for test_steps; returns real avg travel time.'''
        obs = self.env.reset()
        self.metric.clear()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for t in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(
                        actions.flatten(),
                        collect_obs=(t == self.action_interval - 1),
                    )
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)
                self.metric.update(rewards)
            if all(dones):
                break
        return self.metric.real_average_travel_time()

    def heldout_eval(self, e):
        '''Evaluate (no learning) on held-out demand files and log mean ATT.'''
        atts = []
        for route in self.demand_heldout:
            if hasattr(self.world, 'set_route_file'):
                self.world.set_route_file(route)
            att = self._run_eval_episode()
            atts.append(att)
            self.logger.info(
                "HELDOUT episode:{}/{}, demand:{}, travel time:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(
                    e, self.episodes, route, att, self.metric.rewards(),
                    self.metric.queue(), self.metric.delay(), int(self.metric.throughput()))
            )
            self.writeLog("HELDOUT", e, att, 100, self.metric.rewards(),
                          self.metric.queue(), self.metric.delay(), self.metric.throughput())
            if hasattr(self.world, 'get_median_waiting_time'):
                mw = self.world.get_median_waiting_time()
                if mw is not None:
                    self.logger.info(
                        "HELDOUT episode:{}/{}, demand:{}, median wait:{}".format(
                            e, self.episodes, route, mw)
                    )
        mean_att = float(np.mean(atts)) if atts else 0.0
        self.logger.info(
            "HELDOUT_MEAN episode:{}/{}, travel time:{} (n={})".format(
                e, self.episodes, mean_att, len(atts))
        )
        self.writeLog("HELDOUT_MEAN", e, mean_att, 100, 0, 0, 0, 0)
        return mean_att

    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        att = self._run_eval_episode()
        self.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(\
            e, self.episodes, att, self.metric.rewards(),\
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
        self.writeLog("TEST", e, att,\
            100, self.metric.rewards(),self.metric.queue(),self.metric.delay(),self.metric.throughput())
        if getattr(self.env.world, 'crossing_proxy_ctrl', None) is not None:
            self.env.world.crossing_proxy_ctrl.log_summary()
        return att

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()
        if not drop_load:
            [ag.load_model(self.episodes) for ag in self.agents]
        attention_mat_list = []
        if self.demand_heldout and hasattr(self.world, 'set_route_file'):
            self.heldout_eval(self.episodes)
            return self.metric
        if self.demand_train_file and hasattr(self.world, 'set_route_file'):
            self.world.set_route_file(self.demand_train_file)
        obs = self.env.reset()
        for a in self.agents:
            a.reset()
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                actions = np.stack(actions)
                rewards_list = []
                for t in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(
                        actions.flatten(),
                        collect_obs=(t == self.action_interval - 1),
                    )
                    i += 1
                    rewards_list.append(np.stack(rewards))
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (self.metric.real_average_travel_time(), \
            self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))
        if getattr(self.env.world, 'crossing_proxy_ctrl', None) is not None:
            self.env.world.crossing_proxy_ctrl.log_summary()
        return self.metric

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" +\
            "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

