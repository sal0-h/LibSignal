# How DQN actually trains on SUMO Grid4×4 in LibSignal

A code-traced execution model of **DQN traffic-signal training** on `--world sumo --network sumo4x4`. This is not a DQN textbook. It is a map of what this repository does from `python run.py` until `{episode}_{rank}.pt` files exist.

**Code version:** commit `64ad9a55d0e7c081bb9c379cb85da5180c62b265` (current `master` at time of writing).  
**Live probe:** a SUMO `libsumo` world was constructed from `configs/sim/sumo4x4.cfg` and the merged `dqn.yml` settings; intersection IDs, phase counts, lane lists, network shapes, and the phase-application state machine were read off the running objects.

**Two experiment configs appear below.** Unless a subsection says otherwise, numbers are the default command:

```bash
python run.py --agent dqn --world sumo --network sumo4x4 --seed 42 --ngpu -1
```

The demand-bag variant is `--agent dqn_odh_l1` (same DQN class, different trainer/world YAML). Differences are tabulated in §0.

Related docs (do not duplicate them): [AGENT_OBSERVATIONS.md](AGENT_OBSERVATIONS.md) (why there are 32 agents), [TSC_CONFIG_REPORT.md](TSC_CONFIG_REPORT.md) (every YAML key), [SIGNAL_CONTROL_THEORY.md](SIGNAL_CONTROL_THEORY.md) (what the 8 green states mean).

---

## 0. Verdicts (read this first)

1. **An episode is a fixed-length SUMO rollout**, not a “all vehicles finished” episode. Default: **3600 simulation seconds**. The Gym `done` flag is hard-coded `False` and is **not stored** in replay, so DQN always bootstraps as if the task were continuing.
2. **An RL step (decision) happens every `action_interval = 10` SUMO seconds.** The same discrete action is applied on each of those 10 `env.step()` calls. Reward stored in replay is the **mean of 10 per-second rewards**. The next-state is the observation at the **end** of the 10 seconds; intermediate observations are discarded.
3. **Grid4×4 is not 16 DQN agents.** `sumo4x4` has **32 SUMO traffic-light IDs**. LibSignal builds **32 separate `DQNAgent` objects**. **32 independent neural-network policies are learned.** There is no weight sharing. The 16 interior junctions (`A0`–`D3`) are the real 8-phase controllers; the 16 fringe portals are 1-action dummy lights that still get a full DQN (replay, optimizer, target net).
4. **One training sample is one intersection’s transition** `(s, phase, a, r, s', phase')` written into **that intersection’s own** `deque`. Experiences from different intersections are **never mixed**.
5. **Gradient updates:** after a global warmup of `learning_start = 1000` decisions, **every decision** calls `ag.train()` on **every agent**. That is **32 independent RMSprop steps per decision**, each on a batch of 64 from that agent’s buffer.
6. **What is optimized:** each intersection’s online `DQNNet` Q-function, by MSE to a one-step TD target from its own target network, using only **local incoming lane counts + a one-hot of the last commanded green**. No neighbor features, no global reward.

---

## 0.1 Config values actually used

YAML merge: `configs/tsc/<agent>.yml` includes `base.yml`; child keys win (`utils/logger.py` `build_config` / `merge_dicts`). Simulator file: `configs/sim/sumo4x4.cfg`.

| Parameter | Default `--agent dqn` | `--agent dqn_odh_l1` | Source |
|-----------|----------------------:|---------------------:|--------|
| `model.name` | `dqn` | `dqn` (same class) | `dqn.yml` |
| `train_model` / `test_model` | True / True | True / True | `dqn.yml` / `base.yml` |
| `episodes` | **200** | **100** | `base.yml` / `od_hub_1800_base.yml` |
| `steps` / `test_steps` | **3600** / 3600 | **1800** / 1800 | same |
| `action_interval` | **10** | 10 | `base.yml` |
| `learning_start` | **1000** (overrides base 5000) | 1000 | `dqn.yml` |
| `buffer_size` | **5000** | 5000 | `dqn.yml` |
| `update_model_rate` | **1** | 1 | `base.yml` |
| `update_target_rate` | **10** | 10 | `base.yml` |
| `test_when_train` | **True** | **False** | `base.yml` / `od_hub_1800_base.yml` |
| `heldout_eval_every` | 0 (off) | **10** | odh YAML |
| `gamma` | **0.95** | 0.95 | `base.yml` |
| `learning_rate` | **0.001** | 0.001 | `base.yml` |
| `batch_size` | **64** | 64 | `base.yml` |
| `grad_clip` | **5.0** | 5.0 | `base.yml` |
| `epsilon` / decay / min | **0.1 / 0.995 / 0.01** | same | `dqn.yml` (overrides base 0.5 / 0.99 / 0.05) |
| `phase` / `one_hot` | **True / True** | True / True | `dqn.yml` |
| demand file | `grid4x4.rou.xml` every episode | rotate `train_00`…`train_09.rou.xml` | sim cfg / `od_hub_1800_base.yml` |
| SUMO `interval` | 1.0 s | 1.0 s | `sumo4x4.cfg` |
| `world.step_length` | **1** (hardcoded) | 1 | `world_sumo.py` |
| `max_distance` | **200 m** (hardcoded) | 200 | `world_sumo.py` |

`trainer.yellow_length` (5 in `base.yml`) is copied to `TSCTrainer.yellow_time` and **never used**. `sumo4x4.cfg`’s `"yellow_length": 3` is **not read** by the SUMO world. Yellow timing comes from `Intersection.yellow_phase_time = min(original TLS phase durations)` — see §4.

---

## 1. Big picture: episode, RL step, transition, update

### 1.1 What is an episode?

In `TSCTrainer.train()` (`trainer/tsc_trainer.py`):

```text
for e in range(episodes):          # 200 (default)
    maybe swap demand file
    metric.clear()
    last_obs = env.reset()         # World.reset() restarts SUMO
    for a in agents: a.reset()     # rebuild generators; do NOT reset nets/buffer/ε
    i = 0
    while i < steps:               # 3600 (default)
        ... one or more RL decisions ...
    log TRAIN metrics
    maybe save checkpoint
    maybe greedy train_test()
```

Verified:

- Episode length is **`trainer.steps` simulation seconds**, because each `env.step()` advances SUMO by `step_length = 1` second (`World.step` → `step_sim` → one `simulationStep()` with `step_ratio = 1`).
- `TSCEnv.step` always returns `dones = [False] * n_agents` (`environment.py`). The `if all(dones): break` in the trainer is **dead** on this path. Training never ends because the network emptied.
- Average travel time / throughput at episode end count **only vehicles that have arrived** (`World.vehicles`). Vehicles still on the net are omitted from ATT.

So an episode is: **one SUMO process start → 3600 one-second ticks under a (possibly changing) joint signal policy → close/restart.**

### 1.2 What is an RL step?

A **decision** is the body of `if i % action_interval == 0` in `TSCTrainer.train()`:

1. Read each agent’s **commanded** phase (`virtual_phase`) via `get_phase()`.
2. Choose a joint action: uniform random if still in warmup, else each agent’s `get_action(..., test=False)` (ε-greedy).
3. Hold that action for **`action_interval` (10) consecutive `env.step()` calls**.
4. Average the 10 per-step rewards.
5. `remember(...)` one transition per agent.
6. `total_decision_num += 1`.
7. Possibly `ag.train()` and/or `ag.update_target_network()` for **all** agents.

Between two decisions, SUMO runs for **10 seconds**. The agent cannot change its discrete action during that window; `world.step(actions)` is called every second with the **same** integer.

### 1.3 When is a transition built and inserted?

**After** the 10-second interval, in the trainer, not inside the env:

```text
ag.remember(last_obs, last_phase, action, actions_prob, mean_reward,
            obs, cur_phase, done, key)
```

`DQNAgent.remember` (`agent/dqn.py`) appends to a per-agent `deque`:

```text
(key, (last_obs, last_phase, actions, rewards, obs, cur_phase))
```

**Dropped on purpose (verified):** `done`, `actions_prob`, and `key` are unused for learning. `get_action_prob` is always `None` for DQN (`agent/base.py`). `done` is always `False` anyway.

`last_obs` is the observation from the **start** of the interval (or env.reset at the first decision). `obs` is taken only on the **last** of the 10 env steps (`collect_obs=(t == action_interval - 1)`). Intermediate `get_ob()` calls are skipped.

### 1.4 When does neural-network training happen?

Same `while` iteration, **after** `remember`:

```text
if total_decision_num > learning_start
   and total_decision_num % update_model_rate == update_model_rate - 1:
        [ag.train() for ag in self.agents]
```

With `update_model_rate = 1`, the modulo test is `n % 1 == 0`, i.e. **always**. So:

- Decisions `1 … 1000`: random actions, **no** `train()`.
- Decision `1001` onward: **one `train()` per agent per decision**.

Target nets: `update_target_rate = 10` → hard copy when `total_decision_num % 10 == 9` **and** `> 1000`, i.e. first at global decision **1009**, then every 10 decisions.

`total_decision_num` is **global across episodes**, not reset.

### 1.5 How many optimizer updates vs environment steps?

Let \(D = \texttt{steps} / \texttt{action\_interval}\) decisions per episode, \(N=32\) agents, \(E\) episodes, \(L=1000\) warmup decisions.

| Quantity | Default DQN | `dqn_odh_l1` |
|----------|------------:|-------------:|
| SUMO seconds / episode | 3600 | 1800 |
| RL decisions / episode \(D\) | **360** | **180** |
| Transitions / episode | \(32 \times D\) = **11 520** | **5 760** |
| Global decisions \(E \times D\) | 200 × 360 = **72 000** | 100 × 180 = **18 000** |
| `train()` calls **per agent** | \(72000-1000\) = **71 000** | **17 000** |
| RMSprop steps **in the run** | \(32 \times 71000\) = **2 272 000** | \(32 \times 17000\) = **544 000** |
| Target copies per agent | 7100 | 1700 |
| Extra greedy rollouts | 200 × 3600 s (`test_when_train`) | held-out eval every 10 episodes × 3 files |

Yes: **16 interior intersections (and 16 portals) each emit their own transition every 10 s**, so you get **32× as many independent samples as a single-intersection experiment**, not 32× as many updates to one shared net.

### 1.6 What stops training?

`TSCTask.run()` (`task/task.py`): if `train_model`, run `trainer.train()` to completion of the episode loop; then if `test_model`, run `trainer.test()`. There is no early stopping, no validation-based halt, no “converged Q-loss” criterion.

### 1.7 Timeline that matches the code

```text
Runner.run()
  TSCTask.run()
    TSCTrainer.train()
      for episode e = 0 … 199:
        [_select_train_demand(e)]            # no-op unless demand_set is configured
        last_obs ← TSCEnv.reset()
            World.reset()                    # libsumo.close + start; new Intersection objects
            observe all TLs; _update_infos()
            last_obs ← [ag.get_ob() for ag]  # generators may still point at pre-reset Intersections
        for ag in agents: ag.reset()         # rebind generators to new Intersection objects
                                             #   networks, optimizers, ε, replay KEPT

        i = 0
        while i < 3600:
          if i % 10 == 0:                    # RL decision
            last_phase ← [ag.get_phase()]    # virtual_phase (commanded green index)
            if total_decision_num <= 1000:
                actions ← [ag.sample()]      # uniform over that TL’s greens
            else:
                actions ← [ag.get_action(obs, phase, test=False)]  # ε-greedy
            rewards_list = []
            for t in 0..9:                   # 10 SUMO seconds, SAME action
                World.step(actions):
                    for each TL: Intersection.pseudo_step(action)
                    simulationStep()         # 1 second
                    Intersection.observe()
                    _update_infos()
                rewards_list ← ag.get_reward()   # every second
                obs ← ag.get_ob() only if t==9   # next_state
                i += 1
            r ← mean(rewards_list over 10 s)     # per agent
            remember(...)                        # 32 independent deques
            total_decision_num += 1
            last_obs ← obs
          if total_decision_num > 1000:
              32 × DQNAgent.train()              # 32 independent gradient steps
          if total_decision_num > 1000 and n % 10 == 9:
              32 × update_target_network()
        log TRAIN (ATT, q_loss mean, reward, queue, delay, throughput)
        every 5 episodes: save target nets  {e}_{rank}.pt
        if test_when_train: greedy 3600 s rollout (no remember, no train)

    TSCTrainer.test()                        # another greedy 3600 s; does NOT load .pt
                                             #   (drop_load=True); uses in-memory online nets
```

---

## 2. The 4×4 multi-intersection setup: how many policies?

### 2.1 Construction (not an inference from the word “multi-agent”)

```text
World.__init__ / World.reset
    intersection_ids = eng.trafficlight.getIDList()     # 32 IDs
    one Intersection per ID

TSCTrainer.create_agents
    agent0 = DQNAgent(world, rank=0)
    num_agent = len(world.intersections) / agent0.sub_agents
    # DQNAgent.sub_agents = 1  →  num_agent = 32
    agents[i] = DQNAgent(world, rank=i)   for i = 0..31

TSCEnv.__init__
    n_agents = len(agents) * agents[0].sub_agents
    assert len(intersection_ids) == n_agents
```

Live `getIDList()` order (also agent `rank`):

```text
A0 A1 A2 A3 B0 B1 B2 B3 C0 C1 C2 C3 D0 D1 D2 D3
bottom0..3  left0..3  right0..3  top0..3
```

Interior 16: `type="traffic_light_right_on_red"` grid nodes, **8 green phases**, 12 incoming lanes.  
Fringe 16: portal nodes in `grid4x4.tll.xml` with states `G` / `y` / `r`; after LibSignal’s green filter, **1 green**, **1 incoming lane**.

There is **no filter** that drops portals (`docs/AGENT_OBSERVATIONS.md`).

### 2.2 Are parameters shared?

**No.** Each `DQNAgent.__init__` calls `_build_model()` twice (online + target), creates its own `optim.RMSprop`, its own `deque`, its own `self.epsilon`. `create_agents` does not copy `state_dict`s.

Interior vs fringe nets even have **different shapes** (live):

| Role | `ob_length` | `|A|` | `DQNNet` |
|------|------------:|------:|----------|
| Interior (e.g. `A0`, rank 0) | 12 + 8 = **20** | **8** | `Linear(20,20) → 20 → 8` |
| Fringe (e.g. `top0`, rank 28) | 1 + 1 = **2** | **1** | `Linear(2,20) → 20 → 1` |

A 1-action fringe DQN can only output Q(s, a=0). It still samples minibatches and takes gradient steps. Those updates **do not** affect interior controllers.

### 2.3 One policy or many?

**32 policies.** “When I train DQN on Grid4x4, how many neural-network policies are actually being learned?” → **32**, of which **16 are 8-phase intersection controllers** and **16 are degenerate 1-phase portal controllers**.

Checkpoints: `data/output_data/tsc/sumo_dqn/<network>/<prefix>/model/{episode}_{rank}.pt` with `rank ∈ {0,…,31}`.

### 2.4 How this differs from other agents (only where it clarifies DQN)

| Agent | Objects created on `sumo4x4` | Parameters | Replay |
|-------|-----------------------------:|------------|--------|
| **DQN** | **32** `DQNAgent` | **independent** | **32 deques** |
| PressLight | 32 `PressLightAgent` | independent (same factory) | 32 deques |
| IPPO (`ppo_pfrl`) | 32 | independent | PFRL buffers per agent |
| MaxPressure | 32 | no NN | n/a |
| **CoLight** | **1** object, `sub_agents = 32` | **one GNN**, shared | **one** buffer of joint-ish samples |
| **MPLight** | **1** object, `sub_agents = 32` | **one FRAP net**, shared | PFRL, `update_interval=sub_agents` |

CoLight/MPLight are the algorithms for which “one policy controlling the 4×4” is literally true. DQN in this codebase is **independent Q-learning**, one learner per TLS ID.

---

## 3. State / observation construction

### 3.1 Generators (one intersection)

`DQNAgent.__init__` / `reset`:

| Role | Class | Args |
|------|--------|------|
| Observation | `LaneVehicleGenerator` | `fns=['lane_count']`, `in_only=True`, `average=None` |
| Phase (not in Gym obs) | `IntersectionPhaseGenerator` | `targets=['cur_phase']` |
| Reward | `LaneVehicleGenerator` | `fns=['lane_waiting_count']`, `in_only=True`, `average='all'`, `negative=True` |

`TSCEnv` returns **only** `agent.get_ob()`. Phase is concatenated later inside `get_action` and `_batchwise` because `model.phase: True` and `one_hot: True`.

### 3.2 What `lane_count` is

Each SUMO second, `Intersection.observe(step_length=1, max_distance=200)` walks vehicles on the intersection’s registered lanes:

- Vehicle must be on that lane **and** have `getNextTLS` distance ≤ **200 m**. Approach edges are **~273–286 m**, so the **far ~75–86 m of each approach is invisible**.
- Partial-observability hooks (`obs_penetration`, `obs_count_noise_std`) default to off.
- `lane_count` is the number of those vehicles (integer). It is **not** a SUMO `lastStepVehicleNumber` dump of the full lane.

Incoming roads for **A0** (live, already sorted NESW-ish by `_sort_roads`):

```text
A1A0     lanes A1A0_0, A1A0_1, A1A0_2          # from north
B0A0     lanes B0A0_0, B0A0_1, B0A0_2          # from east
bottom0A0 lanes bottom0A0_0, _1, _2            # from south
left0A0  lanes left0A0_0, _1, _2               # from west
```

Lane order inside a road: sorted by the trailing lane index (`RIGHT=True` → 0,1,2). Generator docstring claims “waiting = speed < 0.1”; that is **true for SUMO’s own waiting time**, but see §5 for the sticky `waiting_times` dict actually used for reward.

Outgoing lanes are **not** in the DQN observation (`in_only=True`). PressLight *does* include them; that is a different agent.

### 3.3 Shape

`get_ob()` does `np.array([ob_generator.generate()])` → shape **`(1, 12)`** on interior, **`(1, 1)`** on fringe (`top0` live: lane `A3top0_2` only — the portal TLS’s controlled-link mapping, not all three physical lanes).

Policy input (training and greedy):

\[
x = \big[\; n_{\ell_0},\ldots,n_{\ell_{11}} \;\Vert\; \mathbf{1}_{\text{phase}} \;\big] \in \mathbb{R}^{20}
\]

`utils.idx2onehot(phase, n_actions)` with `phase` shape `(1,)` → `(1, 8)`.

**Neighbors:** none. No adjacency, no other intersection’s queues.

**Current phase:** yes, as one-hot of `virtual_phase` (last **commanded green index**), not SUMO’s raw TLS index (which can be a generated clearance phase ≥ 8).

**Raw vs aggregated:** per-lane vehicle counts, unnormalized. `vehicle_max` is stored on the agent and **unused** (CoLight divides by it; DQN does not).

### 3.4 Conceptual vector for interior intersection `A0`

```text
intersection A0  (rank 0)
Gym obs  = get_ob()            shape (1, 12)     -- NOT what the net sees alone
phase    = virtual_phase       in {0,…,7}

policy input x ∈ R^{20} =
[
  # incoming lane_count, 200 m detector, order = generator.lanes
  n(A1A0_0), n(A1A0_1), n(A1A0_2),          # north approach, 3 lanes
  n(B0A0_0), n(B0A0_1), n(B0A0_2),          # east
  n(bottom0A0_0), n(bottom0A0_1), n(bottom0A0_2),
  n(left0A0_0),  n(left0A0_1),  n(left0A0_2),
  # one-hot of commanded green (phase True, one_hot True)
  1[phase==0], …, 1[phase==7]
]
```

At `t = 0` after reset this is all zeros and phase `0` (empty net). The eight greens are the unique non-yellow, not-all-red states from `grid4x4.tll.xml`, rewritten with duration 1 s in the RL program (see [SIGNAL_CONTROL_THEORY.md](SIGNAL_CONTROL_THEORY.md) for movement decoding). DQN does not use `mplight.yml` `phase_pairs`; **action index = index in that filtered green list**.

---

## 4. Action space

### 4.1 What action `k` means

`self.action_space = Discrete(len(intersection.phases))`.

- Interior: **8 actions**, `k ∈ {0,…,7}`.  
  `k` selects `green_phases[k]`, which is installed as **SUMO program index `k`** in the rewritten `_rl` program (greens first, then generated clearance phases 8…63).
- Fringe: **1 action**, always `0`.
- **Yellow / all-red are not actions.** `World.generate_valid_phase` drops any original state containing `y`, and drops states that are only `r`/`s`.

This is **phase selection**, not “keep / switch” and not “extend current green by Δt”.

### 4.2 Can the agent switch every RL step? Minimum green?

DQN has **no** `t_min` (MaxPressure does). Intended control loop:

- A new integer may be chosen every 10 s.
- That integer is fed to `Intersection.pseudo_step` **every second** for those 10 s.
- If the integer is unchanged, the same green is re-commanded.
- If it changes, the world is supposed to insert a clearance phase then the new green (`prep_phase` / `create_yellows`).

So the **shortest green the *agent* can request** is one action interval = **10 s**. There is no additional DQN-side minimum green.

### 4.3 Where the integer becomes a SUMO command

```text
TSCTrainer  →  TSCEnv.step(actions.flatten())
           →  World.step(action)
           →  intersections[i].pseudo_step(action[i])
           →  eng.trafficlight.setPhase(id, phase_index)   # libsumo or traci
           →  eng.simulationStep()
```

`setPhase` is in `Intersection._change_phase` and `prep_phase` (`world/world_sumo.py`).

### 4.4 What actually happens on Grid4×4 interior (verified, not textbook)

Original interior TLS programs in `grid4x4.tll.xml` are **eight greens of duration 30 s and no yellows**. Therefore:

```text
yellow_phase_time = min(original phase durations) = 30
```

`current_phase_time` starts at 0 each episode and is incremented every SUMO second. The branch that would reset it to 0 sits **inside** `current_phase_time < yellow_phase_time` and also requires `current_phase_time > yellow_phase_time` — a contradiction. **It never resets.**

Live 40-second probe on `A0` (action 0 for 10 s, 1 for 20 s, 2 after t=30):

- For **the first 30 s**, `pseudo_step` stays in the “clearance” branch and **re-applies `current_phase`**, ignoring new actions except at `current_phase_time == 0` (the first second). Requesting action 1 at t=10 **did not change the SUMO phase**.
- From **t ≥ 30 s** onward, `current_phase_time >= 30` → **immediate `_change_phase(action)` every second**. Switching 1→2 applied **instantly**, with no clearance interval.

So for this network, after the first 30 seconds of every episode, **phase changes are instantaneous `setPhase` to the commanded green**. The 56 generated “yellow” phases in the 64-phase RL program are largely unused after that point. (YAML `yellow_length` does not change this.)

Fringe portals: `yellow_phase_time = 3`, one green `G`; the agent has nothing to switch to.

**Intended story** (comments in `pseudo_step`): on a phase change, look up `yellow_dict["i_j"]`, `setPhase` to that clearance index, hold until `yellow_phase_time`, then `setPhase` to the new green; if the action matches the current green, just stay. **On Grid4×4 interior, that story only has a chance in the first 30 s, and after that switches are immediate.**

---

## 5. Reward (Grid4×4 DQN)

### 5.1 Generator

Same class as the observation, different statistic:

```text
LaneVehicleGenerator(..., ["lane_waiting_count"], in_only=True, average="all", negative=True)
```

then `DQNAgent.get_reward`:

```text
reward = squeeze(generate()) * 12
```

PressLight does **not** multiply by 12 and uses intersection **pressure** (`Σ in − Σ out` vehicle counts), not waiting count. DQN’s reward is **local**; there is **no** global term in the TD target. Metrics log a *sum* of local rewards for plotting only (`Metrics.rewards`).

### 5.2 Mathematical meaning (interior)

`average="all"` is the mean of **per-road means**, not a masked sum. Interior A0 has 4 incoming roads × 3 lanes. That coincides with the mean over all 12 lanes:

\[
\bar q = \frac{1}{12}\sum_{\ell \in \text{in}(i)} q_\ell,
\quad
r_{\text{gen}} = -\bar q,
\quad
r = 12 \cdot r_{\text{gen}} = -\sum_{\ell \in \text{in}(i)} q_\ell.
\]

**For a 12-incoming-lane intersection, the stored reward is minus the total incoming waiting count.** The `* 12` is a CityFlow-era 12-movement scale factor that happens to recover a sum here.

Fringe (1 lane): \(r = -12\, q_{\text{that lane}}\) — the same `*12` **over-weights** portals in the logged sum of rewards; it does not leak into interior TD updates because buffers are separate.

### 5.3 What “waiting” is (implementation vs docstring)

Docstring: vehicles with speed < 0.1 m/s.  
Implementation (`Intersection.observe`):

- If SUMO `getWaitingTime(v) > 0` (speed has been < 0.1 since last motion), the vehicle enters `waiting_times`.
- While it remains in that dict, **every subsequent second adds `step_length`**, even if the vehicle is moving again.
- `lane_waiting_count` counts vehicles on the lane with `wait > 0`.
- Cleanup on departure is **commented out**. Leftover dict entries do not affect counts (only current lane occupants are iterated) but a vehicle that once queued is still counted as waiting **until it leaves this intersection’s observed lanes**.

So the reward is **not** a clean instantaneous stopped-vehicle count. It is closer to “vehicles on the approach that have queued at least once this episode (within 200 m).”

### 5.4 When it is measured

- Computed **every SUMO second** (even when `collect_obs=False`).
- The replay scalar is the **arithmetic mean of the 10 values** in the action interval (`np.mean(rewards_list, axis=0)`).
- It is **not** an integral of waiting time and **not** only the value at the end of the interval.

Relative to the action: the action is held during those 10 seconds; \(r\) is the average local waiting snapshot **while that action (and whatever SUMO phase actually resulted, see §4.4) was in force**.

---

## 6. Replay buffer

| Question | Answer in this code |
|----------|---------------------|
| Structure | `collections.deque(maxlen=buffer_size)` per `DQNAgent` |
| Capacity | **5000** transitions **per intersection** |
| One entry | `(key, (obs_t, phase_t, action, reward, obs_{t+1}, phase_{t+1}))` |
| `obs_*` | `ndarray` shape `(1, L)` |
| `phase_*` | `int8` array shape `(1,)` — commanded green |
| `action` | `ndarray` shape `(1,)` from `argmax` or `randint` |
| `reward` | `numpy.float64` scalar (already ×12 and interval-averaged) |
| Sampling | `random.sample(deque, batch_size)` — **uniform, without replacement in the batch** |
| Prioritized replay | **No** |
| Mixing across intersections | **No** |
| Mixing across episodes | **Yes**, until the deque evicts the oldest |
| Cleared on episode end | **No** |
| Cleared between runs | Yes — new process, new agents |
| When training may start | After **1000 global decisions**; each buffer then has 1000 samples (> 64) |
| LMDB `OnFlyDataset` | `initiate()` is called; `flush()` in the trainer is **commented out**. Offline dataset is unused. |

Eviction: 5000 / 360 ≈ **13.9 episodes** of history per interior agent on the default run (27.8 episodes on 180-decision odh episodes). Early-episode transitions from episode 0 are gone long before episode 199.

The just-written transition **is** in the deque before `train()` in the same decision, so it can appear in that minibatch.

---

## 7. One DQN gradient update, mapped to code

Textbook target (what the math is aiming at):

\[
y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
\]

LibSignal (`DQNAgent.train`, `agent/dqn.py`):

| Symbol | Code |
|--------|------|
| \(s\) | `_batchwise`: `obs_t` concatenated with one-hot `phase_t` → `b_t`, shape `(64, 20)` interior |
| \(s'\) | same for `obs_tp`, `phase_tp` → `b_tp` |
| \(a\) | stored action, flattened to `(64,)` |
| \(r\) | stored scalar reward, `(64,)` |
| \(\gamma\) | `self.gamma = 0.95` |
| \(Q_{\theta^-}\) | `self.target_model` (`DQNNet`) |
| \(Q_\theta\) | `self.model` |
| \(\max_{a'} Q(s',a')\) | `torch.max(self.target_model(b_tp, train=False), dim=1)[0]` |
| Terminal factor \((1-d)\) | **absent** (`done` not in the tuple) |
| Double DQN | **No** — greedy action and evaluation both use the target net |
| Dueling / PER / n-step | **No** |

**Online net:** 3-layer MLP, ReLU on the first two layers, linear output (`DQNNet`). Hidden width **20** (not configurable in `dqn.yml`).

**Target net:** identical architecture; `load_state_dict` hard copy (`update_target_network`).

**Loss (differs from textbook scalar TD MSE):**

```text
q_values = model(b_t)                         # (64, 8), with grad
target_f = q_values.detach().clone()
target_f[i, a_i] = y_i                        # only taken action replaced
loss = MSELoss(reduction='mean')(q_values, target_f)
```

For every non-taken action, target equals the detached Q, so that component’s error is 0. Mean over **all 8 outputs** ⇒

\[
\mathcal{L} = \frac{1}{8}\,(Q_\theta(s,a) - y)^2
\]

on interior agents (divide by 1 on fringe). Gradients are **1/|A| smaller** than MSE on the scalar \(Q(s,a)\). PressLight still uses an older two-forward version of the same trick; DQN was later changed to a single forward (comment in `train()`).

**Optimizer:** `torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, centered=False, eps=1e-7)`. (PyTorch defaults would be `alpha=0.99`, `eps=1e-8`.) Only **online** parameters; the target net has no optimizer.

**Grad clip:** `clip_grad_norm_(model.parameters(), 5.0)` then `optimizer.step()`.

**ε schedule:** **not** a function of wall time. After every successful `train()`:

```text
if epsilon > epsilon_min: epsilon *= epsilon_decay
```

Starting 0.1, decay 0.995, min 0.01 ⇒ \(0.1 \times 0.995^k = 0.01\) at \(k \approx 460\) **gradient steps** (about **1.3 episodes** after warmup on the default run). Each of the 32 agents has its own `epsilon` float; they decay in lockstep because they `train()` equally often. Exploration draws use the **global** `numpy` RNG, so the agents’ random bits are coupled.

Warmup uses `sample()` (**always random**), not ε-greedy. ε does not decay during warmup because `train()` is not called.

### 7.1 Pseudocode for one LibSignal DQN update

```text
# called independently for agent i, after remember()
batch = uniform_sample(agent.replay_buffer, 64)

s  = concat(obs_t,  one_hot(phase_t,  n_A))    # (64, ob_length)
s' = concat(obs_tp, one_hot(phase_tp, n_A))

y = r + 0.95 * max_over_actions target_net(s')   # no (1-done)

q = online_net(s)                                # (64, n_A)
q_target_vec = q.detach().clone()
q_target_vec[range(64), a] = y

loss = mean( (q - q_target_vec)^2 )              # averaged over n_A outputs
loss.backward()
clip_grad_norm(online_net, 5.0)
RMSprop.step()
epsilon = max(0.01, epsilon * 0.995)

# separately, every 10th global decision after warmup:
target_net.load_state_dict(online_net.state_dict())
```

---

## 8. Training loop across one episode (Grid4×4, 32 TLs)

Assume default config: 3600 s, 10 s decisions, 16 interior + 16 fringe.

```text
t_sim = 0 s
  env.reset(); 32 observations (mostly zeros)
  32 × virtual_phase = 0

decision d = 1 (global total_decision_num becomes 1 after remember)
  32 phases read
  if still warming up: 32 uniform random actions
     interior a ∈ {0..7}, fringe a = 0
  SUMO seconds t = 1..10:
     32 × pseudo_step(a); simulationStep(); observe; reward
  r_i = mean of 10 local rewards
  32 transitions appended (next_state = obs at t=10)
  no train() yet (1 ≤ 1000)

decision d = 2 … 360
  same pattern; i runs 10,20,…,3600
  episode ends when i hits 3600

# later episodes: when total_decision_num hits 1001 (during episode 2:
#   360 + 360 + 281 = 1001), that decision is ε-greedy AND train() runs
#   32 times. Thereafter every decision: 32 gradient steps.
```

**Counts for this episode (default):**

| | Episode 0 (warmup) | A typical episode after warmup |
|--|-------------------:|-------------------------------:|
| RL decisions | 360 | 360 |
| Transitions written | 32 × 360 = 11 520 | 11 520 |
| Gradient steps | 0 | 32 × 360 = **11 520** |
| Target copies | 0 | 36 per agent (360/10) except episode that contains the 1009 boundary |

Episode 2 is the first with learning: decisions 721–1080 globally; train from 1001; first target copy at 1009.

If `test_when_train: True`, after the training rollout there is a **second** 3600 s greedy simulation that does **not** write replay or call `train()`.

---

## 9. What persists from episode 1 to episode 2

| State | Survives `env.reset()` / `ag.reset()`? |
|-------|----------------------------------------|
| Online `DQNNet` weights | **Yes** (agents are created once in `create_agents`) |
| Target net weights | **Yes** |
| RMSprop moments | **Yes** |
| `epsilon` | **Yes** (continues decaying only when `train()` runs) |
| Replay `deque` | **Yes** (not cleared) |
| `total_decision_num` | **Yes** (trainer field) |
| SUMO vehicles, clocks, TLS programs | **No** — `World.reset` closes and restarts SUMO, rebuilds `Intersection` objects (`current_phase_time` back to 0, `virtual_phase` 0) |
| Route file | Same file by default; odh rotates (see §10) |
| Python/NumPy/Torch RNGs | **Not reset**; they continue |
| SUMO `--seed` | Re-passed on every `sumo` start (same seed each episode) |

`ag.reset()` only rebuilds generators bound to the **new** `Intersection` objects. That is required because `World.reset` throws the old objects away.

Episode 2 adds learning because the **same Q-networks and replay** see a new trajectory (and, after warmup, new gradient steps). It is not a fresh DQN.

---

## 10. Demand and generalization

### 10.1 Default `grid4x4.rou.xml`

- **1473 vehicles**, explicit `depart` times from 0 through **3494** s, **fixed routes** (not a flow sampler). Last departures sit inside a 3600 s episode.
- Every training episode starts SUMO with this same file and the same `--seed`.
- Planned demand is therefore the **same insertion script** every episode, not a new sample from an OD distribution.

That is **not** the same traffic movie. Realized trajectories still differ because:

1. **The policy’s actions differ** (random warmup, then ε-greedy, then a changing Q). Different greens → different queues → different downstream arrivals.
2. SUMO insertion can **delay** a vehicle if its depart lane is blocked; that couples demand realization to congestion.
3. Lane-changing / car-following still use SUMO RNG, but the seed is **the same each restart**, so given identical control and identical insertion, those draws would repeat. Control is not identical across episodes once learning starts.

What a policy **can memorize**: this one 4×4 geometry, this one 8-phase encoding, this one 1473-vehicle timetable and route set, plus whatever diversity ε-greedy and self-induced congestion produce. It is **not** trained to an explicit distribution over OD matrices unless you change the route file.

### 10.2 Demand bag (`--agent dqn_odh_l1`)

`configs/tsc/od_hub_1800_base.yml` sets `world.demand_set` to ten `train_XX.rou.xml` files. `TSCTrainer._select_train_demand(e)` does `demand_set[e % 10]` then `World.set_route_file` **before** `env.reset()`.

- Episode 0 → `train_00`, episode 1 → `train_01`, …, episode 10 → `train_00` again.
- Episode length 1800 s; `test_when_train` off; every 10 episodes a greedy eval on three `hold_*.rou.xml` files.
- Final `test()` uses `demand_heldout` if set (`TSCTrainer.test`).

Multiple route files change the learning problem from “fit this one demand script” to “fit a small bag of related OD realizations,” which is why a policy that looked strong on `grid4x4.rou.xml` can degrade on held-out hubs. The **RL machinery in §§1–9 is unchanged**; only the SUMO `-r` argument and a few trainer counters change.

---

## 11. Training vs testing

### 11.1 How the runner distinguishes them

`TSCTask.run`:

```text
if model.train_model: trainer.train()
if model.test_model:  trainer.test()   # drop_load=True by default
```

`model.load_model` in YAML is **never read**. `test(drop_load=True)` **does not** call `ag.load_model`. The trailing evaluation uses the **in-memory online nets**. `extras/run_new_metrics.py` is the path that explicitly `load_model`s a checkpoint.

Greedy eval during training: `train_test` → `_run_eval_episode` with `get_action(..., test=True)`.

### 11.2 What is off in evaluation

| | Training decision | Eval / `test()` |
|--|-------------------|-----------------|
| Action | warmup: uniform; else ε-greedy on **online** net | `test=True` **skips** the ε branch → `argmax Q_online` |
| Replay `remember` | yes | **no** |
| `train()` / RMSprop | after warmup, every decision | **no** |
| Target net | used only inside `train()` | **not used** for acting |
| ε decay | on `train()` | does not decay |
| SUMO + reward generators | yes (reward still computed; used for logs) | yes |
| Metrics | ATT, throughput, queue, delay, mean reward, q_loss | same except q_loss is logged as `100` placeholder in `writeLog("TEST")` |

Code-accurate contrast:

```text
TRAINING (after warmup)
  s, phase → ε-greedy(online Q) → hold 10 s in SUMO
        → r = mean_t(-Σ waiting) → remember → minibatch TD update on online Q
        → every 10 decisions: θ⁻ ← θ

TESTING
  s, phase → argmax_a online Q(s,a) → hold 10 s in SUMO
        → metrics from completed trips / queues
        → no remember, no backward, no target
```

### 11.3 Metrics from SUMO

| Logged name | Definition in this repo |
|-------------|-------------------------|
| real avg travel time | mean of `(arrive_time − depart_time)` over **arrived** vehicles (`World.get_vehicles`) |
| throughput | `len(World.vehicles)` = number **arrived** |
| queue | mean over decisions and agents of `sum(lane_waiting_count)` on incoming lanes (`RLAgent.get_queue`) |
| delay (`--delay_type apx`) | mean of per-agent `lane_delay` = \(1 - \bar v / v_{\lim}\) on incoming lanes |
| rewards | sum over intersections of the DQN reward, averaged over decisions |

Ground-truth ATT/throughput do **not** go through the 200 m detector or partial-obs corruption.

### 11.4 Saved vs evaluated weights (easy to miss)

`save_model` writes **`target_model.state_dict()`**, not the online net. Acting always uses **`self.model`**. After the last decision of a 72 000-decision run, the target is **one decision-sync behind** (last copy at 71 999). The default trailing `test()` uses online weights; a later `load_model(200)` would load the slightly lagged target into **both** nets.

---

## 12. Code-level architecture

```text
run.py  Runner
   │  build_config(configs/tsc/dqn.yml + base.yml)
   │  Registry: command, world, trainer, model, logger
   ▼
task/task.py  TSCTask.run
   ▼
trainer/tsc_trainer.py  TSCTrainer
   ├─ create_world  → world/world_sumo.py  World
   ├─ create_agents → agent/dqn.py         DQNAgent × 32
   ├─ create_metrics→ common/metrics.py    Metrics
   └─ create_env    → environment.py       TSCEnv
         │
         ├─ step/reset
         ▼
      World.step / reset
         ├─ Intersection.pseudo_step  → libsumo/traci trafficlight.setPhase
         ├─ simulationStep
         ├─ Intersection.observe      → full_observation[lane]
         └─ _update_infos             → lane_count, lane_waiting_count, …
                │
                ▼
         generator/lane_vehicle.py          LaneVehicleGenerator
         generator/intersection_phase.py    IntersectionPhaseGenerator
                │
                ▼
         DQNAgent.get_ob / get_reward / get_phase / get_action / remember / train
                │
                ▼
         DQNNet (online, target) + RMSprop + deque replay
```

| Component | File | Responsibility | Called by | Returns |
|-----------|------|----------------|-----------|---------|
| `Runner` | `run.py` | CLI, config registry, construct task | `__main__` | — |
| `TSCTask.run` | `task/task.py` | train then test flags | `Runner.run` | — |
| `BaseTrainer.create` | `trainer/base_trainer.py` | seed, device, create_* | ctor | — |
| `TSCTrainer.train` | `trainer/tsc_trainer.py` | episode/decision loop, remember, train, eval | `TSCTask` | — |
| `TSCTrainer.create_agents` | same | `num_agent = n_TL / sub_agents` | `create` | `self.agents` |
| `TSCEnv.step` | `environment.py` | `world.step`, gather obs/reward, `done=False` | trainer | obs, rewards, dones, {} |
| `World` | `world/world_sumo.py` | SUMO process, infos, reset, demand swap | env / trainer | — |
| `Intersection` | same file | phases, `pseudo_step`, `observe` | `World` | — |
| `DQNAgent` | `agent/dqn.py` | local DQN, replay, ε | trainer | actions, loss |
| `DQNNet` | `agent/dqn.py` | 20-20-`|A|` MLP | agent | Q-values |
| `LaneVehicleGenerator` | `generator/lane_vehicle.py` | vector or scalar from `world.get_info` | agent | `ndarray` |
| `IntersectionPhaseGenerator` | `generator/intersection_phase.py` | `virtual_phase` | agent | `[phase]` |
| `Metrics` | `common/metrics.py` | accumulate queue/delay/reward; ATT via world | trainer | scalars |
| `OnFlyDataset` | `dataset/onfly_dataset.py` | LMDB (unused; flush commented out) | trainer ctor | — |

**Call chain for one trained decision (after warmup):**

`TSCTrainer.train` → `DQNAgent.get_phase` → `DQNAgent.get_action` → `DQNNet.forward(train=False)` → `TSCEnv.step` × 10 → `World.step` → `Intersection.pseudo_step` → `setPhase` + `simulationStep` + `observe` → `DQNAgent.get_reward` / `get_ob` → `DQNAgent.remember` → `DQNAgent.train` → `target_model.forward` + `model.forward` + `RMSprop.step`.

---

## 13. Worked example: one interior intersection, then ×16 / ×32

Take **A0** (rank 0) during a post-warmup decision. Numbers are illustrative in magnitude but the **shapes and formulas are live/code-true**.

### 13.1 Suppose

```text
s_t lane counts (12,) ≈ [3, 1, 0,  4, 2, 0,  5, 1, 0,  2, 2, 1]
current commanded phase = 0          # NT+ST-style green (index 0 in tll.xml)
ε = 0.01                             # already at min after ~460 updates
Q_online(s_t, ·) ≈ [2.1, 0.4, 1.3, 0.8, 1.9, 0.2, 0.5, 1.0]
```

### 13.2 Action selection

`get_action(ob, phase, test=False)`: `np.random.rand() <= 0.01`? Usually no. Concatenate 12 counts with `[1,0,0,0,0,0,0,0]` → `(1, 20)`. Forward online net, `argmax` → **action 0** (stay) or **4** if that Q were larger, etc. If the random draw hits, `sample()` returns `randint(0, 8)` as shape `(1,)`.

The other 31 agents do this **with their own nets** on **their own** `(1, L)` vectors. Fringe always selects 0.

### 13.3 SUMO execution (10 s)

`actions.flatten()` length 32, index 0 is A0’s integer. For each of 10 seconds:

1. `A0.pseudo_step(a)` → `setPhase("A0", …)` as in §4.4 (after t≥30 s this is `setPhase("A0", a)` immediately).
2. `simulationStep()` moves every vehicle in the 4×4, not just A0.
3. `observe` rebuilds A0’s 200 m lane counts / waiting counts.
4. Reward snapshot \(r^{(u)} = -\sum_{\ell \in \text{in}(A0)} q_\ell^{(u)}\).

Neighbors’ lights (B0, A1, …) are changing in the same ticks; they affect A0’s next counts **through the simulator**, not through A0’s observation vector.

### 13.4 Reward, transition, replay

```text
r = mean_{u=1..10} r^{(u)}     # e.g. -18.4
s_{t+1} = lane counts at second 10
phase_{t+1} = virtual_phase after those steps  # commanded a, even during clearance

A0.replay_buffer.append(
  ("e_d_A0", (s_t, phase_t, a, r, s_{t+1}, phase_{t+1}))
)
```

31 other deques get their own tuples. **No concatenation into a joint buffer.**

### 13.5 Minibatch, target, loss, backward

`random.sample(A0.replay_buffer, 64)` — 64 past A0-only experiences, possibly from older episodes.

```text
y_j = r_j + 0.95 * max_{a'} Q_target(s'_j, a')
L   = mean over j and over 8 action heads of (Q_online(s_j, ·) - y_scattered)^2
backward → clip 5 → RMSprop on A0.model only
```

A0’s target net updates only when the **global** decision counter hits `…9`. B0’s target is a different tensor, copied on the same schedule but from B0’s online net.

### 13.6 Across all 16 interior (and 16 fringe) intersections

The same 10 SUMO seconds are **shared physics**. Learning is **not** shared:

```text
for each of 32 TLS:
    own obs → own ε-greedy → own action int
joint World.step(32 ints)
for each of 32 TLS:
    own reward → own deque
for each of 32 TLS:
    own 64-sample TD update on own MLP
```

The 16 interior policies are what one would colloquially call “the Grid4×4 DQN controller.” The 16 portal policies are extra independent learners on 1-D almost-constant actions.

---

## 14. Final synthesis

1. **What is optimized?**  
   For each TLS ID, the parameters \(\theta\) of a local MLP \(Q_\theta(x, a)\) so that \(Q\) matches one-step TD targets \(r + 0.95 \max_{a'} Q_{\theta^-}(x', a')\), with \(x =\) (incoming 200 m lane counts \(\Vert\) one-hot commanded phase) and \(r =\) (interval-mean of −incoming waiting counts, ×12). No joint action-value, no neighbor encoder, no ATT in the loss.

2. **What is one training sample?**  
   One intersection’s `(obs, phase, action, mean_reward, next_obs, next_phase)` tuple in **that** agent’s deque.

3. **How often is a sample generated?**  
   Every **10 SUMO seconds**, **per TLS** (32 per decision; 11 520 per default episode).

4. **How often are gradients applied?**  
   After global decision 1000: **once per decision per agent** (`update_model_rate=1`). Default run: **71 000** RMSprop steps **per** interior net.

5. **How many DQN policies on Grid4×4?**  
   **32 independent networks.** **16** nontrivial 8-action intersection policies + **16** 1-action portal policies. Not one shared DQN. Not 16.

6. **What can one intersection’s policy see?**  
   Only its incoming lanes’ detector counts (≤200 m) and its own commanded phase one-hot. Not outgoing queues, not neighbor TLs, not waiting times as features (waiting is **reward only**).

7. **How does an action affect SUMO?**  
   The integer indexes a green state in the rewritten TLS program; `trafficlight.setPhase` is called every simulation second. On this net, after the first 30 s of an episode, a change of integer typically becomes an **immediate** green change (see §4.4). Vehicles then move under SUMO car-following for 10 s.

8. **What survives between episodes?**  
   Online net, target net, optimizer state, ε, replay, global decision counter, Python RNGs. SUMO is torn down and restarted; demand file is reused (or rotated under odh).

9. **What changes at evaluation?**  
   `test=True` disables ε; no replay writes; no backward; target net unused for acting; default `test()` does not reload `.pt` files; ATT/throughput/queue/delay are logged from SUMO.

10. **What is the artifact called a “trained DQN controller”?**  
    On disk: **32 files** `{episodes}_{rank}.pt`, each the **target** MLP for one TLS, plus logger metrics. In RAM at the end of `train()`: **32 online MLPs** (what `test()` actually rolls out). Operationally, the controller is a **decentralized set of 16 interior greedy argmax-Q policies** (plus 16 unused/trivial portal nets) that, every 10 s, map local lane counts + phase to one of 8 greens on the Grid4×4 interior.

---

## Appendix A — Implementation quirks worth remembering

These are **verified in code**, not speculation:

| Quirk | Where | Effect on the mental model |
|-------|--------|----------------------------|
| `done` never True and not stored | `environment.py`, `remember` | Always bootstraps at episode boundaries |
| MSE over full Q-vector | `DQNAgent.train` | Loss/grad scaled by \(1/\|A\|\) |
| Save target, act online | `save_model` / `get_action` | Checkpoint ≠ last acting weights |
| `* 12` reward | `get_reward` | Interior: −Σ waiting; fringe: −12× one lane |
| Sticky `waiting_times` | `Intersection.observe` | Reward ≠ instantaneous stopped count |
| 200 m cap | `max_distance` | Tail of 273–286 m approaches unseen |
| `yellow_phase_time = 30` interior | min original 30 s greens | First 30 s of each episode: later actions ignored; then instant switches |
| 32nd–16th agents are portals | `getIDList()` | 16 extra DQNs train on 1-action lights |
| `OnFlyDataset.flush` commented out | `tsc_trainer.py` | Replay is RAM-only |
| `get_action_prob` always `None` | `BaseAgent` | Trainer still calls it; unused |
| `vehicle_max` unused | `DQNAgent` | Counts are raw |
| First `get_ob` after reset may use old generator objects | `env.reset` then `ag.reset` | Lane IDs match, so usually harmless |

## Appendix B — Default numeric cheat sheet

```text
Network            sumo4x4 / grid4x4.net.xml
TLS / DQN agents   32 (16 interior 8-phase, 16 fringe 1-phase)
Episode            3600 SUMO seconds
Decision           every 10 s  →  360 / episode
Warmup             1000 decisions (~2.8 episodes), uniform random
Replay             5000 per agent, uniform, not shared
Batch / γ / lr     64 / 0.95 / 0.001 RMSprop
ε                  0.1 × 0.995^k → 0.01  (k ≈ 460 trains)
Target copy        every 10 decisions after warmup
Episodes           200
Interior net       x ∈ R^{20} → 20 → 20 → 8
Reward             r = −Σ incoming waiting_count, averaged over 10 s
Demand             same grid4x4.rou.xml (1473 vehicles) every episode
Stop               fixed episode count, then greedy test()
Artifact           model/{e}_{rank}.pt × 32  (target weights)
```
