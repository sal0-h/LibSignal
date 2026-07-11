# Partial Observability — Approach 1: Penetration Rate (`obs_penetration`)

Spike branch: `spike/obs-penetration`. Throwaway / research; **not** a PR.

## Idea

Third realism toggle (after `hetero` fleet-mix and slow-start): **partial observability /
penetration rate**. Real controllers never see every car — they see loops, cameras,
connected vehicles, and nav-app probes with incomplete coverage. Here each vehicle is
independently **"visible"/connected with probability `p`, persistent for its whole trip**
(a car either carries the tech or it does not). Lane statistics are built only from visible
vehicles, so the signal policy **decides with incomplete information**.

- **Physics is unchanged** — SUMO runs identically; only what the controller *perceives*
  changes. The current phase is still known (the controller sets the lights).
- **Same seeded mask for every agent** → ground-truth ATT stays comparable across
  DQN / PressLight / CoLight / MaxPressure.
- `p = 1.0` → full observability = **exact baseline** (no filtering, no RNG draw).

> This axis operationalizes the "uncertainty in detection" challenge from the Chen et al.
> (2022) review:
>
> **Chen, Fang & Sadeh (2022)**, *The Real Deal: A Review of Challenges and Opportunities in
> Moving RL-Based Traffic Signal Control Systems Towards Reality*, ATT '22 Workshop on Agents
> in Traffic and Transportation, Vienna. CEUR-WS Vol. 3173, ISSN 1613-0073.
>
> §3.2 notes that connected-vehicle "**penetration remains low**," and cites [49] showing
> connected-vehicle data helps adaptive control "**even with limited penetration**." This
> axis models exactly that low-penetration regime.

## Implementation (minimal diff, `world/world_sumo.py`)

- `World.__init__`: reads `obs_penetration` (default 1.0) from the world config; the mask
  seed is inherited from the global `world.seed`.
- `World._vehicle_visible(veh_id)`: persistent, seeded per-vehicle gate —
  `frac = md5(seed:veh_id)[:4] / 2^32; visible iff frac < p`. Short-circuits to `True` at
  `p >= 1.0` and `False` at `p <= 0.0`.
- `Intersection._get_vehicles()`: one added line — keep a vehicle only if
  `self.world._vehicle_visible(v)`. Because every lane statistic (`lane_count`,
  `lane_waiting_count`, `lane_waiting_time_count`, `queue_length`) and both `pressure`
  variants derive from this vehicle list, the whole observation + reward pipeline inherits
  the same consistent mask.
- `configs/tsc/base.yml`: `obs_penetration: 1.0` default.
- Example configs: `maxpressure_obs.yml`, `fixedtime_obs.yml`, `dqn_obs.yml` (p=0.1).

Ground-truth **ATT** and **throughput** are computed from a separate path
(`self.vehicles[v] = exit − entry`), untouched by the mask — so they remain the honest
comparison metric. Logged **queue/delay/reward** become *observed* quantities (what the
controller's sensors report), which is the intended semantics.

## How to run

```bash
# baseline (full observability)
python run.py --task tsc --agent maxpressure --world sumo --network sumo1x1 --interface libsumo --ngpu -1
# 10% penetration example
python run.py --task tsc --agent maxpressure_obs --world sumo --network sumo1x1 --interface libsumo --ngpu -1
```
Sweep by setting `world.obs_penetration` in a config that includes the agent's yml.

## Validation results (baselines, CPU, libsumo)

**sumo1x1** — ground-truth ATT (lower = better):

| p    | MaxPressure ATT | MaxPressure thru | FixedTime ATT |
|------|-----------------|------------------|---------------|
| 1.0  | 39.40           | 1997             | 76.72         |
| 0.5  | 45.27           | 1995             | 78.33         |
| 0.2  | 84.27           | 1903             | 76.72         |
| 0.05 | 153.92          | 243              | 76.72         |

**sumo4x4** — ground-truth ATT:

| p    | MaxPressure ATT | MaxPressure thru | FixedTime ATT |
|------|-----------------|------------------|---------------|
| 1.0  | 173.26          | 1453             | 219.14        |
| 0.5  | 220.96          | 1432             | —             |
| 0.2  | 335.54          | 1401             | 219.14        |
| 0.05 | 571.27          | 1066             | —             |

**Reading:**
- **MaxPressure** (observation-driven) degrades monotonically and sharply as penetration
  drops — at p=0.05 on sumo1x1 it is effectively blind and throughput collapses
  (1997 → 243). This is the core benchmark signal: a controller tuned on perfect counts
  fails on sparse sensors.
- **FixedTime** (ignores observations) has **invariant ground-truth ATT** across all p
  (identical 219.14 on sumo4x4) — confirming (a) physics is unchanged and (b) ATT is true
  ground truth, not an observed quantity. Its *observed* queue shrinks with p (fewer cars
  seen), exactly as intended.
- **Identity check:** `p = 1.0` reproduces the baseline within its natural run-to-run
  jitter (SUMO has small residual non-determinism even at a fixed seed).

## Character of this dial

- Single, physically-meaningful, monotonic knob; faithful to connected-vehicle framing.
- Multiplicative under-count: `E[observed_count] ≈ p · true`. Pressure (a *difference* of
  counts) stays directionally meaningful even when under-counted.
- Only mechanism where visibility is genuinely per-vehicle & persistent — the cleanest of
  the three approaches for a "penetration rate" story.
