# Partial Observability — Approach 2: Gaussian Count Noise (`obs_count_noise_std`)

Spike branch: `spike/obs-gauss-noise`. Throwaway / research; **not** a PR.

## Idea

Third realism toggle (after `hetero` fleet-mix and slow-start): **partial / noisy
observability**. Real detectors (loops, cameras) miscount vehicles. Here every per-lane
vehicle count the controller reads is corrupted by **Gaussian measurement noise**, so the
signal policy **decides with noisy information** while the traffic itself is unchanged.

- **Physics is unchanged** — SUMO runs identically; only the counts the controller reads
  are perturbed. The phase is still known.
- **Unbiased**: `E[observed] = true`; noise is re-sampled every step (measurement noise
  flickers), rounded, and clamped `>= 0`.
- **Same seeded noise for every agent** → ground-truth ATT stays comparable across
  DQN / PressLight / CoLight / MaxPressure.
- `sigma = 0.0` → exact counts = **baseline** (no draws).

Two modes:
- `additive` (default): `observed = true + N(0, sigma^2)`.
- `proportional`: `observed = true + N(0, (sigma*true)^2)` — noise grows with volume.

This axis operationalizes the "uncertainty in detection" challenge from the Chen et al.
(2022) review, and directly follows the Gaussian-noise method of Tan et al. (2020):

> **Chen, Fang & Sadeh (2022)**, *The Real Deal: A Review of Challenges and Opportunities
> in Moving RL-Based Traffic Signal Control Systems Towards Reality*, ATT '22 Workshop on
> Agents in Traffic and Transportation, Vienna. CEUR-WS Vol. 3173, ISSN 1613-0073. — §3
> ("Uncertainty in detection") surveys detector noise/failure; ref. [59] therein is the
> primary method this axis reproduces.
>
> **Tan, Sharma & Sarkar (2020)**, *Robust Deep Reinforcement Learning for Traffic Signal
> Control*, Journal of Big Data Analytics in Transportation 2(3):263–274.
> doi:10.1007/s42421-020-00029-6. — inject a **discrete zero-mean Gaussian perturbation
> into the queue-length state dimension** (per element, floored at 0) to robustify RL TSC.
> Our `additive` mode is a faithful reproduction of that mechanism.

## Implementation (minimal diff, `world/world_sumo.py`)

- `World.__init__`: reads `obs_count_noise_std` (0.0), `obs_noise_mode` ('additive'); the
  noise seed is inherited from the global `world.seed`.
- `World._apply_obs_count_noise()`: a single per-step post-pass that perturbs the counts
  stored in each intersection's `full_observation` **in place** — `lane_count` and
  `queue_length` share one detector draw (they are the same physical count here),
  `lane_waiting_count` gets its own. Uses a per-step RNG keyed on `(seed, sim_time)`.
- Hooked at the top of `World._update_infos()` (called every step and on reset, before any
  info function reads the observation), so `lane_count`, `queue_length`,
  `lane_waiting_count` and both `pressure` variants inherit the **same** noisy realization
  consistently.
- `configs/tsc/base.yml`: `obs_count_noise_std: 0.0`, `obs_noise_mode: additive`.
- Example configs: `maxpressure_noise.yml`, `fixedtime_noise.yml`, `dqn_noise.yml` (sigma=2).

Ground-truth **ATT/throughput** use a separate path (`self.vehicles`) and are untouched.

## How to run

```bash
python run.py --task tsc --agent maxpressure_noise --world sumo --network sumo1x1 --interface libsumo --ngpu -1
```
Sweep by setting `world.obs_count_noise_std` (and optionally `obs_noise_mode`) in a config
that includes the agent's yml.

## Validation results (baselines, CPU, libsumo)

**sumo1x1** — ground-truth ATT (additive):

| sigma | MaxPressure ATT | MaxPressure thru | FixedTime ATT |
|-------|-----------------|------------------|---------------|
| 0     | 39.40           | 1997             | 76.72         |
| 1     | 40.57           | 1995             | 76.72         |
| 2     | 42.97           | 1999             | 76.72         |
| 5     | 53.88           | 1987             | 78.33         |

**sumo4x4** — ground-truth ATT (additive):

| sigma | MaxPressure ATT | MaxPressure thru |
|-------|-----------------|------------------|
| 0     | 173.26          | 1453             |
| 2     | 200.81          | 1439             |
| 5     | 220.28          | 1435             |

**proportional** mode (sumo1x1 MaxPressure): sigma=0.5 -> 40.21, sigma=1.0 -> 44.52.

**Reading:**
- **MaxPressure** degrades **monotonically but gracefully** with sigma — much gentler than
  the penetration axis, because unbiased noise partly averages out in the pressure
  *difference* (opposing lanes' errors cancel in expectation). This is the expected
  "robust-to-unbiased-noise" behaviour and a useful contrast to penetration's biased
  under-count.
- **FixedTime** ground-truth ATT is **invariant** to sigma (ignores observations),
  confirming physics is unchanged and ATT is true ground truth; its *observed* queue rises
  with sigma (spurious counts).
- **Identity check:** sigma=0 reproduces the baseline exactly.

## Character of this dial

- Clean, isolated **noise** axis (as opposed to *incompleteness*): counts are unbiased, so
  it separates "imprecise sensing" from "missing coverage."
- Additive sigma is scale-dependent (a sigma of 2 means more on a lane that holds 4 cars
  than on one holding 40); `proportional` mode removes that if a scale-free knob is wanted.
- Rounding + clamp at 0 introduces a tiny positive bias only when true counts are near 0.

## Fidelity to Tan et al. (2020) [59]

Matches the source method: discrete zero-mean Gaussian, sampled **per lane**, **floored at
0** ("the true queue length can be reduced to zero at max"), **re-sampled every step**. Our
`additive` mode is their exact mechanism.

Deviations (deliberate, none are bugs):
- **Scope.** [59] perturb the **queue-length dimension only** (leaving phase/speed untouched)
  to isolate the effect. We perturb `lane_count` (= this codebase's `queue_length`, one
  shared draw) **and** `lane_waiting_count` — since MaxPressure/DQN decide on
  `lane_count`→pressure, that is the correct primary analog; the extra `waiting_count` draw
  goes marginally beyond their protocol. Restrict to the agent's count feature if strict
  isolation is wanted.
- **Parameter.** [59] dial a "noise level δ" (the discrete-Gaussian **tail spread**; δ=5,10
  in their experiments); we expose **σ = standard deviation** (δ ≈ a few σ) — the more
  standard parameterization.
- **`proportional` mode is our extension** — [59] use a fixed spread independent of the
  count (≡ our `additive` mode).

Note on LibSignal semantics: the field named `queue_length` here counts **all** vehicles on
the lane (identical to `lane_count`), not stopped vehicles; the stopped/queued count is
`lane_waiting_count`. [59]'s "queue length" means *stopped* vehicles — functionally
irrelevant here because we noise the feature the agent actually consumes.
