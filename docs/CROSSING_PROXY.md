# Crossing Proxy — Actuated Pedestrian-Crossing Delay (Axis 5)

Fifth realism axis for the realism-decomposed RL-TSC benchmark. Models **capacity
loss from pedestrian crossing service** at signalized intersections without explicit
SUMO pedestrian entities.

See also: `docs/PARTIAL_OBSERVABILITY.md` (axis 4), `results.ipynb` slow-start notes
(axis 2), `data/raw_data/vTypes_mixed.add.xml` (axis 3).

---

## What this axis models (and what it does not)

**Models (traffic-side effect of actuated crossings):**

- Stochastic **ped calls** on some through-green cycles (not every cycle).
- **Service time** during which conflicting vehicle approaches cannot discharge
  (walk interval + clearance / FDW), consuming effective green time.
- **NEMA concurrent walk** semantics: during N–S through green, the proxy blocks
  **E–W incoming** lanes (and vice versa).

**Does not model:**

- Pedestrian agents, sidewalks, spatial crossing, gap acceptance, Barnes dance, or
  separate ped phases in the TLS state string.
- Pedestrian delay metrics (primary metric remains **vehicle ATT**).

**Paper label:** *actuated crossing-delay proxy* — not “full pedestrian simulation.”

---

## Why not full SUMO pedestrians?

SUMO supports full multimodal simulation (`<person>`, `walkingArea`, `crossing`,
ped links in TLS state strings). That path is valid for **ped-aware TSC** papers but
is a poor fit for this benchmark because:

| Issue | Impact on factorial benchmark |
|-------|-------------------------------|
| `grid4x4.net.xml` has **no** ped infrastructure | Requires net rebuild / new artifact |
| TLS state strings would grow ped links | Breaks fixed 8-phase vehicle action space |
| Person demand + routes | New calibration layer, not a YAML toggle |
| LibSignal obs are **vehicle lane counts** | Agents stay blind unless obs pipeline changes |

The proxy targets the same **mechanism** actuated-ped analysis uses in practice:
random extensions of effective red / green consumption on conflicting movements
(e.g. Cheng et al., TRB 2008 on actuated ped delay), while keeping **one net, eight
phases, one metric column** comparable to hetero / slow_start / partial obs.

Full SUMO peds remain a future **multimodal extension** (`docs/TECHNICAL_ANALYSIS.md`
§E), not this axis.

---

## Mechanism

1. **Static lane map** (`data/raw_data/grid4x4/crossing_proxy_lanes.json`), generated once
   by `extras/gen_crossing_proxy_lanes.py` from the net + `phase_pairs` in
   `configs/tsc/mplight.yml`.
2. On **entry** to an eligible through-green phase (indices **0** and **4** on grid4x4):
   - With probability `crossing_call_prob`, start a crossing event (max one active per TLS).
   - Sample service duration `T ~ Uniform(service_min, service_max)` seconds.
   - Set `lane.setMaxSpeed(lane, 0)` on pre-mapped **conflicting incoming** lanes.
3. After `T` sim-seconds (wall clock, may span phase change — models FDW clearance),
   restore original lane max speeds.
4. Per-TLS RNG: `Random(md5(f"{seed}:{tls_id}") + crossing_seed_offset)` so demand
   seed 42 reproduces the same call sequence across agents.

**Physics change:** yes (vehicles actually queue). **Observation change:** indirect
(lane counts / pressure reflect halts). Same as real crossings from the controller’s
view when ped counts are not instrumented.

---

## Default parameters (calibrated / literature-anchored)

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `crossing_call_prob` | **0.12** | Urban actuated calls on ~10–15% of through greens (sparse–moderate corners) |
| `crossing_service_min` | **7.0 s** | ~12 m crosswalk at MUTCD 4 ft/s (≈4 s walk) + minimum clearance |
| `crossing_service_max` | **10.0 s** | Walk + typical FDW / clearance window for short urban crosswalk |
| `crossing_seed_offset` | **7** | Isolates crossing RNG from SUMO demand + partial-obs streams |

Tune `crossing_call_prob` after a MaxPressure smoke run so homo MP ATT (~173 s on
corrected 4×4) rises modestly (+3–8%), not catastrophically.

---

## Configuration

```yaml
# configs/tsc/base.yml (defaults — exact baseline when false)
world:
  crossing_proxy: false
  crossing_call_prob: 0.12
  crossing_service_min: 7.0
  crossing_service_max: 10.0
  crossing_seed_offset: 7
```

```json
// configs/sim/sumo4x4.cfg
"crossingProxyLanes": "raw_data/grid4x4/crossing_proxy_lanes.json"
```

Agent bundles (same pattern as `*_slow_start.yml`, `*_hetero.yml`):

- `configs/tsc/maxpressure_crossing_proxy.yml`
- `configs/tsc/fixedtime_crossing_proxy.yml`
- `configs/tsc/dqn_crossing_proxy.yml`
- `configs/tsc/presslight_crossing_proxy.yml`
- `configs/tsc/colight_crossing_proxy.yml`

---

## How to run

Regenerate lane map after net / phase layout changes:

```bash
export SUMO_HOME="${SUMO_HOME:-/opt/homebrew/opt/sumo/share/sumo}"  # adjust if needed
python extras/gen_crossing_proxy_lanes.py --network grid4x4
```

Smoke test (single episode, CPU):

```bash
python run.py -a maxpressure_crossing_proxy -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix crossing_proxy_smoke
```

Full benchmark row (200 episodes, match other axes):

```bash
# Baselines (BRF only — no training loop)
python run.py -a maxpressure_crossing_proxy -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix baseline_crossing_proxy

python run.py -a fixedtime_crossing_proxy -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix baseline_crossing_proxy

# RL (200 ep default)
python run.py -a dqn_crossing_proxy -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix crossing_proxy_4x4

python run.py -a presslight_crossing_proxy -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix crossing_proxy_4x4

python run.py -a colight_crossing_proxy -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix crossing_proxy_4x4
```

Logger paths follow the usual convention:

`data/output_data/tsc/sumo_{agent}/sumo4x4/{prefix}/logger/`

---

## Composability

| Axis | Composes with crossing_proxy? |
|------|-------------------------------|
| hetero | Yes |
| slow_start | Yes (hetero ⊥ slow_start rule unchanged) |
| obs_penetration / obs_count_noise_std | Yes |
| homo control | `crossing_proxy: false` |

---

## Files touched

| File | Role |
|------|------|
| `world/world_sumo.py` | `CrossingProxyController`, hook in `step_sim` / `reset` |
| `configs/tsc/base.yml` | Default params |
| `configs/sim/sumo4x4.cfg` | Lane map path |
| `data/raw_data/grid4x4/crossing_proxy_lanes.json` | Per-TLS phase → lane lists |
| `extras/gen_crossing_proxy_lanes.py` | Regenerate lane map |
| `configs/tsc/*_crossing_proxy.yml` | Per-agent enable flag |

---

## Expected effects

- **ATT:** increase vs homo (vehicles wait through service intervals).
- **Throughput:** may decrease under saturation (same blind spot as slow-start on large nets).
- **FixedTime:** degrades (physics); does not benefit from obs-only invariance.
- **MaxPressure / RL:** see halts via queue/pressure; no explicit ped channel.

Compare homo baselines (MP **173.26 s**, FT **219.14 s** on corrected 4×4) before drawing
ranking conclusions.

---

## Verification checklist

1. Log line at startup: `[CrossingProxy] enabled — p=0.12, service=[7.0,10.0]s, TLS=32, ...`
2. Homo run with `crossing_proxy: false` — bit-identical to pre-branch baseline.
3. MP crossing_proxy ATT > homo MP ATT (after tuning if needed).
4. FT crossing_proxy ATT > homo FT ATT; FT pen-only ATT unchanged vs homo (sanity).
