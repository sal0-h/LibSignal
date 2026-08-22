# MaxPressure alignment with Varaiya (2013)

Issue [#11](https://github.com/sal0-h/libsignal/issues/11) asked whether this repo’s
MaxPressure agent matches the **original** method. Short answer: **the previous default
did not**; the default is now a practical Varaiya-aligned controller, with an optional
legacy LibSignal mode.

## Original method (Varaiya 2013)

At each intersection, for every turn movement \((l, m)\):

\[
w(l,m) = x(l,m) - \sum_{p \in \mathrm{Out}(m)} r(m,p)\, x(m,p)
\]

Pressure of a stage (compatible set of movements) \(S\):

\[
\gamma(S) = \sum_{(l,m):\,S(l,m)=1} c(l,m)\, w(l,m)
\]

Actuate \(S^\star = \arg\max_S \gamma(S)\).  
\(x\) are **per-movement queues**, \(r\) turn ratios, \(c\) saturation flows. Exit links
contribute \(x(m,\cdot)=0\) on the downstream side. Demand is not required.

Reference: P. Varaiya, *Max pressure control of a network of signalized intersections*,
Transportation Research Part C, 2013.

## What the old code did (LibSignal heuristic)

```python
pressure = sum(lane_count[start] - lane_count[end]
               for start, end in phase_available_lanelinks[phase])
```

| Aspect | Varaiya 2013 | Old LibSignal code |
|--------|--------------|--------------------|
| State metric | Queue length \(x(l,m)\) | Total `lane_count` (moving + stopped) |
| Downstream | Expected queue on link \(m\) via turn ratios | Specific end-lane vehicle count |
| Exit links | Downstream weight \(= 0\) | Still subtracted end-lane counts |
| Saturation \(c(l,m)\) | Multiplies each weight | Implicitly \(1\) for all |
| Min green | Optional variant | `t_min` (kept) |

Docs previously called MaxPressure “queue-based,” but `get_action` used `lane_count`.
Upstream DaRL LibSignal ships the same heuristic; it is a useful baseline in RL-TSC
papers, not a faithful Original-MP.

## Current default (`mp_variant: varaiya`)

`agent/maxpressure.py` now defaults to:

1. **Queues** — `lane_waiting_count` (speed &lt; 0.1 m/s) as the proxy for \(x\).
2. **Exit-aware downstream** — if the receiving road is not an approach to any
   signalized intersection, downstream contribution is **0** (matches isolated
   intersections: pressure = sum of upstream queues).
3. **Link-level downstream** — on internal links, subtract the **total** waiting count
   on the receiving road (proxy for \(\sum_p x(m,p)\) when turn ratios are unavailable).
4. **Equal saturation** — `sat_flow: 1.0` unless overridden in the agent YAML.
5. **`t_min`** — minimum green (Varaiya’s minimum-green modification).

### Remaining approximations

- No per-movement queues when a lane serves multiple turns (shared lanes over-count).
- No online turn ratios \(r(m,p)\); downstream uses total link queue, not
  \(\sum_p r(m,p)x(m,p)\).
- Uniform \(c\) unless you set `sat_flow`.
- Decisions every `action_interval` with `t_min`, not a pure every-period store-and-forward step.

## Legacy mode

To reproduce the pre-fix LibSignal baseline:

```yaml
# configs/tsc/maxpressure.yml (or a copy)
model:
  name: maxpressure
  t_min: 10
  mp_variant: libsignal   # lane_count in−out heuristic
```

Default (omit or set explicitly):

```yaml
model:
  name: maxpressure
  t_min: 10
  mp_variant: varaiya
  sat_flow: 1.0
```

## Smoke test

```bash
source .venv/bin/activate
python run.py --agent maxpressure --world sumo --network sumo1x1 --seed 42 --ngpu -1
```
