# Demand-bag experiment (Grid4x4, 1200 s)

## The problem

Default LibSignal Grid4x4 training replays the **same demand file every episode** — same vehicles, departures, routes. The world is **stationary**.

Under that protocol, RL often looks like it “converges”:
- train ATT drops early
- the curve flats
- the run looks successful

But the agent may mostly be **memorizing one traffic movie**, not learning a controller that works under different demand. The usual every-episode “test” is also on that **same** file (greedy policy), so it cannot detect memorization. That is an **easy, misleading scoreboard** — not classic label leakage, but in-distribution eval on the training movie.

For a realism-oriented benchmark, that is a protocol failure mode: strong train curves do not imply transferable TSC skill.

## What this experiment does

Keep network, action interval, and episode length fixed. Change only **training demand variety**, then score on **unseen demand**.

| Setting | Value |
|---------|--------|
| Network | `sumo4x4` (Grid4x4) |
| Episode length | **1200 s** (shorter than the usual 3600 s hour) |
| Agent (main) | IDQN (`dqn_*`) |
| Seeds | 42 and 43 |
| Held-out | `hold_00…02` every 10 episodes, **no learning** |

Shorter episodes are mainly **efficiency**: more episodes (and more demand variety) per wall-clock time. They do **not** by themselves stop overfitting. ATT from 1200 s runs must **not** be compared to old 3600 s numbers.

### Two training conditions (same held-out tests)

| Arm | Config | Train demand | Intent |
|-----|--------|--------------|--------|
| **Fixed** (easy baseline) | `dqn_fixed1200` | same `fixed_1200.rou.xml` every episode | usual “one movie” protocol |
| **Bag** (harder) | `dqn_bag1200` | rotate `train_00…09.rou.xml` | force learning across varied demand |

**Bag** here means a folder of route files, not bootstrap ensembles. Learning updates happen **only** on train files. Held-out files are eval-only (locked test set). Same 1200 s length for train and held-out — what changes is the **traffic pattern**, not the clock length.

### Reference jobs (optional / separate)

`maxpressure_1200` / `fixedtime_1200`: no RL training; score once on the same held-out files. MaxPressure is the serious non-memorizing bar; FixedTime is a weak floor.

## Why the runs are meaningful

1. **Fixed vs bag on held-out** — does demand variety during training improve generalization?
2. **Train vs held-out gap** — large gap on fixed ⇒ memorization; smaller gap on bag ⇒ more genuine learning.
3. **vs MaxPressure on held-out** — is any of this useful control, or only better than a weak movie-fit?

This supports a realism-composed benchmarking story: published-style Grid4x4 wins can be artifacts of a stationary demand protocol; a fairer scoreboard reports **held-out demand ATT** (and the train−held-out gap).

## What we expect

| Signal | Likely if memorization is real | Likely if learning is already solid |
|--------|--------------------------------|-------------------------------------|
| Fixed **train** ATT | Looks good, flats early | Also fine |
| Fixed **held-out** ATT | Weaker / flatter; big train−held-out gap | Close to train |
| Bag **train** ATT | Noisier, may look worse than fixed | Similar |
| Bag **held-out** ATT | **Improves and beats fixed** (success) | ≈ fixed |
| vs MaxPressure | Competitive or better on held-out strengthens the claim; losing to MP can still support a **protocol critique** | Same bar |

**Main plot:** `HELDOUT_MEAN` ATT vs episode (fixed vs bag, both seeds).  
**Secondary:** train ATT (for the gap) and MaxPressure as a flat held-out line.

Success for the paper narrative: bag helps on held-out; fixed looks strong on train but weaker off the training movie.

## Jobs (2 seeds)

| Array task | Run |
|------------|-----|
| 0 | `dqn_fixed1200` seed 42 |
| 1 | `dqn_fixed1200` seed 43 |
| 2 | `dqn_bag1200` seed 42 |
| 3 | `dqn_bag1200` seed 43 |

Submit example:

```bash
sbatch --array=0,2 --mcs-label="${MCS_LABEL}" extras/slurm_demand_bag_1200.sh   # seed 42
sbatch --array=1,3 --mcs-label="${MCS_LABEL}" extras/slurm_demand_bag_1200.sh   # seed 43
```

Operational details (generate demand files, smoke test, log paths): see `extras/DEMAND_BAG_1200.md`.
