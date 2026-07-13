# Demand-bag 1200s experiment (`feat/demand-bag-1200`)

Check whether IDQN “learns” a single Grid4x4 traffic movie, and whether a
rotating demand bag improves held-out ATT.

## Arms

| Config | Episode | Train demand | Eval |
|--------|---------|--------------|------|
| `dqn_fixed1200` | 1200 s | `fixed_1200.rou.xml` every reset | held-out every 10 ep |
| `dqn_bag1200` | 1200 s | `train_00..09` rotate | held-out every 10 ep |
| `maxpressure_1200` / `fixedtime_1200` | 1200 s | — | held-out once (flat refs) |

PressLight twins: `presslight_fixed1200`, `presslight_bag1200`.  
CoLight twins: `colight_fixed1200`, `colight_bag1200`.

Pilot defaults: **100 episodes**, **2 seeds** (42, 43), **3 held-out files**.
Bump `trainer.episodes` to 200 and add seed 44 when the gap looks real.

PressLight + CoLight (4 jobs, seed 42 only):

```bash
sbatch --mcs-label="${MCS_LABEL}" extras/slurm_demand_bag_1200_pl_colight.sh
# 0 PL fixed | 1 PL bag | 2 CL fixed | 3 CL bag
```

## One-time: generate demand files

```bash
python extras/gen_demand_bag_grid4x4.py
# -> data/raw_data/grid4x4/demand_bag/{fixed_1200,train_*,hold_*}.rou.xml
```

## Local smoke (1–2 min)

```bash
source .venv/bin/activate
python extras/gen_demand_bag_grid4x4.py
# temporarily set episodes: 2 in demand_bag_1200_base.yml if you want a quick check
python run.py -a dqn_bag1200 -w sumo -n sumo4x4 --seed 42 --ngpu -1 --prefix smoke_bag
```

Look for `HELDOUT_MEAN` lines in the BRF/DTL logs.

## Remote (gpujobs)

```bash
# on Mac
git push -u origin feat/demand-bag-1200

# on server
cd ~/LibSignalFork
git fetch && git checkout feat/demand-bag-1200
python extras/gen_demand_bag_grid4x4.py   # if demand_bag/ not committed yet
mkdir -p logs
export MCS_LABEL=crs-XXXX
sbatch --mcs-label="${MCS_LABEL}" extras/slurm_demand_bag_1200.sh

# PressLight instead of DQN for indices 0-3:
# PRESSLIGHT=1 sbatch --mcs-label="${MCS_LABEL}" extras/slurm_demand_bag_1200.sh
```

## What to plot

From `*_DTL.log`:

- `TRAIN` ATT vs episode (fixed often flats early)
- `HELDOUT_MEAN` ATT vs episode (**main verdict**)
- Final train ATT − held-out ATT (gap ⇒ memorization)

Success: bag improves held-out; fixed looks strong on train, weaker on held-out.
