# Extras — analysis tools (not part of core LibSignal)

Everything here is **self-contained**: scripts, SLURM job, outputs under `extras/output/`.  
No changes to `run.py`, `world/`, `agent/`, or `configs/` are required.

## The 100s cap problem

SUMO's `getAccumulatedWaitingTime()` is capped by default (`--waiting-time-memory=100`).  
That is why plots of `accumulated_waiting_time_s` flatten at 100s.

**Fix (already in `run_vehicle_wait_logs.py`):** per-step `custom_wait_s` — increments each  
second when `speed < 0.1 m/s`. **No cap.** Use this column in your notebook, not  
`accumulated_waiting_time_s`.

Verify after a run:

```bash
head -1 extras/output/maxpressure/sumo7x28/seed42_steps3600/vehicle_waiting_times.csv
# must include: custom_wait_s,trip_status,...
```

End of run should print:

```
custom_wait_s (all vehicles): mean=... p95=... max=...
```

If `max` is **above 100**, the uncapped metric is working.

---

## 1. Copy updated extras to server (from your Mac)

```bash
cd ~/Desktop/LibSignalFork
./extras/copy_to_server.sh
```

---

## 2. Run via SLURM on gpujobs (from login node)

**Do not** run 7×28 on the login node or on your Mac.

`deepnet` is a **compute server name**, not a SLURM partition — do **not** use `#SBATCH --partition=deepnet`.

```bash
ssh mmirzata@172.20.48.59
cd ~/LibSignalFork
mkdir -p logs

export MCS_LABEL=crs-XXXX        # your course label (required)
export CONDA_ENV=traffic
export NETWORK=sumo7x28
export TEST_STEPS=3600
export RUN_NAME=seed42_steps3600

sbatch --mcs-label="${MCS_LABEL}" extras/slurm_vehicle_wait_logs.sh
```

If submission still fails, check valid partitions and time limits:

```bash
sinfo
```

Copy the updated script from your Mac first if you haven't since the fix:

```bash
./extras/copy_to_server.sh
```

Monitor:

```bash
squeue -u $USER
tail -f logs/veh_wait_log_<jobid>.out
```

---

## 3. Download results to Mac

```bash
cd ~/Desktop/LibSignalFork
./extras/download_from_server.sh
```

---

## 4. Jupyter notebook cells (use `custom_wait_s`)

Paste into your notebook (`result_analysis/results.ipynb` or similar):

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV = Path("../extras/output/maxpressure/sumo7x28/seed42_steps3600/vehicle_waiting_times.csv")
df = pd.read_csv(CSV)

# REQUIRED: uncapped metric (not accumulated_waiting_time_s)
METRIC = "custom_wait_s"
assert METRIC in df.columns, "Re-run with updated extras/run_vehicle_wait_logs.py on server"

w = df[METRIC]
print(f"n={len(df)}  max={w.max():.0f}s  p95={w.quantile(0.95):.0f}s  p99={w.quantile(0.99):.0f}s")
print(df.groupby("trip_status")[METRIC].agg(["count", "median", "max"]))
```

```python
# Outlier / long-tail figure
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

waits = np.sort(df[METRIC].to_numpy())
n = len(waits)
p99 = np.percentile(waits, 99)

axes[0].hist(waits, bins=80, color="#4C72B0", edgecolor="white")
axes[0].axvline(100, color="gray", linestyle=":", label="SUMO cap (old metric)")
axes[0].set_xlabel(f"{METRIC} (s)")
axes[0].set_title("Histogram (all vehicles)")
axes[0].legend()

axes[1].plot(np.sort(waits), np.arange(1, n + 1) / n, color="#4C72B0")
axes[1].set_xlabel(f"{METRIC} (s)")
axes[1].set_title("CDF")

ranked = np.sort(waits)[::-1]
colors = np.where(ranked >= p99, "#C44E52", "#4C72B0")
axes[2].scatter(np.arange(1, n + 1), ranked, s=6, c=colors, alpha=0.6)
axes[2].axhline(p99, color="#C44E52", linestyle="--", label=f"p99={p99:.0f}s")
axes[2].set_xlabel("rank (worst first)")
axes[2].set_ylabel(f"{METRIC} (s)")
axes[2].set_title("Rank plot — red = top 1%")
axes[2].legend()

plt.tight_layout()
plt.show()
```

```python
# Worst-off vehicles (stranded at sim end often dominate the tail)
stranded = df[df["trip_status"] == "on_map_at_end"].nlargest(20, METRIC)
stranded[["vehicle_id", "travel_time_s", METRIC, "delay_s", "trip_status"]]
```

---

## Output layout

```
extras/output/<agent>/<network>/<run_name>/
  vehicle_waiting_times.csv
  vehicle_waiting_times_meta.json
```

`extras/output/` is gitignored.
