# SUMO Networks Available in LibSignal

All available SUMO networks with their paths, road network topology, and traffic flow files.

For a recommendation of which maps are more interesting than the default `sumo4x4`
grid, see [ALTERNATIVE_NETWORKS.md](ALTERNATIVE_NETWORKS.md) (issue #32).

## Quick Reference

| Network Name | Type | TLs | Config File | Road Network | Traffic Flow |
|---|---|---:|---|---|---|
| sumo1x1 | Real (Cologne) | 1 | [sumo1x1.cfg](../configs/sim/sumo1x1.cfg) | `data/raw_data/cologne1/cologne1.net.xml` | `cologne1.rou.xml` |
| sumo1x3 | Real corridor (Cologne) | 3 | [sumo1x3.cfg](../configs/sim/sumo1x3.cfg) | `data/raw_data/cologne3/cologne3.net.xml` | `cologne3.rou.xml` |
| sumo_cologne3 | Real corridor (Cologne) | 3 | [sumo_cologne3.cfg](../configs/sim/sumo_cologne3.cfg) | same as `sumo1x3` | same |
| sumo1x21 | Real arterial (Ingolstadt) | 21 | [sumo1x21.cfg](../configs/sim/sumo1x21.cfg) | `data/raw_data/ingolstadt21/ingolstadt21.net.xml` | `ingolstadt21.rou.xml` |
| sumo4x4 | Synthetic grid | 32 | [sumo4x4.cfg](../configs/sim/sumo4x4.cfg) | `data/raw_data/grid4x4/grid4x4.net.xml` | `grid4x4.rou.xml` |
| sumo4x4_gui | Synthetic grid (GUI) | 32 | [sumo4x4_gui.cfg](../configs/sim/sumo4x4_gui.cfg) | same as `sumo4x4` | same |
| sumo7x28 | Synthetic large grid (NY) | 196 | [sumo7x28.cfg](../configs/sim/sumo7x28.cfg) | `data/raw_data/manhattan_28x7/manhattan_28x7.net.xml` | `manhattan_28x7.rou.xml` |
| sumo_atlanta1x5 | Real arterial stub (Atlanta) | 5 | [sumo_atlanta1x5.cfg](../configs/sim/sumo_atlanta1x5.cfg) | `data/raw_data/atlanta_1x5/atlanta_1x5.net.xml` | `atlanta_1x5.rou.xml` |
| sumo1x1_colight | Cologne 1×1 (legacy paths) | 1 | [sumo1x1_colight.cfg](../configs/sim/sumo1x1_colight.cfg) | cologne1 JSON convert paths | — |
| sumohz1x1 | Hangzhou 1×1 (bc-tyc 10) | 1 | [sumohz1x1.cfg](../configs/sim/sumohz1x1.cfg) | `hangzhou_1x1_bc-tyc_18041610_1h/` | matching `.rou.xml` |
| sumohz1x1_config2 | Hangzhou 1×1 (qc-yn 08) | 1 | [sumohz1x1_config2.cfg](../configs/sim/sumohz1x1_config2.cfg) | `hangzhou_1x1_qc-yn_18041608_1h/` | matching `.rou.xml` |
| sumohz1x1_config3 | Hangzhou 1×1 (kn-hz 08) | 1 | [sumohz1x1_config3.cfg](../configs/sim/sumohz1x1_config3.cfg) | `hangzhou_1x1_kn-hz_18041608_1h/` | matching `.rou.xml` |
| sumohz1x1_config4 | Hangzhou 1×1 (sb-sx 07) | 1 | [sumohz1x1_config4.cfg](../configs/sim/sumohz1x1_config4.cfg) | `hangzhou_1x1_sb-sx_18041607_1h/` | matching `.rou.xml` |
| sumohz4x4 | Hangzhou Gudang 4×4 | 16 | [sumohz4x4.cfg](../configs/sim/sumohz4x4.cfg) | `hangzhou_4x4_gudang_18041610_1h/` | matching `.rou.xml` |
| sumohz4x4_hetero | Hangzhou 4×4 hetero | 16 | [sumohz4x4_hetero.cfg](../configs/sim/sumohz4x4_hetero.cfg) | `hangzhou_4x4_hetero/` | `_m.rou.xml` |

---

## Network Details

### Synthetic Grid Networks

#### **sumo4x4**
- **Topology**: Urban grid 4×4 interior (`A0`–`D3`) plus 16 fringe portal TLs
  (`top`/`bottom`/`left`/`right` `0–3`) — **32 traffic lights total**, so LibSignal
  creates **32 agents** (one per SUMO TL ID). See [AGENT_OBSERVATIONS.md](AGENT_OBSERVATIONS.md).
- **Use case**: Medium-scale multi-agent coordination (default in most recent experiments)
- **Path**:
  - Config: `configs/sim/sumo4x4.cfg`
  - Network: `data/raw_data/grid4x4/grid4x4.net.xml`
  - Flow: `data/raw_data/grid4x4/grid4x4.rou.xml`
- **Run**:
  ```bash
  python run.py --agent dqn --world sumo --network sumo4x4 --seed 42 --ngpu -1
  ```

#### **sumo7x28**
- **Topology**: Large synthetic Manhattan-style grid **28×7 = 196** intersections
  (`data/raw_data/manhattan_28x7/`)
- **Use case**: Large-scale / scalability experiments (slow on CPU)
- **Path**:
  - Config: `configs/sim/sumo7x28.cfg`
  - Network: `data/raw_data/manhattan_28x7/manhattan_28x7.net.xml`
  - Flow: `data/raw_data/manhattan_28x7/manhattan_28x7.rou.xml`

#### **sumo1x1_colight**
- **Topology**: Same Cologne single intersection as `sumo1x1`, with legacy convert-file paths
- **Use case**: Historical CoLight config; prefer `sumo1x1` for normal runs

---

### Real-World Networks (Germany / US)

#### **sumo1x1** (Cologne1)
- **Topology**: Single real Cologne intersection
- **Use case**: Minimal / smoke tests
- **Path**: `data/raw_data/cologne1/`

#### **sumo1x3** / **sumo_cologne3**
- **Topology**: Real Cologne **3-intersection corridor** (not a synthetic 1×3 arterial)
- **Use case**: Small real multi-intersection corridor; coordination / green-wave style tasks
- **Path**: `data/raw_data/cologne3/`
- **Note**: Both configs point at the same map; prefer `sumo_cologne3` for clarity.

#### **sumo1x21** (Ingolstadt21)
- **Topology**: Real Ingolstadt arterial — **21 TLs**, irregular geometry, ~853 edges,
  heterogeneous phase counts
- **Use case**: Best “interesting” alternative to `sumo4x4` for topology realism
- **Path**: `data/raw_data/ingolstadt21/`
- **Demand**: trip-based (`<trip>` entries in `.rou.xml`)

#### **sumo_atlanta1x5**
- **Topology**: Atlanta arterial stub — **5 TLs**, heterogeneous phase plans (2/4/8)
- **Use case**: Short real corridor; stress-test mixed action-space sizes
- **Path**: `data/raw_data/atlanta_1x5/`

---

### Real-World Urban Networks (Hangzhou, China)

#### **sumohz1x1**
- **Topology**: Single Hangzhou intersection (bc-tyc, peak hour)
- **Path**: `data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/`

#### **sumohz1x1_config2, config3, config4**
Different Hangzhou sites / hours (actual `.cfg` targets):
- **config2**: `hangzhou_1x1_qc-yn_18041608_1h`
- **config3**: `hangzhou_1x1_kn-hz_18041608_1h`
- **config4**: `hangzhou_1x1_sb-sx_18041607_1h`

#### **sumohz4x4**
- **Topology**: Hangzhou Gudang **4×4** (16 TLs, regular grid geometry, real demand)
- **Path**: `data/raw_data/hangzhou_4x4_gudang_18041610_1h/`

#### **sumohz4x4_hetero**
- **Topology**: Same area with heterogeneous vehicle types (`hangzhou_4x4_hetero/`)

---

## How to Run Experiments

### Quick Start
```bash
source .venv/bin/activate   # also sets SUMO_HOME on Cursor Cloud

python run.py --agent maxpressure --world sumo --network sumo1x1 --seed 42 --ngpu -1
python run.py --agent maxpressure --world sumo --network sumo_cologne3 --seed 42 --ngpu -1
python run.py --agent maxpressure --world sumo --network sumo1x21 --seed 42 --ngpu -1
```

### SUMO-GUI
```bash
sumo-gui -c data/raw_data/ingolstadt21/ingolstadt21.sumocfg
# or via LibSignal (requires --interface traci):
python run.py --agent fixedtime --world sumo --network sumo4x4_gui --interface traci --ngpu -1
```

### Train DQN on Different Networks
```bash
python run.py --agent dqn --world sumo --network sumo1x1 --seed 42 --ngpu -1
python run.py --agent dqn --world sumo --network sumo4x4 --seed 42 --ngpu -1
python run.py --agent dqn --world sumo --network sumohz1x1 --seed 42 --ngpu -1
python run.py --agent dqn --world sumo --network sumo1x21 --seed 42 --ngpu -1
```

---

## Configuration File Structure

Each network's `.cfg` file specifies:
- **network**: Name/ID of topology (used by some agents' `signal_config`)
- **roadnetFile**: Path to SUMO `.net.xml` (road topology)
- **flowFile**: Path to SUMO `.rou.xml` (traffic routes)
- **combined_file**: Optional `.sumocfg` when present
- **interval**: Simulation step size (typically 1.0 second)
- **yellow_length**: Duration of yellow phase (seconds)
- **gui**: Whether to show SUMO GUI (`true`/`false`; needs `--interface traci`)

Example (`configs/sim/sumo4x4.cfg`):
```json
{
  "network": "grid4x4",
  "interval": 1.0,
  "seed": 0,
  "dir": "data/",
  "roadnetFile": "raw_data/grid4x4/grid4x4.net.xml",
  "flowFile": "raw_data/grid4x4/grid4x4.rou.xml",
  "no_warning": true,
  "name": "debug",
  "yellow_length": 3,
  "gui": false
}
```

---

## Network Selection Guide

| Goal | Recommended Network | Why |
|---|---|---|
| Quick test | `sumo1x1` | Instant runs, single intersection |
| Algorithm dev | `sumo1x1`, `sumo4x4` | Fast iteration, manageable |
| Leave the 4×4 habit | **`sumo1x21`**, `sumo_cologne3` | Real irregular / corridor topology |
| Real demand, grid geometry | `sumohz4x4` | Hangzhou peak-hour flows |
| Heterogeneous phases | `sumo_atlanta1x5`, `sumo1x21` | Mixed phase counts per TL |
| Scalability study | `sumo7x28` | 196 TLs |
| Multi-vehicle types | `sumohz4x4_hetero` | Heterogeneous traffic |

See [ALTERNATIVE_NETWORKS.md](ALTERNATIVE_NETWORKS.md) for the full issue #32 analysis.

---

## Common Issues

**"No SUMO in environment path"**
→ `source .venv/bin/activate` (exports `SUMO_HOME`), or set it manually from the `sumo` package.

**Simulation very slow**
→ Cloud VMs are CPU-only (`--ngpu -1`). Prefer smaller maps (`sumo_cologne3`) before `sumo7x28`.

**Network not found error**
→ Verify `.net.xml` and `.rou.xml` exist; `--network` must match a file in `configs/sim/`.

**GUI does not open with libsumo**
→ Use `--interface traci` when `gui: true`.
