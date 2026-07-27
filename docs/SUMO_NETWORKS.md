# SUMO Networks Available in LibSignal

All available SUMO networks with their paths, road network topology, and traffic flow files.

## Selected network (issue #34)

Upcoming experiments use **`--network sumo1x21`** — the real Ingolstadt arterial
(`data/raw_data/ingolstadt21/`), **21** traffic lights. This replaces the synthetic
`sumo4x4` grid as the default multi-intersection map for homo / no-realism runs
(follow-up: issue #35).

```bash
python run.py --agent maxpressure --world sumo --network sumo1x21 --seed 42 --ngpu -1
```

## Quick Reference

| Network Name | Type | Intersections | Config File | Road Network | Traffic Flow |
|---|---|---|---|---|---|
| sumo1x1 | Real (Cologne) | 1 | [sumo1x1.cfg](../configs/sim/sumo1x1.cfg) | `data/raw_data/cologne1/cologne1.net.xml` | `cologne1.rou.xml` |
| sumo1x3 / sumo_cologne3 | Real corridor (Cologne) | 3 | [sumo1x3.cfg](../configs/sim/sumo1x3.cfg) | `data/raw_data/cologne3/cologne3.net.xml` | `cologne3.rou.xml` |
| **sumo1x21** | **Real arterial (Ingolstadt)** | **21** | [sumo1x21.cfg](../configs/sim/sumo1x21.cfg) | **`data/raw_data/ingolstadt21/ingolstadt21.net.xml`** | **`ingolstadt21.rou.xml`** |
| sumo4x4 | Synthetic grid | 32 TLs | [sumo4x4.cfg](../configs/sim/sumo4x4.cfg) | `data/raw_data/grid4x4/grid4x4.net.xml` | `grid4x4.rou.xml` |
| sumo7x28 | Synthetic large grid (NY) | 196 | [sumo7x28.cfg](../configs/sim/sumo7x28.cfg) | `data/raw_data/manhattan_28x7/manhattan_28x7.net.xml` | `manhattan_28x7.rou.xml` |
| sumohz1x1 | Hangzhou 1×1 | 1 | [sumohz1x1.cfg](../configs/sim/sumohz1x1.cfg) | `hangzhou_1x1_bc-tyc_18041610_1h/` | matching `.rou.xml` |
| sumohz4x4 | Hangzhou Gudang 4×4 | 16 | [sumohz4x4.cfg](../configs/sim/sumohz4x4.cfg) | `hangzhou_4x4_gudang_18041610_1h/` | matching `.rou.xml` |

> Note: older docs incorrectly listed `sumo1x21` as `arterial1x6` and `sumo7x28` as
> `cologne7x28`. The `.cfg` files point at **ingolstadt21** and **manhattan_28x7**.

---

## Network Details

### Synthetic Grid Networks

#### **sumo4x4**
- **Topology**: Urban grid 4×4 interior (`A0`–`D3`) plus 16 fringe portal TLs
  (`top`/`bottom`/`left`/`right` `0–3`) — **32 traffic lights total**, so LibSignal
  creates **32 agents** (one per SUMO TL ID). See [AGENT_OBSERVATIONS.md](AGENT_OBSERVATIONS.md).
- **Use case**: Medium-scale multi-agent coordination (previous default)
- **Path**:
  - Config: `configs/sim/sumo4x4.cfg`
  - Network: `data/raw_data/grid4x4/grid4x4.net.xml`
  - Flow: `data/raw_data/grid4x4/grid4x4.rou.xml`

#### **sumo7x28**
- **Topology**: Large synthetic Manhattan-style grid **28×7 = 196** intersections
  (`data/raw_data/manhattan_28x7/`)
- **Use case**: Large-scale / scalability experiments (slow on CPU)
- **Path**:
  - Config: `configs/sim/sumo7x28.cfg`
  - Network: `data/raw_data/manhattan_28x7/manhattan_28x7.net.xml`
  - Flow: `data/raw_data/manhattan_28x7/manhattan_28x7.rou.xml`

---

### Real-World Networks

#### **sumo1x1** (Cologne1)
- Single real Cologne intersection — smoke / unit tests
- Path: `data/raw_data/cologne1/`

#### **sumo1x3** / **sumo_cologne3**
- Real Cologne **3-intersection corridor**
- Path: `data/raw_data/cologne3/`

#### **sumo1x21** (Ingolstadt21) — **selected**
- Real Ingolstadt arterial — **21 TLs**, irregular geometry, trip-based demand
- Config: `configs/sim/sumo1x21.cfg`
- Network: `data/raw_data/ingolstadt21/ingolstadt21.net.xml`
- Flow: `data/raw_data/ingolstadt21/ingolstadt21.rou.xml`
- Smoke-tested with MaxPressure, fixedtime, SOTL, DQN, PressLight, PPO (`--ngpu -1`)
- MPLight/FRAP need a `signal_config` entry before use (not wired yet)

#### **sumohz1x1** / **sumohz4x4**
- Hangzhou real demand (1×1 and Gudang 4×4). See configs under `configs/sim/sumohz*.cfg`.

---

## How to Run

```bash
source .venv/bin/activate   # Cursor Cloud; or conda activate traffic on lab servers

# Selected map
python run.py --agent maxpressure --world sumo --network sumo1x21 --seed 42 --ngpu -1 --prefix salman_ingolstadt

# Previous default (synthetic grid)
python run.py --agent maxpressure --world sumo --network sumo4x4 --seed 42 --ngpu -1
```

Outputs land in:

`data/output_data/tsc/sumo_<agent>/sumo1x21/<prefix>/`

(same layout Madina used, e.g. `madina_4x4`, `madina_cologne3`).

---

## Network Selection Guide

| Goal | Recommended Network |
|---|---|
| **Default multi-intersection experiments** | **`sumo1x21`** |
| Quick smoke test | `sumo1x1` |
| Small real corridor | `sumo_cologne3` |
| Synthetic grid (legacy) | `sumo4x4` |
| Scalability stress | `sumo7x28` |
