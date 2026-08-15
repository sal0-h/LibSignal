# Doha Corniche SUMO benchmark

Real OpenStreetMap road topology for central Doha (Corniche / West Bay), with
**synthetic** demand. This is not a calibrated reproduction of QarSUMO or of
observed Doha traffic.

CLI network name: `--network sumo_doha`  
Data: `data/raw_data/doha_corniche/`

## What is in this folder

| File | Committed? | Role |
|------|------------|------|
| `doha_corniche.base.net.xml` | yes | Frozen stock OSM import. Never patch this file. |
| `doha_corniche.net.xml` | yes | LibSignal benchmark (base + explicit patches) |
| `doha_corniche.rou.xml` | yes | 3600 s synthetic smoke / default demand |
| `doha_corniche.sumocfg` | yes | SUMO-only config |
| `SOURCE.json` | yes | bbox, SUMO version, osmGet/netconvert commands, date |
| `STATS.json` | yes | junction / TLS / lane statistics |
| `od_hubs/` | yes | TAZ + train/hold/fixed demand bags |
| `_build/` | **no** | OSM extract and netconvert working files |

Patches (never mixed into the importer): `extras/doha_force_joins.nod.xml`,
`extras/doha_connections.con.xml`, `extras/doha_tls.tll.xml`.
Leftovers: `extras/doha_patches_PUNCHLIST.md`.

## Regenerate

```bash
export SUMO_HOME=...          # eclipse-sumo tools dir
python extras/build_doha_network.py          # OSM → stock import into .net.xml
# Freeze once (do not overwrite an existing base):
#   cp data/raw_data/doha_corniche/doha_corniche.net.xml \
#      data/raw_data/doha_corniche/doha_corniche.base.net.xml
python extras/apply_doha_patches.py          # base + explicit patches → .net.xml
python extras/doha_network_stats.py
python extras/validate_sumo_tls.py --net data/raw_data/doha_corniche/doha_corniche.net.xml
python extras/gen_od_hub_demand_doha.py      # TAZ + demand bags + default .rou.xml
```

Bounding box and rationale: `extras/doha_bbox.yml`.

## Run

SUMO only:

```bash
sumo -c data/raw_data/doha_corniche/doha_corniche.sumocfg --duration-log.statistics
```

LibSignal (FixedTime / MaxPressure):

```bash
python run.py -a fixedtime -w sumo -n sumo_doha --seed 42 --ngpu -1 --prefix doha_ft
python run.py -a maxpressure -w sumo -n sumo_doha --seed 42 --ngpu -1 --prefix doha_mp
```

OD-hub bags (1800 s, 10 train / 3 holdout):

```bash
python run.py -a maxpressure_odh_l1_doha -w sumo -n sumo_doha --seed 42 --ngpu -1 --prefix doha_odh
python run.py -a fixedtime_odh_l1_doha -w sumo -n sumo_doha --seed 42 --ngpu -1 --prefix doha_odh
```

## Notes

- Vehicle-only import (`--keep-edges.by-vclass passenger`). Pedestrian/bike ways are dropped.
- Traffic lights come from OSM tags plus netconvert `--tls.guess-signals` / `--tls.join`.
  Several controllers are joined clusters (`joinedS_*`, `GS_cluster_*`).
- Action counts are **not** uniform (typically 1–8 greens). Independent agents
  (FixedTime, MaxPressure, DQN, PressLight) are fine. CoLight uses
  `Discrete(max n)` without masking.
- Demand is synthetic (gravity OD + Corniche hub). Do not treat ATT as a field measurement.
