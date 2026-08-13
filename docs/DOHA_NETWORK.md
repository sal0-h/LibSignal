# Doha Corniche real-network benchmark

A first-class SUMO/LibSignal dataset: **real central-Doha topology** from current
OpenStreetMap, plus **synthetic** demand. It is not an attempt to reproduce the
QarSUMO Corniche extract (Chen et al., ACM SIGSPATIAL 2020, [arXiv:2010.03289](https://arxiv.org/abs/2010.03289));
that paper is cited only as precedent for using this geography.

CLI name: `--network sumo_doha`  
Files: [`data/raw_data/doha_corniche/`](../data/raw_data/doha_corniche/README.md)

## Bounding box

WGS84 west, south, east, north (see [`extras/doha_bbox.yml`](../extras/doha_bbox.yml)):

```
51.508, 25.278, 51.550, 25.338
```

Chosen as a compact, contiguous Corniche corridor (~4.2 × 6.7 km) covering West Bay /
Al Dafna, Al Corniche Street, Al Bidda, Msheireb / central Doha, and Old Doha / port
approaches. The Pearl, Lusail, Education City, and the airport are out of scope.

## Regeneration

Requires official SUMO tools (`osmGet.py`, `netconvert`, `od2trips`, `duarouter`).
Recorded conversion used Eclipse SUMO **1.27.0**.

```bash
python extras/build_doha_network.py                 # OSM download + netconvert
python extras/doha_network_stats.py                 # STATS.json
python extras/validate_sumo_tls.py \
  --net data/raw_data/doha_corniche/doha_corniche.net.xml
python extras/gen_od_hub_demand_doha.py             # TAZ + demand bags
```

The OSM XML is **not** committed (`data/raw_data/doha_corniche/_build/`). `SOURCE.json`
stores the bbox, acquisition date, SUMO version, and exact commands.

netconvert follows the [official OSM import recommendations](https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html)
(`--geometry.remove --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join`)
plus a vehicle-only filter, geo-boundary clip, largest-component keep, and a
reproducible drop of all-red TLS phases so LibSignal's `min(duration)` yellow
heuristic is not 1 s.

## LibSignal compatibility

| Topic | Behaviour |
|-------|-----------|
| Agents | One agent per SUMO TLS id (`trafficlight.getIDList()`), including joined controllers. |
| Action space | Per-intersection `Discrete(n_green)`. Counts differ across Doha junctions. |
| Lane ids | `world_sumo.sumo_edge_id_from_lane` / `sumo_lane_index` (`rsplit('_', 1)`). Same as the old `[:-2]` rule on grid4x4 (single-digit indices). |
| Empty greens | World construction **raises** if a TLS has no green phase after the yellow/all-red filter. |
| MaxPressure | Still maps only `G` and `s` onto lanelinks (unchanged). Permissive `g` movements are ignored, as on Cologne / Ingolstadt. |
| CoLight | Shared `Discrete(max n)` with **no** action mask. Heterogeneous Doha phase counts can yield invalid actions. Do not treat CoLight as supported on this map without further work. |
| Yellow timing | Unchanged legacy rule: `yellow_phase_time = min(original phase durations)`. All-red states are stripped from this net so the min is the 3 s yellow. |

Validate TLS programs:

```bash
python extras/validate_sumo_tls.py --net data/raw_data/doha_corniche/doha_corniche.net.xml
python extras/validate_sumo_tls.py --net data/raw_data/grid4x4/grid4x4.net.xml
python extras/test_lane_id_parse_grid4x4.py
```

## Demand

Synthetic only. TAZ: fringe (N/E/S/W), internal 2×2, hub = OSM name `شارع الكورنيش`
(Al Corniche). Sampling matches the grid4x4 OD-hub generator (gravity, ~65% fringe
origins, hub-heavy destinations, shoulder+peak timeline). Routing: `od2trips` +
`duarouter`. Short trips (<3 edges or <400 m) are dropped.

## Run

```bash
sumo -c data/raw_data/doha_corniche/doha_corniche.sumocfg --duration-log.statistics

python run.py -a fixedtime    -w sumo -n sumo_doha --seed 42 --ngpu -1 --prefix doha_ft
python run.py -a maxpressure  -w sumo -n sumo_doha --seed 42 --ngpu -1 --prefix doha_mp
python run.py -a maxpressure_odh_l1_doha -w sumo -n sumo_doha --seed 42 --ngpu -1 --prefix doha_odh
```
