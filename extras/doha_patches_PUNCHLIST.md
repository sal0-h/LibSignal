# Doha explicit-patch punch list (leftovers — not silently flattened)

Pipeline: `doha_corniche.base.net.xml` + `extras/doha_force_joins.nod.xml` +
`extras/doha_connections.con.xml` + `extras/doha_tls.tll.xml` → `doha_corniche.net.xml`.
Rebuild: `python extras/apply_doha_patches.py`.

## Done (this pass)

- Al Bidda × Markhiya: 4-corner join, slips `4818928605` / `4818928608` stay separate priority merges
- C-Ring × Ras Abu Aboud: 4-corner join, slips `6935427567` / `6935427560` stay outside
- Rayyan × Jassim: 4-corner join, `10768755344` stays outside
- Omar Al Mukhtar × Conference Centre: 2-node join, slip `2428034575` stays outside
- Markhiya × Aneza: 2-node join; ramp cluster `cluster_4816587471_*` not in the join
- Rumaila: `joinedS_589…` gone; four stop-line TLS with explicit `tl` / `linkIndex`
- `extras/validate_sumo_tls.py` PASS (31 TLS, min phase 3 s)
- Demand edges: 0 missing after regen
- FixedTime (`--prefix doha_patch_ft`): ATT 490 s, completion 86.5%
- MaxPressure (`--prefix doha_patch_mp`): ATT 549 s, completion 83.5%

## West Bay graph traces (node `5579258852` not joined)

Ugly 2D is accepted. Test is connectivity, not z.

- Omar surface `12691031#0` → `12691031#4` stays on Omar through `5579258852`
- Southbound trunk `521217266` → `12690821` is Lusail continuing as Aneza (OSM name change on the same carriageway)
- Northbound trunk `535241553` → `12691202` is Aneza continuing as Lusail
- Real ramp `1038098549` → `1038098548` → Omar `12691031#4`
- Direct Omar `12691031#3` → Lusail `406068650` has **no route** (XY overlap is not a graph link)

## Leftovers (stop here)

- `GS_*` guessed lights (including West Bay `GS_5078731267`) were not geometrically audited
- `1240229136` and `96444930` remain Discrete(1) diverge fixtures
- Rumaila’s four controllers are Discrete(1) through-signals (correct split, not pretty 4-ways)
- West Bay is still an ugly 2D stack; that is accepted
- Lusail/Aneza OSM names swap on the grade-separated trunk; not flattened
- Unsigned junctions and roundabouts were not rewritten
- Other OSM `cluster_*` / `GS_cluster_*` TLS (not in the explicit join file) were left as imported
- Joined 4-ways may warn `Intersecting left turns` (wide junction radius)
- Pedestrian OSM crossings used as vehicle TLS were not stripped
