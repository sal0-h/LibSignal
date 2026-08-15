#!/usr/bin/env python3
"""Apply explicit Doha patches onto the frozen base net.

  doha_corniche.base.net.xml  (never overwritten)
    + extras/doha_force_joins.nod.xml
    + extras/doha_connections.con.xml   (if non-empty)
    + extras/doha_tls.tll.xml           (if non-empty)
    → doha_corniche.net.xml

Stock netconvert only. No join heuristics.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NET_DIR = REPO / "data" / "raw_data" / "doha_corniche"
BASE = NET_DIR / "doha_corniche.base.net.xml"
OUT = NET_DIR / "doha_corniche.net.xml"
NOD = REPO / "extras" / "doha_force_joins.nod.xml"
CON = REPO / "extras" / "doha_connections.con.xml"
TLL = REPO / "extras" / "doha_tls.tll.xml"


def _netconvert() -> str:
    home = os.environ.get("SUMO_HOME")
    if home:
        cand = Path(home) / "bin" / "netconvert"
        if cand.exists():
            return str(cand)
    raise SystemExit("SUMO_HOME/bin/netconvert not found")


def _has_patch(path: Path, root_tag: str) -> bool:
    if not path.exists() or path.stat().st_size < 20:
        return False
    text = path.read_text()
    return f"<{root_tag}" in text and "</" in text


def _drop_allred(net_path: Path) -> int:
    """Same filter as extras/build_doha_network.sanitize_allred_phases."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(net_path)
    root = tree.getroot()
    removed = 0
    for logic in root.findall("tlLogic"):
        for phase in list(logic.findall("phase")):
            state = phase.get("state") or ""
            if state and set(state) <= {"r"}:
                logic.remove(phase)
                removed += 1
    if removed:
        tree.write(net_path, encoding="UTF-8", xml_declaration=True)
    return removed


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f"missing frozen base {BASE}")
    if BASE.resolve() == OUT.resolve():
        raise SystemExit("refusing to use the same file as base and output")
    cmd = [
        _netconvert(),
        "--sumo-net-file",
        str(BASE),
        "--output-file",
        str(OUT),
        "--tls.yellow.time",
        "3",
        "--tls.allred.time",
        "0",
    ]
    if _has_patch(NOD, "nodes"):
        cmd += ["--node-files", str(NOD)]
    if _has_patch(CON, "connections"):
        cmd += ["--connection-files", str(CON)]
    if _has_patch(TLL, "tlLogics"):
        cmd += ["--tllogic-files", str(TLL)]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = NET_DIR / "_build" / "doha_patches.netconvert.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text((proc.stderr or "") + (proc.stdout or ""))
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"netconvert failed (exit {proc.returncode}). See {log}")
    removed = _drop_allred(OUT)
    if removed:
        print(f"removed {removed} all-red TLS phase(s)")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
