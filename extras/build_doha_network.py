#!/usr/bin/env python3
"""
Reproducible OSM → SUMO conversion for the Doha Corniche benchmark.

Downloads a current OpenStreetMap extract (not committed) and converts it with
official SUMO netconvert options. Writes:

  data/raw_data/doha_corniche/doha_corniche.net.xml
  data/raw_data/doha_corniche/SOURCE.json
  data/raw_data/doha_corniche/_build/   (gitignored working files)

Usage:
    python extras/build_doha_network.py
    python extras/build_doha_network.py --skip-download
    python extras/build_doha_network.py --download-only
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BBOX_FILE = REPO / "extras" / "doha_bbox.yml"
OUT_DIR = REPO / "data" / "raw_data" / "doha_corniche"
BUILD_DIR = OUT_DIR / "_build"
PREFIX = "doha_corniche"
NET_OUT = OUT_DIR / f"{PREFIX}.net.xml"
SOURCE_OUT = OUT_DIR / "SOURCE.json"
WARN_LOG = BUILD_DIR / "netconvert_warnings.txt"


def _sumo_home() -> Path:
    env = os.environ.get("SUMO_HOME")
    if env:
        p = Path(env)
        if (p / "tools" / "osmGet.py").exists():
            return p
    # eclipse-sumo wheel (pip) layout
    for candidate in (
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "sumo",
        Path.home() / ".local" / "lib" / "python3.13" / "site-packages" / "sumo",
        Path.home() / ".conda" / "envs" / "traffic" / "lib" / "python3.10" / "site-packages" / "sumo",
    ):
        if (candidate / "tools" / "osmGet.py").exists():
            return candidate
    raise SystemExit(
        "Cannot find SUMO_HOME with tools/osmGet.py. "
        "Install eclipse-sumo and export SUMO_HOME."
    )


def _which(name: str, sumo_home: Path) -> str:
    candidate = sumo_home / "bin" / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"Cannot find binary {name!r}. Set SUMO_HOME or PATH.")


def _load_bbox():
    data = {}
    note_lines = []
    in_note = False
    for raw in BBOX_FILE.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if in_note:
            if line.startswith(" ") or line.startswith("\t") or not line.strip():
                if line.strip():
                    note_lines.append(line.strip())
                continue
            in_note = False
        if not line.strip():
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if key in ("west", "south", "east", "north"):
            data[key] = float(val)
        elif key == "area_note":
            in_note = True
            if val and val != ">":
                note_lines.append(val)
    data["area_note"] = " ".join(note_lines)
    bbox = f"{data['west']},{data['south']},{data['east']},{data['north']}"
    return data, bbox


def download_osm(osmget: str, bbox: str, osm_path: Path) -> list[str]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        osmget,
        "--bbox", bbox,
        "--prefix", PREFIX,
        "--output-dir", str(BUILD_DIR),
        "--retries", "6",
        "--retry-delay", "20",
        "--verbose",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))
    # osmGet.py --bbox writes <prefix>_bbox.osm.xml
    produced = BUILD_DIR / f"{PREFIX}_bbox.osm.xml"
    if not produced.exists():
        produced = BUILD_DIR / f"{PREFIX}.osm.xml"
    if not produced.exists():
        matches = list(BUILD_DIR.glob("*.osm.xml"))
        if not matches:
            raise SystemExit("osmGet.py finished but no .osm.xml was written")
        produced = matches[0]
    if produced.resolve() != osm_path.resolve():
        shutil.copy2(produced, osm_path)
    return cmd


def netconvert_cmd(netconvert: str, osm_path: Path, net_tmp: Path, typemap: Path, bbox: str) -> list[str]:
    """Official recommended OSM import options, vehicle-centric.

    Recommended baseline (SUMO docs, Networks/Import/OpenStreetMap):
      --geometry.remove --ramps.guess --junctions.join
      --tls.guess-signals --tls.discard-simple --tls.join

    Extra, documented here because this is a TSC benchmark:
      passenger-only edges, clip to the chosen geo bbox so OSM ways that
      continue far outside the download window are not kept, drop isolated
      fragments, keep the largest weakly-connected component, slack for
      unsigned service arms, static TLS (LibSignal replaces programs),
      fixed yellow=3s so LibSignal's min(duration) yellow heuristic is not
      1s all-red.
    """
    return [
        netconvert,
        "--osm-files", str(osm_path),
        "--output-file", str(net_tmp),
        "--type-files", str(typemap),
        "--geometry.remove", "true",
        "--ramps.guess", "true",
        "--junctions.join", "true",
        "--tls.guess-signals", "true",
        "--tls.guess-signals.slack", "1",
        "--tls.discard-simple", "true",
        "--tls.join", "true",
        "--tls.default-type", "static",
        "--tls.yellow.time", "3",
        "--tls.allred.time", "0",
        "--keep-edges.by-vclass", "passenger",
        "--keep-edges.in-geo-boundary", bbox,
        "--remove-edges.isolated", "true",
        "--keep-edges.components", "1",
        "--remove-edges.by-type",
        "highway.track,highway.services,highway.path,highway.footway,"
        "highway.cycleway,highway.steps,highway.pedestrian,highway.bridleway,"
        "highway.corridor,highway.bus_guideway",
        "--output.original-names", "true",
        "--output.street-names", "true",
        "--proj.utm", "true",
    ]


def convert(netconvert: str, osm_path: Path, typemap: Path, bbox: str) -> tuple[list[str], str]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    net_tmp = BUILD_DIR / f"{PREFIX}.net.xml"
    cmd = netconvert_cmd(netconvert, osm_path, net_tmp, typemap, bbox)
    cfg = BUILD_DIR / f"{PREFIX}.netccfg"
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd + ["--save-configuration", str(cfg)], check=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    WARN_LOG.write_text(proc.stderr + ("\n" if proc.stdout else "") + proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(
            f"netconvert failed (exit {proc.returncode}). See {WARN_LOG}\n"
            f"{proc.stderr[-4000:]}"
        )
    sanitize_allred_phases(net_tmp)
    shutil.copy2(net_tmp, NET_OUT)
    return cmd, proc.stderr


def sanitize_allred_phases(net_path: Path) -> int:
    """Drop all-red phases from imported TLS programs.

    LibSignal sets yellow_phase_time = min(original phase durations). A 1s
    all-red leftover would make every generated yellow 1s. Removing all-red
    states does not change LibSignal's green-phase action set (those phases
    are already discarded by generate_valid_phase). This is a reproducible
    XML filter, not a hand edit of geometry.
    """
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
        print(f"removed {removed} all-red TLS phase(s) from {net_path.name}")
    return removed


def categorize_warnings(stderr: str) -> dict:
    counts = {}
    samples = {}
    for line in stderr.splitlines():
        line = line.strip()
        if not line:
            continue
        key = "other"
        low = line.lower()
        if "warning:" in low:
            if "discarding unusable type" in low or "discarding unknown compound" in low:
                key = "discarding_unusable_type"
            elif "referenced geometry" in low:
                key = "unknown_geometry_ref"
            elif "only 1 node" in low:
                key = "way_one_node"
            elif "restriction relation" in low:
                key = "restriction_relation"
            elif "could not be determined" in low:
                key = "restriction_direction"
            elif "joining" in low or "join" in low:
                key = "junction_or_tls_join"
            elif "not joined" in low:
                key = "not_joined"
            elif "speed" in low:
                key = "speed"
            elif "ramp" in low:
                key = "ramps"
            elif "traffic light" in low or "tls" in low:
                key = "tls"
            elif "connection" in low or "conflict" in low:
                key = "connections"
            elif "edge" in low and "removed" in low:
                key = "edges_removed"
            elif "shap" in low or "geometry" in low:
                key = "geometry"
            elif "permission" in low or "vclass" in low:
                key = "permissions"
            elif "roundabout" in low:
                key = "roundabout"
            elif "ptstop" in low or "public transport" in low:
                key = "pt"
            else:
                key = "warning_other"
        elif "error:" in low:
            key = "error"
        else:
            continue
        counts[key] = counts.get(key, 0) + 1
        samples.setdefault(key, [])
        if len(samples[key]) < 5:
            samples[key].append(line)
    return {"counts": counts, "samples": samples}


def _portable(cmd, sumo_home: Path):
    out = []
    home = str(sumo_home)
    repo = str(REPO)
    for tok in cmd:
        if tok.startswith(home):
            tok = "$SUMO_HOME" + tok[len(home):]
        elif tok.startswith(repo + os.sep):
            tok = tok[len(repo) + 1:]
        out.append(tok)
    return out


def write_source(bbox_data, bbox, sumo_home, sumo_ver, osm_cmd, nconv_cmd, warn_cat, osm_path: Path):
    if osm_cmd == ["<reused existing OSM extract>"]:
        osm_cmd_out = [
            "python",
            "$SUMO_HOME/tools/osmGet.py",
            "--bbox", bbox,
            "--prefix", PREFIX,
            "--output-dir", "data/raw_data/doha_corniche/_build",
            "--retries", "6",
            "--retry-delay", "20",
            "--verbose",
        ]
    else:
        osm_cmd_out = _portable(osm_cmd, sumo_home)
    payload = {
        "network": "doha_corniche",
        "cli_name": "sumo_doha",
        "acquisition_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "bbox_west_south_east_north": bbox,
        "bbox": {
            "west": bbox_data["west"],
            "south": bbox_data["south"],
            "east": bbox_data["east"],
            "north": bbox_data["north"],
            "note": bbox_data.get("area_note", "").strip(),
        },
        "sumo_version": sumo_ver,
        "sumo_home": "$SUMO_HOME",
        "typemap": "data/typemap/osmNetconvert.typ.xml (default motor-vehicle OSM map; not UrbanDe — Doha arterials keep OSM maxspeed)",
        "vehicle_filter": "keep-edges.by-vclass passenger",
        "osm_file_not_committed": "data/raw_data/doha_corniche/_build/doha_corniche.osm.xml",
        "osm_bytes": osm_path.stat().st_size if osm_path.exists() else None,
        "osmGet_command": osm_cmd_out,
        "netconvert_command": _portable(nconv_cmd, sumo_home),
        "netconvert_warning_categories": warn_cat,
        "references": {
            "sumo_osm_import": "https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html",
            "qarsumo_context_only": "Chen et al., QarSUMO, ACM SIGSPATIAL 2020, arXiv:2010.03289",
        },
    }
    SOURCE_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {SOURCE_OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    sumo_home = _sumo_home()
    os.environ["SUMO_HOME"] = str(sumo_home)
    osmget = str(sumo_home / "tools" / "osmGet.py")
    netconvert = _which("netconvert", sumo_home)
    typemap = sumo_home / "data" / "typemap" / "osmNetconvert.typ.xml"
    if not typemap.exists():
        raise SystemExit(f"missing typemap {typemap}")

    ver = subprocess.check_output([netconvert, "--version"], text=True).splitlines()[0]
    bbox_data, bbox = _load_bbox()
    osm_path = BUILD_DIR / f"{PREFIX}.osm.xml"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    osm_cmd = []
    if args.skip_download:
        if not osm_path.exists():
            raise SystemExit(f"--skip-download but {osm_path} is missing")
    else:
        if args.force_download or not osm_path.exists():
            osm_cmd = download_osm(osmget, bbox, osm_path)
        else:
            print(f"reusing {osm_path}")
            osm_cmd = ["<reused existing OSM extract>"]

    if args.download_only:
        print(f"downloaded {osm_path} ({osm_path.stat().st_size} bytes)")
        return

    nconv_cmd, stderr = convert(netconvert, osm_path, typemap, bbox)
    warn_cat = categorize_warnings(stderr)
    write_source(bbox_data, bbox, sumo_home, ver, osm_cmd, nconv_cmd, warn_cat, osm_path)
    print(f"wrote {NET_OUT} ({NET_OUT.stat().st_size} bytes)")
    print("warning categories:", json.dumps(warn_cat["counts"], indent=2))


if __name__ == "__main__":
    main()
