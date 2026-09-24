"""Turn recorded arena results into the good-and-fast scoreboard.

Every arena run reports its result to the service as a `score` event whose `details` is the
full ArenaResult (engine, mode, seed, difficulty, collisions, collisions_by_kind, near_misses,
distance, mean_speed, ticks_met / ticks_missed, think_p50_ms ...). This module reads those
records back and aggregates them per (mode, engine, difficulty):

  good   collisions per run, of which thrown rocks, near-misses, distance, mean speed
  fast   think latency p50 (the engine's own), and in realtime mode the share of decision
         ticks the engine actually met - the Tetris "could not keep up" number

Re-running a combination replaces the earlier result (last write wins), so a matrix can be
repeated after a change and the table reflects the latest build.

    uv run python -m server.scoreboard            # newest telemetry file
    uv run python -m server.scoreboard --all      # every file in data/
    uv run python -m server.scoreboard --json out/scoreboard.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import fmean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEY = ("mode", "engine", "difficulty")


def parse_seeds(spec: str | None) -> tuple[int, int] | None:
    """'7-8' -> (7, 8); '7' -> (7, 7); None -> no filter."""
    if not spec:
        return None
    a, _, b = spec.partition("-")
    return int(a), int(b or a)


def load_results(paths: list[str], seeds: tuple[int, int] | None = None, seconds: float | None = None) -> list[dict]:
    """Arena results from telemetry `score` events, deduplicated on (mode, engine, difficulty, seed), last wins.

    `seeds` (inclusive range) and `seconds` restrict the table to one evaluation matrix, so
    collection runs recorded in the same files (other seeds, longer runs) do not dilute it."""
    latest: dict[tuple, dict] = {}
    for path in sorted(paths):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") != "event":
                    continue
                ev = r.get("event") or {}
                d = ev.get("details") or {}
                if ev.get("type") != "score" or "mode" not in d or "seed" not in d:
                    continue  # play-mode episode ends carry reason/episode instead
                if seeds is not None and not (seeds[0] <= int(d["seed"]) <= seeds[1]):
                    continue
                if seconds is not None and float(d.get("seconds", 0)) != seconds:
                    continue
                d = dict(d)
                d["ts"] = r.get("ts")
                d["file"] = os.path.basename(path)
                latest[(d["mode"], d["engine"], int(d.get("difficulty", 3)), int(d["seed"]))] = d
    return list(latest.values())


def aggregate(results: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in results:
        groups[(d["mode"], d["engine"], int(d.get("difficulty", 3)))].append(d)
    rows = []
    for (mode, engine, difficulty), rs in sorted(groups.items()):
        ticks_met = sum(r.get("ticks_met", 0) for r in rs)
        ticks_total = ticks_met + sum(r.get("ticks_missed", 0) for r in rs)
        rows.append({
            "mode": mode, "engine": engine, "difficulty": difficulty, "runs": len(rs),
            "seeds": sorted(int(r["seed"]) for r in rs),
            "collisions_mean": fmean(r.get("collisions", 0) for r in rs),
            "rock_hits_mean": fmean((r.get("collisions_by_kind") or {}).get("projectile", 0) for r in rs),
            "near_misses_mean": fmean(r.get("near_misses", 0) for r in rs),
            "distance_mean": fmean(r.get("distance", 0.0) for r in rs),
            "speed_mean": fmean(r.get("mean_speed", 0.0) for r in rs),
            "ticks_met_pct": 100.0 * ticks_met / ticks_total if ticks_total else None,
            "think_p50_ms": fmean(r.get("think_p50_ms", 0.0) for r in rs),
        })
    return rows


def markdown(rows: list[dict]) -> str:
    out = ["| mode | engine | level | runs | collisions/run | rock hits/run | near-misses/run | distance | speed | ticks met | think p50 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        tm = "-" if r["ticks_met_pct"] is None else f"{r['ticks_met_pct']:.0f}%"
        out.append(f"| {r['mode']} | `{r['engine']}` | {r['difficulty']} | {r['runs']} | {r['collisions_mean']:.2f} | "
                   f"{r['rock_hits_mean']:.2f} | {r['near_misses_mean']:.1f} | {r['distance_mean']:.0f} m | "
                   f"{r['speed_mean']:.1f} m/s | {tm} | {r['think_p50_ms']:.0f} ms |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="read every telemetry file, not just the newest")
    ap.add_argument("--files", nargs="*", help="explicit telemetry files")
    ap.add_argument("--seeds", help="only results with seeds in this inclusive range, e.g. 7-8 (the matrix's EVAL_SEEDS)")
    ap.add_argument("--seconds", type=float, help="only results of runs this long, e.g. 45 (the matrix's EVAL_SECONDS)")
    ap.add_argument("--json", help="also write the rows here")
    args = ap.parse_args(argv)
    files = args.files or sorted(glob.glob(os.path.join(ROOT, "data", "telemetry-*.jsonl")), key=os.path.getmtime)
    if not files:
        print("no telemetry files in data/", file=sys.stderr)
        return 2
    if not args.all and not args.files:
        files = files[-1:]
    results = load_results(files, parse_seeds(args.seeds), args.seconds)
    rows = aggregate(results)
    scope = "".join(f" seeds {args.seeds}" if args.seeds else "" for _ in [0]) + (f" {args.seconds:g} s runs" if args.seconds else "")
    print(f"{len(results)} arena results{scope} from {len(files)} file(s): {', '.join(os.path.basename(f) for f in files)}\n")
    print(markdown(rows) if rows else "(no arena results yet - run `make evaluate` and open the URL it prints)")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"files": files, "rows": rows, "results": results}, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
