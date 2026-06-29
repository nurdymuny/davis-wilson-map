"""section5_convergence_sweep.py — fire the 40-trajectory convergence study.

Per ``inertia_damping/HALCYON_USE_GIGI_FLAG_SPEC.md`` § 5: the load-bearing
test of whether the Section 5 microcanonical-vs-canonical FAIL is a real
93-DOF ergodicity caveat or an artifact of insufficient sample count.

Sweep: ``N_steps ∈ {1000, 2000, 4000, 8000, 16000}`` × 8 independent seeds
= 40 trajectories total. Each fires ``run_validation_report --use-gigi``
against gigi-stream.fly.dev at β=2.5 and writes a sidecar to
``inertia_damping/reports/section5_sweep/n{N}_seed{s}/``.

Per spec § 5, beta-scan + beta-envelope + sector_classifier are disabled
for the sweep — we only need Section 2 (conservation) + Section 5
(microcanonical-vs-canonical) per run, and those are populated by the
leapfrog + canonical heatbath phases which the sweep varies.

Estimated wall: ~80 min total (8 seeds × 5 N_steps; each ~22s at N=100
smoke scales linearly with N_steps; ~10 min per-seed at the full sweep).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

DEFAULT_N_STEPS = [1000, 2000, 4000, 8000, 16000]
DEFAULT_SEEDS = [20260616, 20260617, 20260618, 20260619,
                 20260620, 20260621, 20260622, 20260623]
DEFAULT_OUTPUT = _REPO / "inertia_damping" / "reports" / "section5_sweep"


def _existing_sidecar(out_dir: Path) -> Optional[Path]:
    """Return the most recent completed validation_report_*.json in
    ``out_dir`` (if any), used to skip already-completed trajectories
    when resuming after an interruption."""
    if not out_dir.exists():
        return None
    runs = sorted([p for p in out_dir.iterdir()
                   if p.is_dir() and p.name.startswith("run_")],
                  key=lambda p: p.stat().st_mtime)
    for run in reversed(runs):
        jsons = sorted(run.glob("validation_report_*.json"))
        if jsons:
            return jsons[-1]
    return None


def _run_one(
    n_steps: int,
    seed: int,
    base_url: str,
    api_key: str,
    output_root: Path,
    resume: bool = True,
) -> dict:
    out_dir = output_root / f"n{n_steps}_seed{seed}"
    if resume:
        existing = _existing_sidecar(out_dir)
        if existing is not None:
            return {
                "n_steps": n_steps, "seed": seed, "ok": True,
                "wall_s": 0.0, "skipped": True,
                "run_dir": str(existing.parent),
                "json_path": str(existing),
                "returncode": 0,
            }
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "inertia_damping.run_validation_report",
        "--beta", "2.5",
        "--seed", str(seed),
        "--steps", str(n_steps),
        "--output", str(out_dir),
        "--use-gigi",
        "--gigi-url", base_url,
        "--gigi-api-key", api_key,
        "--no-beta-scan",
        "--no-beta-envelope",
        "--no-sector-classifier",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(_REPO), env=env,
        capture_output=True, text=True, timeout=3600, check=False,
    )
    wall = time.time() - t0
    # Find the produced JSON
    runs = sorted([p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("run_")],
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        return {
            "n_steps": n_steps, "seed": seed, "ok": False, "wall_s": wall,
            "stderr_tail": proc.stderr[-400:] if proc.stderr else "",
            "stdout_tail": proc.stdout[-400:] if proc.stdout else "",
        }
    last = runs[-1]
    json_files = sorted(last.glob("validation_report_*.json"))
    return {
        "n_steps": n_steps, "seed": seed, "ok": True, "wall_s": wall,
        "run_dir": str(last),
        "json_path": str(json_files[-1]) if json_files else None,
        "returncode": proc.returncode,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("GIGI_URL", "http://localhost:3142"))
    ap.add_argument("--api-key", default=os.environ.get("GIGI_API_KEY"))
    ap.add_argument("--n-steps", type=str,
                    default=",".join(str(n) for n in DEFAULT_N_STEPS))
    ap.add_argument("--seeds", type=str,
                    default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the run plan, do not execute")
    ap.add_argument("--no-resume", action="store_true",
                    help="force re-run of trajectories that already have a sidecar")
    args = ap.parse_args()

    if not args.api_key:
        sys.stderr.write("ERROR: GIGI_API_KEY required\n")
        return 2

    n_steps_list = [int(x) for x in args.n_steps.split(",") if x.strip()]
    seeds_list = [int(x) for x in args.seeds.split(",") if x.strip()]
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    plan = [(n, s) for n in n_steps_list for s in seeds_list]
    print(f"Sweep: {len(plan)} runs ({len(n_steps_list)} N × {len(seeds_list)} seeds)")
    print(f"  N_steps:   {n_steps_list}")
    print(f"  seeds:     {seeds_list}")
    print(f"  base_url:  {args.base_url}")
    print(f"  output:    {output_root}")
    if args.dry_run:
        for n, s in plan:
            print(f"  [dry-run] N={n} seed={s}")
        return 0

    results: List[dict] = []
    t_total = time.time()
    resume = not args.no_resume
    for i, (n, s) in enumerate(plan):
        print(f"\n[{i+1}/{len(plan)}] N={n} seed={s} ...", flush=True)
        r = _run_one(n, s, args.base_url, args.api_key, output_root, resume=resume)
        results.append(r)
        if r.get("skipped"):
            print(f"    SKIP (existing sidecar)", flush=True)
            continue
        status = "OK" if r["ok"] else "FAIL"
        print(f"    {status} in {r['wall_s']:.1f}s "
              f"(returncode={r.get('returncode','-')})", flush=True)
        if not r["ok"]:
            print(f"    stderr_tail: {r.get('stderr_tail','')[:300]}", flush=True)
    total_wall = time.time() - t_total

    # Write sweep manifest
    import json
    manifest = {
        "started_at": time.time() - total_wall,
        "wall_seconds": total_wall,
        "n_runs": len(results),
        "n_ok": sum(1 for r in results if r["ok"]),
        "n_fail": sum(1 for r in results if not r["ok"]),
        "n_steps_list": n_steps_list,
        "seeds_list": seeds_list,
        "base_url": args.base_url,
        "runs": results,
    }
    manifest_path = output_root / "sweep_manifest.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nTotal wall: {total_wall/60:.1f} min")
    print(f"  OK:   {manifest['n_ok']}/{len(results)}")
    print(f"  FAIL: {manifest['n_fail']}/{len(results)}")
    print(f"  Manifest: {manifest_path}")
    return 0 if manifest["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
