"""section5_cuda_sweep.py — the whole 40-trajectory Section 5 study as ONE
batched GPU pass + 8 small CPU heatbaths.

What this replaces
------------------
``section5_convergence_sweep.py`` fired 40 separate
``run_validation_report --use-gigi`` subprocesses over fly.io, taking
~13 hours wall. This script does the same physics in ~2 minutes by:

1. Running the canonical Wilson heatbath ONCE per seed on CPU
   (Section-5 P_heatbath; ~7 s per seed × 8 seeds = ~1 min).
2. Running a single batched CUDA leapfrog: 8 seeds in parallel up to
   N=16000 steps, with the running time-average sampled at each of the
   5 checkpoints {1000, 2000, 4000, 8000, 16000}.
3. Composing 40 validation-report sidecars in the same shape as the
   v1.2 production runs, so ``section5_closure_analysis.py`` reads
   them unchanged.

Architectural note for the chapter
----------------------------------
This is the substrate-as-store pattern the v0.5 design memo asked
for. Compute is local + GPU-fast; the substrate's role would be to
*ingest* the 40 trajectories afterward and let us run comparison
queries across the chapter narrative. We are not doing that ingest
step here — the closure verdict is the gating artefact, not the
storage receipt.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

DEFAULT_OUTPUT = _REPO / "inertia_damping" / "reports" / "section5_sweep_cuda"

DEFAULT_N_STEPS = [1000, 2000, 4000, 8000, 16000]
DEFAULT_SEEDS = [20260616, 20260617, 20260618, 20260619,
                 20260620, 20260621, 20260622, 20260623]
TOL_METHOD_CROSSCHECK = 0.02     # mirrors validation_report.TOL


# ---------------------------------------------------------------------------
# Section-5 schema helper — Flyvbjerg-Petersen blocking in the validation_report
# convention so the closure analyzer treats CUDA sidecars identically.
# ---------------------------------------------------------------------------
def _fp_block(x: np.ndarray) -> Dict[str, Any]:
    """Match validation_report._flyvbjerg_petersen_blocking output shape."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    sem_naive = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    if n < 4:
        return {
            "sem_naive": sem_naive, "sem_blocked": sem_naive,
            "n_eff": float(n), "plateau_block_size": 1,
            "plateau_detected": False,
            "blocking_curve": [[1, sem_naive]] if n > 0 else [],
            "regime": "degenerate" if n <= 1 else "single_level",
        }
    arr = x.copy()
    block = 1
    curve: List[List[float]] = []
    while len(arr) >= 4:
        s = float(arr.std(ddof=1) / math.sqrt(len(arr)))
        curve.append([block, s])
        if len(arr) % 2:
            arr = arr[:-1]
        arr = (arr[0::2] + arr[1::2]) / 2.0
        block *= 2
    # Plateau detection: 3 levels in a row within 10%.
    plateau_block = None
    plateau_sem = None
    for i in range(len(curve) - 2):
        s0, s1, s2 = curve[i][1], curve[i + 1][1], curve[i + 2][1]
        if s1 <= 0 or s0 <= 0:
            continue
        if abs(s2 - s1) / s1 < 0.10 and abs(s1 - s0) / s0 < 0.10:
            plateau_block = curve[i + 1][0]
            plateau_sem = curve[i + 1][1]
            break
    if plateau_block is not None:
        sem_blocked = plateau_sem
        n_eff = float(n / plateau_block)
        regime = "plateau"
        plateau_detected = True
        plateau_block_size = plateau_block
    elif len(curve) == 1:
        sem_blocked = curve[0][1]
        n_eff = float(n / curve[0][0])
        regime = "single_level"
        plateau_detected = False
        plateau_block_size = curve[0][0]
    else:
        sem_blocked = curve[-1][1]
        n_eff = float(n / curve[-1][0])
        regime = "no_plateau"
        plateau_detected = False
        plateau_block_size = curve[0][0]
    return {
        "sem_naive": sem_naive, "sem_blocked": float(sem_blocked),
        "n_eff": float(n_eff),
        "plateau_block_size": int(plateau_block_size),
        "plateau_detected": bool(plateau_detected),
        "blocking_curve": curve,
        "regime": regime,
    }


# ---------------------------------------------------------------------------
# Heatbath P_heatbath per seed (Python CPU; small)
# ---------------------------------------------------------------------------
def heatbath_per_seed(
    graph, beta: float, seeds: List[int],
    n_thermalize: int = 200, n_measure: int = 2000, measure_every: int = 1,
) -> Dict[int, Dict[str, Any]]:
    """Run a fresh thermalize + measure heatbath per seed."""
    from inertia_damping import buckyball_action as ba
    from inertia_damping import buckyball_heatbath as hb
    out: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        rng = np.random.default_rng(int(s))
        U = ba.identity_links(graph.n_edges)
        # Thermalize
        for _ in range(n_thermalize):
            hb.heatbath_sweep(U, graph, beta, generator=rng)
        # Measure
        run = hb.heatbath_run(U, graph, beta, n_sweeps=n_measure,
                              measure_every=measure_every, generator=rng)
        Ps = np.asarray(run["P_history"], dtype=np.float64)
        b = _fp_block(Ps)
        out[s] = {
            "P_heatbath": float(Ps.mean()),
            "P_heatbath_sem_naive": b["sem_naive"],
            "P_heatbath_sem_blocked": b["sem_blocked"],
            "P_heatbath_n_eff": b["n_eff"],
            "P_heatbath_plateau_block_size": b["plateau_block_size"],
            "P_heatbath_plateau_detected": b["plateau_detected"],
            "P_heatbath_blocking_curve": b["blocking_curve"],
            "P_heatbath_blocking_regime": b["regime"],
            "n_heatbath_samples": int(Ps.size),
            "P_history": Ps.tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Compose Section-5 sidecars in the validation_report shape
# ---------------------------------------------------------------------------
def make_section5_payload(
    n_steps: int, seed: int,
    heatbath: Dict[str, Any],
    P_time_history: np.ndarray,
) -> Dict[str, Any]:
    b = _fp_block(P_time_history)
    P_time = float(P_time_history.mean())
    P_hb = heatbath["P_heatbath"]
    gap = abs(P_time - P_hb)
    margin = TOL_METHOD_CROSSCHECK / gap if gap > 0 else float("inf")
    verdict = "PASS" if gap <= TOL_METHOD_CROSSCHECK else "FAIL"
    return {
        "available": True,
        "P_time": P_time,
        "P_time_sem": b["sem_blocked"],
        "n_time_samples": int(P_time_history.size),
        "P_time_sem_naive": b["sem_naive"],
        "P_time_sem_blocked": b["sem_blocked"],
        "P_time_n_eff": b["n_eff"],
        "P_time_plateau_block_size": b["plateau_block_size"],
        "P_time_plateau_detected": b["plateau_detected"],
        "P_time_blocking_curve": b["blocking_curve"],
        "P_time_blocking_regime": b["regime"],
        "P_heatbath": P_hb,
        "P_heatbath_sem": heatbath["P_heatbath_sem_blocked"],
        "n_heatbath_samples": heatbath["n_heatbath_samples"],
        "P_heatbath_sem_naive": heatbath["P_heatbath_sem_naive"],
        "P_heatbath_sem_blocked": heatbath["P_heatbath_sem_blocked"],
        "P_heatbath_n_eff": heatbath["P_heatbath_n_eff"],
        "P_heatbath_plateau_block_size": heatbath["P_heatbath_plateau_block_size"],
        "P_heatbath_plateau_detected": heatbath["P_heatbath_plateau_detected"],
        "P_heatbath_blocking_curve": heatbath["P_heatbath_blocking_curve"],
        "P_heatbath_blocking_regime": heatbath["P_heatbath_blocking_regime"],
        "sem_convention": "flyvbjerg_petersen_blocked",
        "gap": gap,
        "tolerance": TOL_METHOD_CROSSCHECK,
        "margin_factor": margin,
        "verdict": verdict,
    }


def write_sidecar(out_dir: Path, payload: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fpath = run_dir / f"validation_report_{ts}.json"
    with open(fpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
    return fpath


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=2.5)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--n-steps-max", type=int, default=16000)
    ap.add_argument("--checkpoints", type=str,
                    default=",".join(str(n) for n in DEFAULT_N_STEPS))
    ap.add_argument("--seeds", type=str,
                    default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--n-heatbath-thermalize", type=int, default=200)
    ap.add_argument("--n-heatbath-measure", type=int, default=2000)
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--device", type=str, default=None,
                    help="torch device ('cuda', 'cpu'); auto-detect if omitted")
    args = ap.parse_args()

    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"== Section 5 CUDA sweep ==")
    print(f"  device:      {device}")
    print(f"  beta:        {args.beta}")
    print(f"  dt:          {args.dt}")
    print(f"  n_steps_max: {args.n_steps_max}")
    print(f"  checkpoints: {checkpoints}")
    print(f"  seeds:       {seeds}")
    print(f"  output:      {output_root}")
    if device.type == "cuda":
        print(f"  cuda device: {torch.cuda.get_device_name(0)}")

    from inertia_damping import buckyball_graph as bgraph
    from inertia_damping.cuda import batched_leapfrog as gpu

    t_total = time.time()
    graph = bgraph.build_truncated_icosahedron()
    print(f"\nBuilding topology...")
    topo = gpu.build_topology_from_graph(graph, device=device)

    # --- Heatbath per seed (CPU) -------------------------------------------
    print(f"\nHeatbath per seed (n_thermalize={args.n_heatbath_thermalize}, "
          f"n_measure={args.n_heatbath_measure})...")
    t_hb = time.time()
    heatbath = heatbath_per_seed(
        graph, args.beta, seeds,
        n_thermalize=args.n_heatbath_thermalize,
        n_measure=args.n_heatbath_measure,
    )
    print(f"  heatbath wall: {time.time() - t_hb:.1f}s")
    for s in seeds:
        hb = heatbath[s]
        print(f"  seed {s}: P_heatbath = {hb['P_heatbath']:.6f} +/- "
              f"{hb['P_heatbath_sem_blocked']:.3e}, "
              f"n_eff={hb['P_heatbath_n_eff']:.1f}, "
              f"regime={hb['P_heatbath_blocking_regime']}")

    # --- Batched CUDA leapfrog (one pass) ----------------------------------
    print(f"\nBatched CUDA leapfrog: B={len(seeds)} seeds × {args.n_steps_max} steps...")
    t_lf = time.time()
    result = gpu.run_batched_trajectory(
        seeds=seeds, beta=args.beta, dt=args.dt,
        n_steps_max=args.n_steps_max,
        checkpoints=checkpoints,
        topo=topo, device=device,
    )
    wall_lf = time.time() - t_lf
    print(f"  leapfrog wall: {wall_lf:.1f}s ({wall_lf*1000/args.n_steps_max:.2f} ms/step)")
    print(f"  H rel drift max per seed: {result['dH_rel_max']}")
    print(f"  max |G| per seed:         {result['G_max_max']}")

    # --- Compose 40 sidecars -----------------------------------------------
    print(f"\nComposing sidecars in {output_root}...")
    P_history = result["P_history"]                # (B, n_steps_max)
    written: List[Path] = []
    for b_idx, seed in enumerate(seeds):
        for n in checkpoints:
            trajectory = P_history[b_idx, :n]
            s5 = make_section5_payload(n, seed, heatbath[seed], trajectory)
            payload = {
                "schema_version": "section5_cuda_sweep_v1",
                "generated_at": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                "beta": args.beta,
                "dt": args.dt,
                "n_steps": int(n),
                "seed": int(seed),
                "device": str(device),
                "cuda_kernel": "inertia_damping.cuda.batched_leapfrog",
                "section_5_method_crosscheck": s5,
                "H_rel_drift_max": float(result["dH_rel_max"][b_idx]),
                "G_max_max": float(result["G_max_max"][b_idx]),
                "H0": float(result["H0"][b_idx]),
                "H_final": float(result["H_final"][b_idx]),
            }
            out_dir = output_root / f"n{n}_seed{seed}"
            fpath = write_sidecar(out_dir, payload)
            written.append(fpath)

    # --- Manifest ----------------------------------------------------------
    manifest = {
        "schema_version": "section5_cuda_sweep_v1",
        "device": str(device),
        "cuda_device_name": (torch.cuda.get_device_name(0)
                             if device.type == "cuda" else None),
        "beta": args.beta,
        "dt": args.dt,
        "n_steps_max": args.n_steps_max,
        "checkpoints": checkpoints,
        "seeds": seeds,
        "n_sidecars": len(written),
        "wall_total_s": time.time() - t_total,
        "wall_leapfrog_s": wall_lf,
    }
    mpath = output_root / "sweep_manifest.json"
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nTotal wall: {(time.time() - t_total)/60:.1f} min "
          f"({len(written)} sidecars)")
    print(f"  Manifest:  {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
