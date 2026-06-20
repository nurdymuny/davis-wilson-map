"""run_battery.py — CLI for the Halcyon Falsification Battery (Section 11).

Per HALCYON_FALSIFICATION_BATTERY_SPEC.md (v2). Standalone executable;
emits the section_11_falsification_battery JSON schema specified in
SPEC v2 §6. Integration into validation_report.py is a follow-up.

Usage:
    python -m inertia_damping.scripts.run_battery [options]

Options:
    --fast           run --battery-fast (3 seeds, 2 Q, 3 omegas; ~15 min)
    --seeds N        number of seeds (default 8)
    --n-equil N      equilibration steps (default 200)
    --n-steps N      driven-phase steps per lock-in (default 1200)
    --dt F           timestep (default 0.02)
    --output DIR     output dir (default: inertia_damping/reports/section11_battery)
    --no-gauge       freeze gauge dynamics (debug only, breaks H8 meaningfulness)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="run --battery-fast (3 seeds, ~15 min)")
    ap.add_argument("--seeds", type=int, default=8,
                    help="number of seeds (default 8; ignored if --fast)")
    ap.add_argument("--n-equil", type=int, default=200,
                    help="equilibration steps per lock-in")
    ap.add_argument("--n-steps", type=int, default=1200,
                    help="driven-phase steps per lock-in")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--output", type=str,
                    default=str(_REPO / "inertia_damping" / "reports" / "section11_battery"))
    ap.add_argument("--no-gauge", action="store_true",
                    help="freeze gauge dynamics (debug only)")
    args = ap.parse_args()

    from inertia_damping import buckyball_graph as bg
    from inertia_damping import falsification_battery as fb

    seeds = 3 if args.fast else args.seeds
    n_steps = 800 if args.fast else args.n_steps
    n_equil = 100 if args.fast else args.n_equil

    print(f"== Halcyon Falsification Battery ==")
    print(f"  Mode:         {'--fast' if args.fast else 'standard'}")
    print(f"  Seeds:        {seeds}")
    print(f"  n_equil:      {n_equil}")
    print(f"  n_steps:      {n_steps}")
    print(f"  dt:           {args.dt}")
    print(f"  gauge:        {'frozen' if args.no_gauge else 'enabled'}")
    print(f"  output:       {args.output}")
    print()

    graph = bg.build_truncated_icosahedron()
    t0 = time.time()
    result = fb.run_battery_fast(
        graph, dt=args.dt,
        freeze_gauge=args.no_gauge,
        verbose=True,
        n_seeds=seeds,
        n_equil=n_equil,
        n_steps=n_steps,
    )
    wall = time.time() - t0

    s11 = result["section_11_falsification_battery"]
    print()
    print(f"Total wall: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"Verdict:    {s11['sudoku_verdict']}")
    print(f"Completion: {s11['completion_invariant_simulation']}")
    print(f"alpha:      {s11['alpha_measured']:.4e}", end="")
    if s11['alpha_sem_blocked'] > 0:
        print(f" +/- {s11['alpha_sem_blocked']:.4e}"
              f"  (|alpha|/sem = "
              f"{abs(s11['alpha_measured']) / s11['alpha_sem_blocked']:.2f})")
    else:
        print()
    print()
    for h_id, g_data in s11["battery"].items():
        struck = g_data["struck"]
        icon = "OK " if struck is True else ("X  " if struck is False else "-- ")
        print(f"  {icon}{h_id}: struck={struck} - {g_data['reason'][:80]}")

    # Save sidecar
    os.makedirs(args.output, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    mode_tag = "fast" if args.fast else "full"
    out_path = Path(args.output) / f"battery_{mode_tag}_{ts}.json"
    payload = {
        "report_metadata": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "spec_version": "HALCYON_FALSIFICATION_BATTERY_SPEC.md v2",
            "mode": mode_tag,
            "wall_seconds": wall,
            "args": vars(args),
        },
        **result,
    }
    raw_bytes = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    with open(out_path, "wb") as fh:
        fh.write(raw_bytes)
    print(f"\nSidecar:      {out_path}")
    print(f"SHA-256:      {sha256}")
    return 0 if s11["sudoku_verdict"] == "PASS_SIMULATION_ONLY" else 1


if __name__ == "__main__":
    sys.exit(main())
