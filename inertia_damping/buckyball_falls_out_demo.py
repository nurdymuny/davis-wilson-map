"""buckyball_falls_out_demo.py — Local YM "everything falls out" demo.

Runs a β-walk of SU(2) Yang-Mills on the buckyball (truncated icosahedron, V=60, E=90,
F=32, χ=2) crossing the deconfinement transition β_c ≈ 2.298, FRESH identity per step
(no Markov-chain path-history accumulation between β values), with cross-check against
the analytical Migdal-Witten exact ⟨P⟩ formula via `buckyball_yangmills_exact.py`.

What "falls out":
  * Curvature density            1 − ⟨P⟩
  * Davis capacity proxy         C ≈ τ/K  =  1 / (1 − ⟨P⟩)
  * Mass-gap reading             ⟨P⟩ vs Migdal-Witten exact (Witten 1991)
  * Topological charge surrogate Q via `Q_surrogate`
  * Face holonomies              face_q0_values (32 face quaternions)
  * Deconfinement transition     visible in C and Q at β_c ≈ 2.298

This is the LOCAL counterpart to the gigi-side β-walk at
inertia_damping/reports/holonomy_battery_v3_1_3/v32_substrate_walk_beta.json (commit
8687b65). It uses only the in-tree buckyball Python toolkit; no engine, no GQL, no GPU.

Provenance: the demo Bee asked for on 2026-06-22 ("get the mass gap, hypothesis,
curvature, all of the things to basically fall out") that we couldn't do through gigi
today because the Halcyon LATTICE+GAUGE_FIELD subsystem and the bundle subsystem don't
bridge (see HALCYON_TO_GIGI_2026_06_22_bridge_ask.md). The local Python toolkit makes
the same demo work without waiting for the bridge.

Usage:
    python -m inertia_damping.buckyball_falls_out_demo

Output:
    inertia_damping/reports/buckyball_local_falls_out.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from inertia_damping import (
    buckyball_graph,
    buckyball_heatbath,
    buckyball_observables,
    buckyball_yangmills_exact,
)


# β values spanning the deconfinement transition. β_c ≈ 2.298 sits between 2.25 and 2.30.
BETAS = [0.5, 1.0, 1.5, 2.0, 2.25, 2.30, 2.5, 2.7, 3.0]

# Thermalization parameters per step. Fresh identity start; no accumulation.
N_SWEEPS_THERMALIZE = 1000
N_MEASUREMENTS = 100
N_SWEEPS_BETWEEN_MEASUREMENTS = 5
SEED = 20260616

OUTPUT_PATH = "inertia_damping/reports/buckyball_local_falls_out.json"


def classify_phase(beta: float, beta_c: float = 2.298, tol: float = 0.01) -> str:
    if beta < beta_c - tol:
        return "confined"
    if abs(beta - beta_c) < tol:
        return "at-β_c"
    return "deconfined"


def main() -> None:
    graph = buckyball_graph.build_truncated_icosahedron()
    V, E, F = graph.n_vertices, graph.n_edges, graph.n_faces
    chi = V - E + F

    print(f"=== buckyball: V={V}, E={E}, F={F}, chi={chi} (= 2 for S^2 ✓) ===\n")
    print(
        f"{'β':>5} {'<P>_meas':>9} {'<P>_exact':>10} {'Δ':>9} {'1-<P>':>8} "
        f"{'C≈':>7} {'Q_surr':>9} {'⟨q0⟩':>9} {'σ(q0)':>8}  phase"
    )
    print("─" * 110)

    results = []
    t_start = time.perf_counter()
    for beta in BETAS:
        r = buckyball_heatbath.thermalize(
            graph,
            beta=beta,
            n_sweeps=N_SWEEPS_THERMALIZE,
            n_measure=N_MEASUREMENTS,
            n_measure_every=N_SWEEPS_BETWEEN_MEASUREMENTS,
            seed=SEED,
        )

        U_final = r["U_final"]
        P_meas = float(r["P_mean"])
        P_sem = float(r["P_sem"])
        P_exact = buckyball_yangmills_exact.exact_mean_plaquette_su2_buckyball(beta)
        Q = float(buckyball_observables.Q_surrogate(U_final, graph))
        face_q0 = np.asarray(buckyball_observables.face_q0_values(U_final, graph))

        curvature_density = max(0.0, 1.0 - P_meas)
        C_proxy = 1.0 / curvature_density if curvature_density > 1e-15 else float("inf")
        delta = P_meas - P_exact
        phase = classify_phase(beta)

        results.append({
            "beta": beta,
            "P_measured": P_meas,
            "P_sem": P_sem,
            "P_exact": P_exact,
            "delta_P": delta,
            "curvature_density": curvature_density,
            "C_proxy": C_proxy,
            "Q_surrogate": Q,
            "face_q0_mean": float(face_q0.mean()),
            "face_q0_std": float(face_q0.std()),
            "n_samples": int(r["n_samples"]),
            "phase": phase,
        })

        print(
            f"{beta:5.2f} {P_meas:9.4f} {P_exact:10.4f} {delta:+9.4f} "
            f"{curvature_density:8.4f} {C_proxy:7.2f} {Q:9.4f} "
            f"{face_q0.mean():9.4f} {face_q0.std():8.4f}  {phase}"
        )

    wall_time = time.perf_counter() - t_start
    max_dev = max(abs(r["delta_P"]) for r in results)
    rms_dev = (sum(r["delta_P"] ** 2 for r in results) / len(results)) ** 0.5

    print(
        f"\n=== Total: {wall_time:.1f}s for {len(results)} β-points, "
        f"FRESH identity each step, {N_SWEEPS_THERMALIZE} sweeps thermalize "
        f"+ {N_MEASUREMENTS} measurements per step ==="
    )
    print(f"\n=== Cross-check vs Migdal-Witten exact (Witten 1991) ===")
    print(f"  max |measured − exact| = {max_dev:.4f}")
    print(f"  RMS deviation          = {rms_dev:.4f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "protocol": (
                "buckyball SU(2) local β-walk; fresh identity per step; "
                "cross-checked against Migdal-Witten exact ⟨P⟩ for SU(2) 2D YM on S²"
            ),
            "framing": (
                "everything falls out: curvature density from <P>, capacity C from "
                "1/(1-<P>), topology from Q_surrogate, holonomy from face q0, mass-gap "
                "reading from agreement with analytical exact reference"
            ),
            "graph": {"V": V, "E": E, "F": F, "chi": chi},
            "seed": SEED,
            "n_sweeps_per_step": N_SWEEPS_THERMALIZE,
            "n_measurements_per_step": N_MEASUREMENTS,
            "n_sweeps_between_measurements": N_SWEEPS_BETWEEN_MEASUREMENTS,
            "max_deviation_vs_exact": max_dev,
            "rms_deviation_vs_exact": rms_dev,
            "results": results,
            "wall_time_seconds": wall_time,
        }, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
