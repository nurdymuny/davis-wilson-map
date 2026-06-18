"""
run_validation_report.py -- End-to-end orchestrator that produces a real
Halcyon validation run + report artifact.

USAGE
-----
    python -m inertia_damping.run_validation_report \
        --beta 2.5 --steps 1000 --seed 20260616 \
        --output reports/

OUTPUTS (under reports/run_<timestamp>/)
----------------------------------------
    trajectory.json                     -- v0.1 schema (same as the cage demo)
    final_state.npz                     -- U_init, E_init, U_final, E_final,
                                            canonical heatbath ensemble,
                                            beta-scan table
    validation_report_<ts>.md           -- human-readable report
    validation_report_<ts>.json         -- machine-readable artifact
    SHA256SUMS                          -- pins every input file

DISCIPLINE
----------
- Foreground execution only. ~20-30 minutes with default flags including
  --beta-envelope and --sector-classifier; ~15 minutes with --no-beta-envelope
  and without --sector-classifier.
- All randomness is seeded. Two reviewers running with the same --seed produce
  the same numbers to FP64 epsilon.
- DOES NOT modify any validated kernel file.

SEED OFFSET ALLOCATION (single source of truth; extend before adding any new
RNG-consuming phase)
--------------------------------------------------------------------------
    [    0,   1000)  trajectory init / initial heatbath thermalisation
    [ 1000,  10000)  canonical heatbath cross-check (seed + 100)
    [10000, 100000)  beta-scan (seed + int(round(b * 1000)))
    [100000, 200000) beta-envelope (seed + 50000 + int(round(b * 1000));
                     so range [150000, 158000) for b in [2.0, 3.5])
    [200000, 300000) sector_classifier (seed + 200000 + per-fold offset)

Any new orchestrator phase MUST extend this table before consuming a new
offset block.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Make sure the package importable when invoked as `-m inertia_damping....`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inertia_damping import buckyball_graph
from inertia_damping import buckyball_action
from inertia_damping import buckyball_heatbath
from inertia_damping import buckyball_integrator
from inertia_damping import buckyball_observables
from inertia_damping import buckyball_yangmills_exact as ym_exact
from inertia_damping import validation_report

DEFAULT_BETA = 2.5
DEFAULT_DT = 0.02
DEFAULT_STEPS = 1000
DEFAULT_THERM_SWEEPS = 200
DEFAULT_MEASURE_EVERY = 20
DEFAULT_BETA_SCAN = [0.5, 1.0, 1.5, 2.5, 4.0, 6.0]
DEFAULT_CROSS_CHECK_STEPS = 4000
DEFAULT_HEATBATH_CROSS_CHECK_SWEEPS = 4096
DEFAULT_HEATBATH_CROSS_CHECK_THERM = 500

# 1.2 schema: beta-envelope local-stability sweep. Edge-probes the operating
# point (NOT an interior re-sample of beta-scan); 1000 measurement sweeps
# per beta to support >=4 Flyvbjerg-Petersen doubling levels.
DEFAULT_BETA_ENVELOPE = [1.5, 2.0, 2.5, 3.0, 3.5]
ENVELOPE_THERM_SWEEPS = 50
ENVELOPE_MEASURE_SWEEPS = 1000
ENVELOPE_LEAPFROG_STEPS = 400
ENVELOPE_MEASURE_EVERY = 20
ENVELOPE_MAX_SIDECAR_BYTES = 64 * 1024

# 1.2 schema: sector_classifier blind-band gate on real trajectory windows
# segmented by observed Q_surrogate bin. Limits sidecar bloat with a hard cap.
CLASSIFIER_MAX_SIDECAR_BYTES = 32 * 1024


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "(unavailable)"


def _phase(label: str) -> float:
    print(f"[{label}] starting...", flush=True)
    return time.time()


def _done(label: str, t0: float) -> None:
    print(f"[{label}] done in {time.time() - t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end Halcyon validation run.")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--dt", type=float, default=DEFAULT_DT)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--therm-sweeps", type=int, default=DEFAULT_THERM_SWEEPS,
                    help="heatbath sweeps for the symplectic-arm cold start")
    ap.add_argument("--measure-every", type=int, default=DEFAULT_MEASURE_EVERY)
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--cross-check-steps", type=int, default=DEFAULT_CROSS_CHECK_STEPS)
    ap.add_argument("--canonical-sweeps", type=int,
                    default=DEFAULT_HEATBATH_CROSS_CHECK_SWEEPS)
    ap.add_argument("--canonical-therm", type=int,
                    default=DEFAULT_HEATBATH_CROSS_CHECK_THERM)
    ap.add_argument("--beta-scan", type=str, default=",".join(str(b) for b in DEFAULT_BETA_SCAN),
                    help="comma-separated list of betas for the wide scan over the Migdal-Witten envelope")
    ap.add_argument("--no-beta-scan", action="store_true",
                    help="skip phase 8a (wide multi-beta scan)")
    ap.add_argument("--beta-envelope", type=str,
                    default=",".join(str(b) for b in DEFAULT_BETA_ENVELOPE),
                    help="comma-separated betas; default edge-probes [1.5,2.0,2.5,3.0,3.5]; adds ~45s per beta")
    ap.add_argument("--no-beta-envelope", action="store_true",
                    help="skip phase 8b (local-stability envelope sweep)")
    ap.add_argument("--no-sector-classifier", action="store_true",
                    help="skip phase 8c (blind classifier on Q_surrogate bins)")
    ap.add_argument("--sector-classifier", action="store_true", default=False,
                    help="blind discrimination test on observed-Q_surrogate-binned trajectory windows; ~30-60s")
    ap.add_argument("--sector-k", type=int, default=5,
                    help="k for the k-NN LOO classifier in phase 8c (default 5)")
    ap.add_argument("--output", type=str, default=os.path.join(_HERE, "reports"))
    ap.add_argument("--no-cross-check", action="store_true",
                    help="skip the canonical heatbath cross-check (Section 5 -> unavailable; phases 8b, 8c also skipped because they reuse the heatbath kernel)")
    args = ap.parse_args()

    t_total = time.time()

    # Output run directory.
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}", flush=True)

    code_commit = _git_head()

    # --- 1. Build the graph -----------------------------------------------
    # Cold-start invariant: this script does not persist or warm-start the
    # gauge field. Each run cold-starts from buckyball_action.identity_links()
    # and writes its U_init / U_final to a per-run final_state.npz sidecar
    # before exit. Halcyon's substrate-side migration onto GIGI assumes an
    # embedded (PyO3/CFFI) binding, NOT the HTTP surface; adding a --resume
    # or --warm-start flag would cross into durable-persistence territory
    # and re-open the Part II HTTP-durable gap as a Halcyon concern.
    t0 = _phase("build_graph")
    graph = buckyball_graph.build_truncated_icosahedron()
    assert graph.verify_euler()
    assert graph.n_vertices == 60 and graph.n_edges == 90 and graph.n_faces == 32
    _done("build_graph", t0)

    # --- 2. Symplectic arm: cold-thermalize via heatbath ------------------
    t0 = _phase(f"heatbath_thermalize x{args.therm_sweeps} @ beta={args.beta}")
    rng_therm = np.random.default_rng(args.seed)
    U_therm = buckyball_action.identity_links(graph.n_edges)
    for _ in range(args.therm_sweeps):
        buckyball_heatbath.heatbath_sweep(U_therm, graph, args.beta, generator=rng_therm)
    _done("heatbath_thermalize", t0)

    # --- 3. Canonical E (Gauss-projected to FP64 floor) -------------------
    t0 = _phase("initialize_E_canonical (CG project)")
    gen_E = torch.Generator()
    gen_E.manual_seed(args.seed + 1)
    E_init_t = buckyball_integrator.initialize_E_canonical(
        graph, args.beta, gauge_group="SU(2)",
        generator=gen_E, project_gauss=True, U=U_therm,
    )
    U_init_t = U_therm.detach().clone()
    H0, K0, V0 = buckyball_integrator.compute_hamiltonian(U_init_t, E_init_t, graph, args.beta)
    print(f"  H_0 = {H0:.6f} (K={K0:.6f}, V={V0:.6f}), K/V = {(K0/max(V0,1e-30)):.4f}",
          flush=True)
    _done("initialize_E_canonical", t0)

    # --- 4. Run leapfrog + harvest snapshots ------------------------------
    t0 = _phase(f"leapfrog x{args.steps} @ dt={args.dt}, measure every {args.measure_every}")
    result = buckyball_observables.integrate_with_states(
        U_init_t, E_init_t,
        dt=args.dt, n_steps=args.steps, graph=graph, beta=args.beta,
        measure_every=args.measure_every, include_initial=False,
    )
    H_hist = result["H_history"]
    print(f"  max |dH/H_0| over {len(H_hist)} snapshots: "
          f"{(max(abs(h - H0) for h in H_hist) / abs(H0)):.3e}", flush=True)
    # Trajectory-wise Gauss history (new sidecar field; kernel-supplied, no
    # double-physics). Length-consistency assertion catches step misalignment.
    assert len(result["U_traj"]) == len(result["E_traj"]) == len(result["steps"]) == len(result["G_history"]), (
        f"integrate_with_states parallel-history mismatch: "
        f"U_traj={len(result['U_traj'])}, E_traj={len(result['E_traj'])}, "
        f"steps={len(result['steps'])}, G_history={len(result['G_history'])}"
    )
    gauss_history_arr = np.asarray(
        [(int(result["steps"][i]), float(result["G_history"][i]))
         for i in range(len(result["steps"]))],
        dtype=np.float64,
    )
    g_argmax = int(gauss_history_arr[:, 1].argmax())
    print(f"  max |G|_inf across trajectory: {gauss_history_arr[g_argmax, 1]:.3e} "
          f"at step {int(gauss_history_arr[g_argmax, 0])} "
          f"(over {gauss_history_arr.shape[0]} snapshots; "
          f"cost: ~50 O(n_edges) ops, negligible)", flush=True)
    _done("leapfrog", t0)

    # --- 5. Dump trajectory.json (v0.1 schema, same as the cage demo) ----
    t0 = _phase("dump_trajectory")
    traj_path = run_dir / "trajectory.json"
    # We force schema_version 0.1 to keep parity with the existing demo file:
    # all v0.2 drill-down arrays are extra noise for the report path. To do
    # this we save the trajectory directly (the kernel dump_trajectory always
    # writes 0.2). Reuse the dumper's frame builder to keep parity:
    n_frames = len(result["U_traj"])
    frames: List[Dict[str, Any]] = []
    for i in range(n_frames):
        U_i = result["U_traj"][i]
        E_i = result["E_traj"][i]
        phases = buckyball_observables.edge_phase(U_i).detach().cpu().numpy()
        kinetic = buckyball_observables.edge_kinetic(E_i).detach().cpu().numpy()
        frames.append({
            "t": float(result["times"][i]),
            "step": int(result["steps"][i]),
            "Q": float(buckyball_observables.Q_surrogate(U_i, graph)),
            "plaquette_mean": float(buckyball_observables._plaquette_mean_from_U(U_i, graph)),
            "edge_phases": [float(x) for x in phases.tolist()],
            "edge_kinetic": [float(x) for x in kinetic.tolist()],
            "control": {},
            "energy": float(result["H_history"][i]),
        })
    payload = {
        "schema_version": "0.1",
        "gauge_group": "SU(2)",
        "lattice": {"vertices": graph.n_vertices, "edges": graph.n_edges, "faces": graph.n_faces},
        "geometry": buckyball_observables._geometry_block(graph),
        "frames": frames,
        "beta": args.beta,
        "dt": args.dt,
        "measure_every": args.measure_every,
        "n_thermalization_sweeps": args.therm_sweeps,
        "n_steps": args.steps,
        "seed": args.seed,
    }
    with open(traj_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"  wrote {traj_path}", flush=True)
    _done("dump_trajectory", t0)

    # --- 6. Long microcanonical run for Section 5 cross-check -------------
    # Phase 6 deliberately uses leapfrog_step (not integrate_with_states) and
    # is NOT instrumented with G_history -- it is a Section 5 cross-check,
    # not the load-bearing trajectory-wise check.
    cross_P_history: List[float] = []
    if not args.no_cross_check:
        t0 = _phase(f"microcanonical cross-check leapfrog x{args.cross_check_steps}")
        U_x = U_init_t.detach().clone()
        E_x = E_init_t.detach().clone()
        for s in range(args.cross_check_steps):
            U_x, E_x = buckyball_integrator.leapfrog_step(U_x, E_x, args.dt, graph, args.beta)
            if s % args.measure_every == 0:
                Uf = buckyball_action.all_face_holonomies(U_x, graph)
                cross_P_history.append(float(Uf[:, 0].mean()))
        _done("microcanonical cross-check", t0)

    # --- 7. Canonical heatbath ensemble at beta for cross-check ----------
    canonical_P_history: List[float] = []
    if not args.no_cross_check:
        t0 = _phase(f"canonical heatbath @ beta={args.beta}: "
                    f"{args.canonical_therm} therm + {args.canonical_sweeps} measure")
        therm_result = buckyball_heatbath.thermalize(
            graph, args.beta,
            n_sweeps=args.canonical_therm,
            n_measure=args.canonical_sweeps,
            n_measure_every=2,
            seed=args.seed + 100,
        )
        canonical_P_history = list(therm_result["P_history"])
        print(f"  canonical <P> = {therm_result['P_mean']:.6f} "
              f"+/- {therm_result['P_sem']:.6f} over {therm_result['n_samples']} samples",
              flush=True)
        _done("canonical heatbath", t0)

    # --- 8. beta-scan -----------------------------------------------------
    beta_scan: List[Dict[str, float]] = []
    if not args.no_beta_scan:
        betas = [float(x.strip()) for x in args.beta_scan.split(",") if x.strip()]
        t0 = _phase(f"beta-scan over {betas}")
        for b in betas:
            print(f"  beta = {b}", flush=True)
            res = buckyball_heatbath.thermalize(
                graph, b,
                n_sweeps=200,
                n_measure=600,
                n_measure_every=2,
                seed=args.seed + int(round(b * 1000)),
            )
            P_exact = ym_exact.exact_mean_plaquette_su2_buckyball(b)
            beta_scan.append({
                "beta": b,
                "P_measured": float(res["P_mean"]),
                "P_sem": float(res["P_sem"]),
                "P_exact": float(P_exact),
                "P_history": [float(x) for x in res["P_history"]],
            })
            print(f"    <P>_meas = {res['P_mean']:.6f}  <P>_exact = {P_exact:.6f}  "
                  f"gap = {abs(res['P_mean']-P_exact):.3e}", flush=True)
        _done("beta-scan", t0)

    # --- 8b. beta-envelope local-stability sweep -------------------------
    # Edge-probes the operating point with short heatbath + short leapfrog
    # at each candidate beta; aggregates only SCALAR summaries (drift max,
    # gauss max, crosscheck gap, regime string) -- never the full G_history
    # or per-face arrays. Strictly capped by ENVELOPE_MAX_SIDECAR_BYTES.
    beta_envelope_history: List[Dict[str, Any]] = []
    if not args.no_cross_check and not args.no_beta_envelope:
        env_betas = [float(x.strip()) for x in args.beta_envelope.split(",") if x.strip()]
        t0 = _phase(f"beta-envelope sweep over {env_betas}")
        rng_state_pre = torch.get_rng_state()  # phase 9 must not be affected
        for b in env_betas:
            print(f"  envelope beta = {b}", flush=True)
            seed_b_heat = args.seed + 50000 + int(round(b * 1000))
            seed_b_E    = args.seed + 60000 + int(round(b * 1000))
            # (c) heatbath
            res = buckyball_heatbath.thermalize(
                graph, b,
                n_sweeps=ENVELOPE_THERM_SWEEPS,
                n_measure=ENVELOPE_MEASURE_SWEEPS,
                n_measure_every=2,
                seed=seed_b_heat,
            )
            P_history_b = list(res["P_history"])
            P_meas_b = float(res["P_mean"])
            P_sem_b = float(res["P_sem"])
            P_exact_b = float(ym_exact.exact_mean_plaquette_su2_buckyball(b))
            migdal_gap_b = abs(P_meas_b - P_exact_b)
            # (d-e) leapfrog
            U_env_b = res["U_final"].detach().clone()
            gen_E_b = torch.Generator(); gen_E_b.manual_seed(seed_b_E)
            E_env_b = buckyball_integrator.initialize_E_canonical(
                graph, b, generator=gen_E_b, project_gauss=True, U=U_env_b,
            )
            H0_b, _, _ = buckyball_integrator.compute_hamiltonian(U_env_b, E_env_b, graph, b)
            env_res = buckyball_observables.integrate_with_states(
                U_env_b, E_env_b,
                dt=args.dt, n_steps=ENVELOPE_LEAPFROG_STEPS, graph=graph, beta=b,
                measure_every=ENVELOPE_MEASURE_EVERY, include_initial=True,
            )
            # (f) drift + max-Gauss; aggregate only
            energy_drift_max = float(max(
                abs(h - H0_b) / abs(H0_b) for h in env_res["H_history"]
            )) if abs(H0_b) > 0 else 0.0
            gauss_max_across_window = float(max(env_res["G_history"]))
            # (g) crosscheck: per-frame plaquette mean from U_traj
            P_micro_per_frame = []
            for U_i in env_res["U_traj"]:
                Uf = buckyball_action.all_face_holonomies(U_i, graph)
                P_micro_per_frame.append(float(Uf[:, 0].mean()))
            P_micro = float(sum(P_micro_per_frame) / max(len(P_micro_per_frame), 1))
            crosscheck_gap = abs(P_meas_b - P_micro)
            # (h) blocking on heatbath for regime
            b_anal = validation_report._flyvbjerg_petersen_blocking(
                np.asarray(P_history_b, dtype=np.float64)
            )
            # (i) Q_surrogate spread across the leapfrog window
            Q_per_frame = [
                float(buckyball_observables.Q_surrogate(U_i, graph))
                for U_i in env_res["U_traj"]
            ]
            sector_proxy_q_std = float(np.std(Q_per_frame, ddof=1)) if len(Q_per_frame) > 1 else 0.0
            # Cap P_history to 200 floats so the sidecar stays under budget.
            P_history_capped = [float(x) for x in P_history_b[-200:]]
            beta_envelope_history.append({
                "beta": float(b),
                "P_meas": P_meas_b,
                "P_meas_sem": P_sem_b,
                "regime": b_anal["regime"],
                "migdal_gap": migdal_gap_b,
                "P_exact": P_exact_b,
                "energy_drift_max": energy_drift_max,
                "gauss_max_across_window": gauss_max_across_window,
                "crosscheck_gap": crosscheck_gap,
                "sector_proxy_q_std": sector_proxy_q_std,
                "P_history": P_history_capped,
            })
            print(f"    drift={energy_drift_max:.2e}  G_max={gauss_max_across_window:.2e}  "
                  f"xcheck={crosscheck_gap:.2e}  MW={migdal_gap_b:.2e}  regime={b_anal['regime']}",
                  flush=True)
        torch.set_rng_state(rng_state_pre)  # restore for phase 9
        env_size = len(json.dumps(beta_envelope_history))
        assert env_size < ENVELOPE_MAX_SIDECAR_BYTES, (
            f"beta_envelope_history JSON would be {env_size} bytes, "
            f"exceeding cap {ENVELOPE_MAX_SIDECAR_BYTES}. Reduce envelope size."
        )
        _done(f"beta-envelope (size {env_size} B / cap {ENVELOPE_MAX_SIDECAR_BYTES})", t0)

    # --- 8c. sector classifier (band discrimination test) ----------------
    # Reads the main-run leapfrog U_traj, bins by observed Q_surrogate into
    # B0/B1/B2, runs an LOO k-NN classifier with Mahalanobis distance, and
    # a label-permutation null. Gate fails on FAIL real-accuracy, FAIL
    # permutation, OR FAIL single-feature ablation (gauge-leak).
    sector_classifier_state: Optional[Dict[str, Any]] = None
    if args.sector_classifier and not args.no_cross_check and not args.no_sector_classifier:
        from inertia_damping import sector_classifier as _sc
        t0 = _phase("sector-classifier (band discrimination)")
        rng_state_pre = torch.get_rng_state()
        sector_classifier_state = _sc.run_classifier_gate(
            graph,
            U_traj=result["U_traj"],
            beta=args.beta,
            k=args.sector_k,
            seed=args.seed + 200000,
        )
        torch.set_rng_state(rng_state_pre)
        # Cap blob size; fail loudly if exceeded.
        blob = json.dumps(sector_classifier_state)
        assert len(blob) < CLASSIFIER_MAX_SIDECAR_BYTES, (
            f"sector_classifier_state JSON would be {len(blob)} bytes, "
            f"exceeding cap {CLASSIFIER_MAX_SIDECAR_BYTES}. Reduce permutation depth."
        )
        print(f"  verdict: {sector_classifier_state.get('verdict')}; "
              f"real_acc = {sector_classifier_state.get('real_accuracy')}; "
              f"p = {sector_classifier_state.get('permutation_p_value')}", flush=True)
        _done("sector-classifier", t0)
    elif args.sector_classifier and args.no_cross_check:
        print("sector classifier skipped: --no-cross-check disables the "
              "heatbath ensemble required for binning", flush=True)

    # --- 9. Write sidecar -------------------------------------------------
    t0 = _phase("write sidecar")
    sidecar_path = run_dir / "final_state.npz"
    sidecar_kwargs: Dict[str, Any] = {
        "U_init": U_init_t.detach().cpu().numpy().astype(np.float64),
        "E_init": E_init_t.detach().cpu().numpy().astype(np.float64),
        "U_final": result["U_final"].detach().cpu().numpy().astype(np.float64),
        "E_final": result["E_final"].detach().cpu().numpy().astype(np.float64),
        "code_commit": code_commit,
        "wall_time_seconds": time.time() - t_total,
        "heatbath_canonical_beta": args.beta,
        "heatbath_canonical_n_thermalization": args.canonical_therm,
    }
    sidecar_kwargs["gauss_history"] = gauss_history_arr
    if canonical_P_history:
        sidecar_kwargs["heatbath_canonical_P_history"] = np.asarray(
            canonical_P_history, dtype=np.float64,
        )
    if beta_scan:
        sidecar_kwargs["beta_scan"] = json.dumps(beta_scan)
    if beta_envelope_history:
        sidecar_kwargs["beta_envelope_history"] = json.dumps(beta_envelope_history)
    if sector_classifier_state is not None:
        sidecar_kwargs["sector_classifier_state"] = json.dumps(sector_classifier_state)
    if cross_P_history:
        sidecar_kwargs["microcanonical_long_P_history"] = np.asarray(
            cross_P_history, dtype=np.float64,
        )
    np.savez(sidecar_path, **sidecar_kwargs)
    print(f"  wrote {sidecar_path}", flush=True)
    _done("write sidecar", t0)

    # --- 10. Generate the report ------------------------------------------
    t0 = _phase("generate_report")
    md_path, json_path = validation_report.generate_report(
        trajectory_path=str(traj_path),
        output_dir=str(run_dir),
        run_metadata={
            "code_commit": code_commit,
            "wall_time_seconds": time.time() - t_total,
        },
        sidecar_path=str(sidecar_path),
        gauge_invariance_seed=args.seed + 7,
    )
    _done("generate_report", t0)

    # --- 11. SHA256SUMS ---------------------------------------------------
    import hashlib
    sums_path = run_dir / "SHA256SUMS"
    with open(sums_path, "w", encoding="utf-8") as fh:
        for p in (traj_path, sidecar_path, md_path, json_path):
            h = hashlib.sha256()
            with open(p, "rb") as g:
                for chunk in iter(lambda: g.read(65536), b""):
                    h.update(chunk)
            fh.write(f"{h.hexdigest()}  {os.path.basename(p)}\n")
    print(f"  wrote {sums_path}", flush=True)

    # --- 12. Surface a "latest" pointer for the demo ----------------------
    latest_dir = Path(args.output) / "latest"
    try:
        if latest_dir.exists() and latest_dir.is_symlink():
            latest_dir.unlink()
        elif latest_dir.exists():
            # On Windows we cannot symlink without privilege; copy instead.
            import shutil
            shutil.rmtree(latest_dir)
        try:
            os.symlink(run_dir, latest_dir, target_is_directory=True)
            print(f"  symlinked {latest_dir} -> {run_dir}", flush=True)
        except (OSError, NotImplementedError):
            import shutil
            shutil.copytree(run_dir, latest_dir)
            print(f"  copied {run_dir} -> {latest_dir}", flush=True)
        # Also write fixed-name pointers the demo button can fetch directly,
        # plus a small manifest the JS can poll without enumerating the dir.
        import shutil
        for src, dst_name in ((md_path, "report.md"), (json_path, "report.json")):
            try:
                shutil.copy(str(latest_dir / src.name), str(latest_dir / dst_name))
            except FileNotFoundError:
                shutil.copy(str(src), str(latest_dir / dst_name))
        manifest = {
            "run_id": run_dir.name,
            "generated_at": ts,
            "md": str((latest_dir / "report.md").as_posix()).split("/inertia_damping/", 1)[-1],
            "json": str((latest_dir / "report.json").as_posix()).split("/inertia_damping/", 1)[-1],
            "wall_time_seconds": time.time() - t_total,
            "code_commit": code_commit,
        }
        with open(latest_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
    except Exception as ex:
        print(f"  WARNING: could not create reports/latest pointer: {ex}", flush=True)

    # --- 13. Verdict summary ----------------------------------------------
    with open(json_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    overall = report["overall_verdict"]
    elapsed_total = time.time() - t_total
    line = (
        f"RESULT: {overall['summary']} (wall time: {elapsed_total:.1f}s)  "
        f"-> {md_path.name}"
    )
    print()
    print(line, flush=True)
    return 0 if overall["passing"] == overall["total"] else 2


if __name__ == "__main__":
    sys.exit(main())
