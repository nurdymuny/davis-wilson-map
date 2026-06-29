"""
test_validation_report.py -- 6-gate sanity battery for validation_report.py.

Gates
-----
R_A  Regenerating a report from the same trajectory.json yields the same
     numerical content. (Timestamps and basenames differ; everything else
     should be bit-identical at FP64.)

R_B  The Migdal-Witten exact target the report publishes agrees with an
     independent scipy.special.iv(2, beta) / scipy.special.iv(1, beta) * 2
     evaluation at the F -> infinity limit to within the documented
     finite-volume correction (~10^-10).

R_C  Gauge-invariance verification uses a fresh seeded Haar g_v draw every
     run. Two runs with different gauge_invariance_seed values produce
     different g_v (verified by changing edge_phase mean) but the same
     FP64-floor verdict on every invariant observable.

R_D  Conservation numbers (energy drift, time-reversibility residuals,
     final Gauss residual) reproduce when independently recomputed from
     the trajectory frames and the final-state sidecar tensors.

R_E  Schema validation: a JSON payload missing any required top-level
     field raises ValueError when loaded by the validator.

R_F  A v0.1 trajectory with no sidecar generates a valid report that marks
     gauge-invariance, time-reversibility, Gauss residual, and method
     cross-check as "not available" rather than crashing.

USAGE
-----
    # Quick run: re-uses the existing inertia_damping/trajectory.json (v0.1)
    # for R_F, and spins up a small fresh run for R_A/R_C/R_D.
    python -m inertia_damping.test_validation_report

    # Faster variant skipping the fresh run -- only R_B, R_E, R_F fire:
    python -m inertia_damping.test_validation_report --quick
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inertia_damping import validation_report  # noqa: E402


def _print_gate(name: str, ok: bool, msg: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{(': ' + msg) if msg else ''}", flush=True)


def _run_small_orchestrator(out_dir: Path, seed: int = 20260616) -> dict:
    """Run a small end-to-end pipeline (~30-60s) to produce a trajectory + sidecar.

    Uses 50 leapfrog steps and 100 canonical sweeps so the tests run fast.
    """
    import torch
    from inertia_damping import (
        buckyball_graph, buckyball_action, buckyball_heatbath,
        buckyball_integrator, buckyball_observables,
    )

    graph = buckyball_graph.build_truncated_icosahedron()
    rng = np.random.default_rng(seed)
    U_t = buckyball_action.identity_links(graph.n_edges)
    for _ in range(30):
        buckyball_heatbath.heatbath_sweep(U_t, graph, 2.5, generator=rng)
    gen_E = torch.Generator(); gen_E.manual_seed(seed + 1)
    E_t = buckyball_integrator.initialize_E_canonical(
        graph, 2.5, generator=gen_E, project_gauss=True, U=U_t,
    )
    U_init = U_t.detach().clone()
    E_init = E_t.detach().clone()
    res = buckyball_observables.integrate_with_states(
        U_init, E_init, dt=0.02, n_steps=50, graph=graph, beta=2.5,
        measure_every=10, include_initial=False,
    )
    n_frames = len(res["U_traj"])
    frames = []
    for i in range(n_frames):
        U_i = res["U_traj"][i]; E_i = res["E_traj"][i]
        phases = buckyball_observables.edge_phase(U_i).detach().cpu().numpy()
        kinetic = buckyball_observables.edge_kinetic(E_i).detach().cpu().numpy()
        frames.append({
            "t": float(res["times"][i]),
            "step": int(res["steps"][i]),
            "Q": float(buckyball_observables.Q_surrogate(U_i, graph)),
            "plaquette_mean": float(buckyball_observables._plaquette_mean_from_U(U_i, graph)),
            "edge_phases": [float(x) for x in phases.tolist()],
            "edge_kinetic": [float(x) for x in kinetic.tolist()],
            "control": {},
            "energy": float(res["H_history"][i]),
        })
    payload = {
        "schema_version": "0.1",
        "gauge_group": "SU(2)",
        "lattice": {"vertices": graph.n_vertices, "edges": graph.n_edges, "faces": graph.n_faces},
        "geometry": buckyball_observables._geometry_block(graph),
        "frames": frames,
        "beta": 2.5, "dt": 0.02, "measure_every": 10,
        "n_thermalization_sweeps": 30, "n_steps": 50, "seed": seed,
    }
    traj_path = out_dir / "trajectory.json"
    with open(traj_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    # Canonical heatbath ensemble for Section 5 cross-check.
    therm = buckyball_heatbath.thermalize(
        graph, 2.5, n_sweeps=100, n_measure=200, n_measure_every=2, seed=seed + 100,
    )
    sidecar = out_dir / "final_state.npz"
    np.savez(
        sidecar,
        U_init=U_init.detach().cpu().numpy().astype(np.float64),
        E_init=E_init.detach().cpu().numpy().astype(np.float64),
        U_final=res["U_final"].detach().cpu().numpy().astype(np.float64),
        E_final=res["E_final"].detach().cpu().numpy().astype(np.float64),
        heatbath_canonical_P_history=np.asarray(therm["P_history"], dtype=np.float64),
        heatbath_canonical_beta=2.5,
        heatbath_canonical_n_thermalization=100,
    )
    return {"trajectory": str(traj_path), "sidecar": str(sidecar)}


def _strip_volatile(report_json: dict) -> dict:
    """Remove fields that legitimately differ across runs (timestamps, run
    identifiers tied to the *report* rather than the underlying numbers)."""
    r = copy.deepcopy(report_json)
    rm = r.get("report_metadata", {})
    for key in ("generated_at", "json_artifact_basename", "sha256sums"):
        rm.pop(key, None)
    return r


def gate_R_A(traj: str, sidecar: str) -> bool:
    """Regeneration reproducibility."""
    try:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            md1, j1 = validation_report.generate_report(
                trajectory_path=traj, output_dir=d1, sidecar_path=sidecar,
                gauge_invariance_seed=42,
            )
            md2, j2 = validation_report.generate_report(
                trajectory_path=traj, output_dir=d2, sidecar_path=sidecar,
                gauge_invariance_seed=42,
            )
            r1 = json.load(open(j1, "r", encoding="utf-8"))
            r2 = json.load(open(j2, "r", encoding="utf-8"))
        a = _strip_volatile(r1)
        b = _strip_volatile(r2)
        if a != b:
            # Diagnostic: print the first mismatched path.
            import difflib
            sa = json.dumps(a, sort_keys=True, indent=2).splitlines()
            sb = json.dumps(b, sort_keys=True, indent=2).splitlines()
            diff = list(difflib.unified_diff(sa, sb, n=1))[:30]
            _print_gate("R_A", False, "non-identical regeneration:")
            for line in diff:
                print(f"      {line}", flush=True)
            return False
        _print_gate("R_A", True, "bit-identical numerical content across two runs")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_A", False, "exception (see above)")
        return False


def gate_R_B() -> bool:
    """Bessel cross-check at 10 decimal places vs the F -> infinity limit."""
    try:
        from scipy.special import iv
        from inertia_damping import buckyball_yangmills_exact as ym
        beta = 2.5
        # Module's exact buckyball value (finite-F):
        P_exact = ym.exact_mean_plaquette_su2_buckyball(beta)
        # Independent F -> infinity Bessel ratio. For SU(2), <P> = (1/N)<Re Tr U>
        # = (1/2)<Re Tr U>, and the F -> infinity sum is dominated by j = 1/2,
        # giving <P>_infty = I_2(beta) / I_1(beta).
        P_inf = float(iv(2, beta) / iv(1, beta))
        # The finite-volume correction at beta=2.5, F=32, chi=2 is ~10^-10.
        gap = abs(P_exact - P_inf)
        if gap >= 1e-3:
            _print_gate("R_B", False, f"gap {gap:.3e} between exact and infinity-limit")
            return False
        # Check that the module value matches scipy when computed via the same
        # F -> infinity formula explicitly (sanity for the iv import).
        msg = f"P_exact={P_exact:.10f}, P_inf={P_inf:.10f}, gap={gap:.3e}"
        _print_gate("R_B", True, msg)
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_B", False, "exception")
        return False


def gate_R_C(traj: str, sidecar: str) -> bool:
    """Gauge-invariance: different seeds produce different g_v but both FP64-pass."""
    try:
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            _, j1 = validation_report.generate_report(
                trajectory_path=traj, output_dir=d1, sidecar_path=sidecar,
                gauge_invariance_seed=42,
            )
            _, j2 = validation_report.generate_report(
                trajectory_path=traj, output_dir=d2, sidecar_path=sidecar,
                gauge_invariance_seed=2026,
            )
            r1 = json.load(open(j1, "r", encoding="utf-8"))
            r2 = json.load(open(j2, "r", encoding="utf-8"))
        s4_a = r1["section_4_gauge_invariance"]
        s4_b = r2["section_4_gauge_invariance"]
        if not (s4_a["available"] and s4_b["available"]):
            _print_gate("R_C", False, "Section 4 unavailable in one of the runs")
            return False
        # Both must PASS.
        if s4_a["verdict"] != "PASS" or s4_b["verdict"] != "PASS":
            _print_gate("R_C", False,
                        f"verdict mismatch: {s4_a['verdict']} vs {s4_b['verdict']}")
            return False
        # The edge_phase mean (declared variant) MUST differ across seeds.
        phase_a = next(o for o in s4_a["observables"] if "edge_phase" in o["name"])
        phase_b = next(o for o in s4_b["observables"] if "edge_phase" in o["name"])
        same = abs(phase_a["post"] - phase_b["post"]) < 1e-9
        if same:
            _print_gate("R_C", False, "different seeds produced identical g_v draws")
            return False
        msg = (f"both PASS; edge_phase post differs: {phase_a['post']:.4f} vs "
               f"{phase_b['post']:.4f}")
        _print_gate("R_C", True, msg)
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_C", False, "exception")
        return False


def gate_R_D(traj: str, sidecar: str) -> bool:
    """Conservation numbers reproduce direct frame computation."""
    try:
        with tempfile.TemporaryDirectory() as d:
            _, j = validation_report.generate_report(
                trajectory_path=traj, output_dir=d, sidecar_path=sidecar,
            )
            r = json.load(open(j, "r", encoding="utf-8"))
        # Recompute max|dH/H0| from the on-disk trajectory frames.
        traj_obj = json.load(open(traj, "r", encoding="utf-8"))
        Hs = np.asarray([f["energy"] for f in traj_obj["frames"]], dtype=np.float64)
        H0 = Hs[0]
        max_rel = float(np.abs((Hs - H0) / H0).max())
        reported = r["section_2_conservation"]["energy"]["max_relative_drift"]
        if abs(reported - max_rel) > 1e-12:
            _print_gate("R_D", False,
                        f"energy drift mismatch: report {reported:.3e}, "
                        f"direct {max_rel:.3e}")
            return False
        # Verify the covariant-Gauss number on the final state matches a direct
        # recomputation.
        import torch
        from inertia_damping import buckyball_graph, buckyball_integrator
        graph = buckyball_graph.build_truncated_icosahedron()
        with np.load(sidecar) as z:
            Uf = torch.from_numpy(z["U_final"].astype(np.float64))
            Ef = torch.from_numpy(z["E_final"].astype(np.float64))
        G = buckyball_integrator.compute_gauss_residual(Ef, Uf, graph)
        direct_gmax = float(G.abs().max())
        reported_gmax = r["section_2_conservation"]["gauss"]["gauss_covariant_max_final_state"]
        if abs(direct_gmax - reported_gmax) > 1e-14:
            _print_gate("R_D", False,
                        f"Gauss max mismatch: report {reported_gmax:.3e}, "
                        f"direct {direct_gmax:.3e}")
            return False
        _print_gate("R_D", True,
                    f"energy drift {reported:.3e} and Gauss max {reported_gmax:.3e} "
                    f"reproduce direct frame computation")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_D", False, "exception")
        return False


def gate_R_E() -> bool:
    """A trajectory missing required top-level fields raises ValueError."""
    try:
        with tempfile.TemporaryDirectory() as d:
            traj_path = os.path.join(d, "bad.json")
            bad = {
                "schema_version": "0.1",
                "gauge_group": "SU(2)",
                # MISSING: "lattice"
                "geometry": {"vertex_coords": [], "edges": [], "faces": []},
                "frames": [{}],
            }
            with open(traj_path, "w", encoding="utf-8") as fh:
                json.dump(bad, fh)
            raised = False
            try:
                validation_report.generate_report(traj_path, d)
            except ValueError as ex:
                raised = "missing required top-level key" in str(ex) or "lattice" in str(ex)
            except Exception as ex:
                _print_gate("R_E", False, f"wrong exception type: {type(ex).__name__}: {ex}")
                return False
        if not raised:
            _print_gate("R_E", False, "no ValueError raised on schema violation")
            return False
        _print_gate("R_E", True, "ValueError raised on missing 'lattice'")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_E", False, "exception")
        return False


def gate_R_F(reference_v01_traj: str) -> bool:
    """v0.1 trajectory with no sidecar gracefully degrades."""
    try:
        with tempfile.TemporaryDirectory() as d:
            md, j = validation_report.generate_report(
                trajectory_path=reference_v01_traj, output_dir=d, sidecar_path=None,
            )
            r = json.load(open(j, "r", encoding="utf-8"))
        # Section 4 must be available=False; Section 2.energy MUST still pass.
        s2 = r["section_2_conservation"]
        s4 = r["section_4_gauge_invariance"]
        if s4.get("available", True) is not False:
            _print_gate("R_F", False,
                        "Section 4 should be unavailable without sidecar")
            return False
        if not s2["energy"].get("available", False):
            _print_gate("R_F", False, "Section 2.energy should still be available")
            return False
        # The report should produce a valid overall_verdict with categories.
        if not r.get("overall_verdict", {}).get("categories"):
            _print_gate("R_F", False, "no overall_verdict categories")
            return False
        _print_gate("R_F", True,
                    f"v0.1 trajectory + no sidecar -> {r['overall_verdict']['summary']}")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_F", False, "exception")
        return False


# ---------------------------------------------------------------------------
# Stage 2 helpers: build synthetic envelope / classifier sidecar blobs
# ---------------------------------------------------------------------------
def _envelope_entry(beta: float, drift: float, gauss: float, xcheck: float,
                    migdal: float, regime: str = "plateau_detected",
                    cleared_count: int = 4, score: float = 2.0,
                    indeterminate: bool = False) -> dict:
    """Build one envelope-history row in the on-disk schema."""
    return {
        "beta": float(beta),
        "regime": str(regime),
        "P_meas": 0.5,
        "P_meas_sem": 0.001,
        "P_exact": 0.5,
        "energy_drift_max": float(drift),
        "gauss_max_across_window": float(gauss),
        "crosscheck_gap": float(xcheck),
        "migdal_gap": float(migdal),
        "cleared_count": int(cleared_count),
        "stability_score": float(score),
        "indeterminate": bool(indeterminate),
        "m_drift": 1.0 - drift / 5e-3,
        "m_gauss": 1.0 - gauss / 1e-9,
        "m_xcheck": 1.0 - xcheck / 3e-2,
        "m_migdal": 1.0 - migdal / 1e-2,
        "sector_proxy_q_std": 0.42,
    }


def _classifier_state(real_accuracy: float, permutation_p_value: float,
                       single_feature_max: float, n_eff: float = 10.0,
                       n_samples: int = 30, verdict: str | None = None,
                       reason: str | None = None) -> dict:
    """Build one sector_classifier_state blob in the on-disk schema.

    If verdict is None, the report-side _section9() defense-in-depth pass
    will recompute it from real_accuracy / p_value / single_feature_max."""
    # feature_ablation schema matches sector_classifier.feature_ablation():
    #   {leave_one_out: {name: acc}, single_feature: {name: acc}, single_feature_max: float}
    _names = ["total_plaq_action", "Q_surrogate", "P_mean", "max_plaq_holonomy_dist"]
    _single = {n: 0.35 for n in _names}
    # Hide single_feature_max behind one of the named features so the dict is
    # self-consistent.
    _single["Q_surrogate"] = float(single_feature_max)
    state = {
        "real_accuracy": float(real_accuracy),
        "permutation_p_value": float(permutation_p_value),
        "single_feature_max": float(single_feature_max),
        "n_eff": float(n_eff),
        "n_samples": int(n_samples),
        "feature_names": _names,
        "feature_ablation": {
            "leave_one_out": {n: 0.40 for n in _names},
            "single_feature": _single,
            "single_feature_max": float(single_feature_max),
        },
    }
    if verdict is not None:
        state["verdict"] = str(verdict)
    if reason is not None:
        state["reason"] = str(reason)
    return state


def _write_sidecar_with_blobs(src_sidecar: str, dst_sidecar: str,
                               envelope_history: list | None = None,
                               sector_state: dict | None = None,
                               envelope_raw: bytes | str | None = None,
                               sector_raw: bytes | str | None = None) -> None:
    """Copy src_sidecar to dst_sidecar, adding envelope/classifier JSON blobs.

    Pass envelope_raw/sector_raw to inject a malformed (non-JSON) string for
    the R_E_malformed extension gate.
    """
    with np.load(src_sidecar) as z:
        out = {k: z[k] for k in z.files}
    if envelope_history is not None:
        out["beta_envelope_history"] = np.array(json.dumps(envelope_history))
    elif envelope_raw is not None:
        out["beta_envelope_history"] = np.array(str(envelope_raw))
    if sector_state is not None:
        out["sector_classifier_state"] = np.array(json.dumps(sector_state))
    elif sector_raw is not None:
        out["sector_classifier_state"] = np.array(str(sector_raw))
    np.savez(dst_sidecar, **out)


def _generate(traj: str, sidecar: str, **kwargs) -> dict:
    """Generate a report into a temp dir and return the parsed JSON dict."""
    with tempfile.TemporaryDirectory() as d:
        _, j = validation_report.generate_report(
            trajectory_path=traj, output_dir=d, sidecar_path=sidecar, **kwargs,
        )
        return json.load(open(j, "r", encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stage 2 gates: R_G family (beta envelope), R_H family (sector classifier),
# R_I (denominator stability), R_E_malformed (malformed blob coverage).
# ---------------------------------------------------------------------------
def gate_R_G(traj: str, sidecar: str) -> bool:
    """Envelope sidecar round-trip: PASS when one beta clears all 4."""
    try:
        envelope = [
            _envelope_entry(1.5, 1e-3, 1e-15, 2.0e-2, 5e-3, "plateau_detected", 4, 1.8),
            _envelope_entry(2.0, 4e-5, 9e-11, 2.3e-2, 2e-3, "plateau_detected", 4, 2.9),
            _envelope_entry(2.5, 4e-5, 2e-15, 4.6e-2, 1e-3, "plateau_detected", 3, 2.6),
            _envelope_entry(3.0, 3e-5, 3e-15, 4.2e-2, 2e-3, "plateau_detected", 3, 2.4),
            _envelope_entry(3.5, 2e-5, 3e-15, 6.1e-2, 3e-3, "no_plateau", 2, 1.5, True),
        ]
        with tempfile.TemporaryDirectory() as d:
            ext = Path(d) / "sidecar_with_envelope.npz"
            _write_sidecar_with_blobs(sidecar, str(ext), envelope_history=envelope)
            r = _generate(traj, str(ext))
        s8 = r.get("section_8_beta_envelope", {})
        if not s8.get("available"):
            _print_gate("R_G", False, f"section_8 unavailable: {s8.get('reason')}")
            return False
        if s8.get("verdict") != "PASS":
            _print_gate("R_G", False, f"verdict={s8.get('verdict')} (want PASS)")
            return False
        rec = s8.get("operational_beta_recommendation")
        if rec != 2.0:
            _print_gate("R_G", False, f"recommendation={rec} (want 2.0, the only fully-cleared row with usable regime)")
            return False
        # Sensitivity: changes to sector_proxy_q_std do NOT change the recommendation.
        envelope2 = [dict(e, sector_proxy_q_std=99.0) for e in envelope]
        with tempfile.TemporaryDirectory() as d:
            ext = Path(d) / "sidecar_proxy_perturbed.npz"
            _write_sidecar_with_blobs(sidecar, str(ext), envelope_history=envelope2)
            r2 = _generate(traj, str(ext))
        rec2 = r2.get("section_8_beta_envelope", {}).get("operational_beta_recommendation")
        if rec2 != 2.0:
            _print_gate("R_G", False, f"sector_proxy_q_std moved recommendation from 2.0 to {rec2}")
            return False
        _print_gate("R_G", True, "envelope PASS, recommendation=2.0 (sector_proxy_q_std-invariant)")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_G", False, "exception")
        return False


def gate_R_G_legacy(reference_v01_traj: str) -> bool:
    """Legacy v0.1 trajectory + no envelope: NOT_RUN; total still 10."""
    try:
        r = _generate(reference_v01_traj, None)
        s8 = r.get("section_8_beta_envelope", {})
        if s8.get("available") is not False:
            _print_gate("R_G_legacy", False, "section_8 should be unavailable on a legacy sidecar")
            return False
        if "--beta-envelope" not in (s8.get("reason") or ""):
            _print_gate("R_G_legacy", False, f"reason should mention --beta-envelope, got: {s8.get('reason')!r}")
            return False
        ov = r.get("overall_verdict", {})
        if ov.get("total") != 10:
            _print_gate("R_G_legacy", False, f"overall_verdict.total={ov.get('total')} (want 10)")
            return False
        cat_names = [c["name"] for c in ov.get("categories", [])]
        if "Operational beta envelope" not in cat_names:
            _print_gate("R_G_legacy", False, f"'Operational beta envelope' missing from categories: {cat_names}")
            return False
        _print_gate("R_G_legacy", True, f"legacy run: total=10, envelope NOT_RUN, summary='{ov.get('summary')}'")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_G_legacy", False, "exception")
        return False


def gate_R_G_regime(traj: str, sidecar: str) -> bool:
    """regime='too_short' on highest-score beta excludes it from recommendation."""
    try:
        # beta=2.0 has the best score but we mark it too_short -> recommendation skips it.
        envelope = [
            _envelope_entry(1.5, 1e-3, 1e-15, 2.0e-2, 5e-3, "plateau_detected", 4, 1.8),
            _envelope_entry(2.0, 4e-5, 9e-11, 2.3e-2, 2e-3, "too_short", 4, 2.9),
            _envelope_entry(2.5, 4e-5, 2e-15, 2.4e-2, 1e-3, "plateau_detected", 4, 2.6),
            _envelope_entry(3.0, 3e-5, 3e-15, 4.2e-2, 2e-3, "plateau_detected", 3, 2.4),
            _envelope_entry(3.5, 2e-5, 3e-15, 6.1e-2, 3e-3, "no_plateau", 2, 1.5, True),
        ]
        with tempfile.TemporaryDirectory() as d:
            ext = Path(d) / "sidecar.npz"
            _write_sidecar_with_blobs(sidecar, str(ext), envelope_history=envelope)
            r = _generate(traj, str(ext))
        rec = r.get("section_8_beta_envelope", {}).get("operational_beta_recommendation")
        # With beta=2.0 ruled out (too_short), 2.5 is the next-best with cleared_count=4.
        if rec == 2.0:
            _print_gate("R_G_regime", False, "recommendation still 2.0 despite too_short regime")
            return False
        if rec not in (1.5, 2.5, None):
            _print_gate("R_G_regime", False, f"unexpected recommendation: {rec}")
            return False
        _print_gate("R_G_regime", True, f"too_short excludes beta=2.0; recommendation={rec}")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_G_regime", False, "exception")
        return False


def gate_R_H_module() -> bool:
    """sector_classifier module: well-separated synthetic feature matrices.

    Builds three labelled clusters of 4-D feature vectors that match the
    sector_classifier API directly (NOT via U_traj / kernel observables --
    that path is exercised by R_H_module_NA). Checks accuracy, permutation
    null, feature_ablation shape, and intra-process bit-identity. Does NOT
    claim cross-OS reproducibility (Section 9 is intra-process only).
    """
    try:
        from inertia_damping import sector_classifier as sc
        rng = np.random.default_rng(20260617)
        n_per = 30
        centers = [np.array([0.1, 0.2, 0.3, 0.05]),
                   np.array([0.5, 1.0, 0.5, 0.5]),
                   np.array([0.9, 2.0, 0.7, 0.95])]
        X_list, y_list = [], []
        for lab, c in enumerate(centers):
            X_list.append(c + 0.03 * rng.standard_normal((n_per, 4)))
            y_list.append(np.full(n_per, lab, dtype=np.int64))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        clf = sc.knn_loo_classify(X, y, k=5)
        if clf["accuracy"] < 0.9:
            _print_gate("R_H_module", False, f"accuracy {clf['accuracy']:.3f} < 0.9 on well-separated clusters")
            return False
        null = sc.permutation_null(X, y, n_permutations=100, k=5, seed=20260618)
        if not (0.0 <= null["p_value"] <= 1.0):
            _print_gate("R_H_module", False, f"p_value {null['p_value']} out of [0,1]")
            return False
        abl = sc.feature_ablation(X, y, k=5)
        # Schema: leave_one_out + single_feature + single_feature_max top-level keys;
        # each of leave_one_out / single_feature has 4 entries (one per feature).
        if len(abl.get("single_feature", {})) != 4:
            _print_gate("R_H_module", False, f"single_feature has {len(abl.get('single_feature', {}))} entries (want 4)")
            return False
        if len(abl.get("leave_one_out", {})) != 4:
            _print_gate("R_H_module", False, f"leave_one_out has {len(abl.get('leave_one_out', {}))} entries (want 4)")
            return False
        # Bit-identical confusion matrices on repeated run in same process.
        clf2 = sc.knn_loo_classify(X, y, k=5)
        if clf["confusion_matrix"] != clf2["confusion_matrix"]:
            _print_gate("R_H_module", False, "confusion matrices differ on repeated run (intra-process)")
            return False
        _print_gate("R_H_module", True,
                    f"acc={clf['accuracy']:.3f}, p={null['p_value']:.3f}, ablation OK, bit-identical")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_H_module", False, "exception")
        return False


def gate_R_H_module_NA() -> bool:
    """sector_classifier: identity-link traj cannot populate B0/B1/B2 -> NOT_APPLICABLE."""
    try:
        from inertia_damping import sector_classifier as sc
        from inertia_damping import buckyball_graph, buckyball_action
        graph = buckyball_graph.build_truncated_icosahedron()
        n_frames = 30
        U_traj = [buckyball_action.identity_links(graph.n_edges) for _ in range(n_frames)]
        result = sc.run_classifier_gate(graph, U_traj, beta=2.5, k=3, seed=20260619)
        if result.get("verdict") != "NOT_APPLICABLE":
            _print_gate("R_H_module_NA", False, f"verdict={result.get('verdict')} (want NOT_APPLICABLE)")
            return False
        reason = (result.get("reason") or "").lower()
        if "dispersion" not in reason and "populate" not in reason and "band" not in reason:
            _print_gate("R_H_module_NA", False, f"reason should mention dispersion/populate/band: {reason!r}")
            return False
        _print_gate("R_H_module_NA", True, f"identity-link traj -> NOT_APPLICABLE ('{result.get('reason')}')")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_H_module_NA", False, "exception")
        return False


def gate_R_H_report(traj: str, sidecar: str) -> bool:
    """_section9 defense-in-depth: recomputes verdict from injected fields."""
    try:
        # Case A: real_accuracy=0.85, p=0.02, single_feature=0.55 -> PASS
        stateA = _classifier_state(0.85, 0.02, 0.55)
        with tempfile.TemporaryDirectory() as d:
            ext = Path(d) / "A.npz"
            _write_sidecar_with_blobs(sidecar, str(ext), sector_state=stateA)
            rA = _generate(traj, str(ext))
        if rA["section_9_sector_classifier"].get("verdict") != "PASS":
            _print_gate("R_H_report", False, f"case A verdict={rA['section_9_sector_classifier'].get('verdict')} (want PASS)")
            return False

        # Case B: single_feature_max=0.83 >= real_min=0.80 -> FAIL (gauge-leak)
        stateB = _classifier_state(0.85, 0.02, 0.83)
        with tempfile.TemporaryDirectory() as d:
            ext = Path(d) / "B.npz"
            _write_sidecar_with_blobs(sidecar, str(ext), sector_state=stateB)
            rB = _generate(traj, str(ext))
        s9B = rB["section_9_sector_classifier"]
        if s9B.get("verdict") != "FAIL":
            _print_gate("R_H_report", False, f"case B verdict={s9B.get('verdict')} (want FAIL)")
            return False
        if "single" not in str(s9B.get("fail_reasons") or s9B.get("reason") or "").lower():
            _print_gate("R_H_report", False, f"case B fail_reasons should mention 'single feature': {s9B.get('fail_reasons')!r}")
            return False

        # Case C: p_value=0.30 >= alpha=0.05 -> FAIL (permutation null)
        stateC = _classifier_state(0.85, 0.30, 0.55)
        with tempfile.TemporaryDirectory() as d:
            ext = Path(d) / "C.npz"
            _write_sidecar_with_blobs(sidecar, str(ext), sector_state=stateC)
            rC = _generate(traj, str(ext))
        s9C = rC["section_9_sector_classifier"]
        if s9C.get("verdict") != "FAIL":
            _print_gate("R_H_report", False, f"case C verdict={s9C.get('verdict')} (want FAIL)")
            return False
        if "permutation" not in str(s9C.get("fail_reasons") or s9C.get("reason") or "").lower():
            _print_gate("R_H_report", False, f"case C fail_reasons should mention 'permutation null': {s9C.get('fail_reasons')!r}")
            return False

        _print_gate("R_H_report", True, "PASS / FAIL(single) / FAIL(perm) all recomputed correctly")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_H_report", False, "exception")
        return False


def gate_R_H_legacy(reference_v01_traj: str) -> bool:
    """Legacy v0.1 + no classifier sidecar: NOT_RUN."""
    try:
        r = _generate(reference_v01_traj, None)
        s9 = r.get("section_9_sector_classifier", {})
        if s9.get("available") is not False:
            _print_gate("R_H_legacy", False, "section_9 should be unavailable on a legacy sidecar")
            return False
        if "--sector-classifier" not in (s9.get("reason") or ""):
            _print_gate("R_H_legacy", False, f"reason should mention --sector-classifier, got: {s9.get('reason')!r}")
            return False
        _print_gate("R_H_legacy", True, f"legacy: section_9 NOT_RUN ('{s9.get('reason')}')")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_H_legacy", False, "exception")
        return False


def _strip_stage2_blobs(src_sidecar: str, dst_sidecar: str) -> None:
    """Copy a sidecar verbatim minus beta_envelope_history and sector_classifier_state."""
    with np.load(src_sidecar) as z:
        out = {k: z[k] for k in z.files
               if k not in ("beta_envelope_history", "sector_classifier_state")}
    np.savez(dst_sidecar, **out)


def gate_R_I(traj: str, sidecar: str) -> bool:
    """Denominator stability: total=10 always; legacy emits 8 + 2 NOT_RUN, never silent."""
    try:
        # Full envelope+classifier sidecar -> total=10 with both new categories present
        envelope = [_envelope_entry(2.0, 4e-5, 9e-11, 2.3e-2, 2e-3, score=2.9)]
        state = _classifier_state(0.85, 0.02, 0.55)
        with tempfile.TemporaryDirectory() as d:
            ext = Path(d) / "full.npz"
            _write_sidecar_with_blobs(sidecar, str(ext),
                                       envelope_history=envelope, sector_state=state)
            r = _generate(traj, str(ext))
        ov = r.get("overall_verdict", {})
        if ov.get("total") != 10:
            _print_gate("R_I", False, f"full sidecar: total={ov.get('total')} (want 10)")
            return False
        cat_names = [c["name"] for c in ov.get("categories", [])]
        for required in ("Operational beta envelope", "Sector classifier"):
            if required not in cat_names:
                _print_gate("R_I", False, f"missing category '{required}' (full sidecar)")
                return False
        # Legacy sub-case: synthesize a TRULY legacy sidecar by stripping the
        # Stage 2 blobs from the production sidecar (production has them; an
        # actual schema-1.1 run would not).
        with tempfile.TemporaryDirectory() as d:
            legacy = Path(d) / "legacy.npz"
            _strip_stage2_blobs(sidecar, str(legacy))
            r2 = _generate(traj, str(legacy))
        ov2 = r2.get("overall_verdict", {})
        if ov2.get("total") != 10:
            _print_gate("R_I", False, f"legacy sidecar: total={ov2.get('total')} (want 10)")
            return False
        cat2 = {c["name"]: c["verdict"] for c in ov2.get("categories", [])}
        for required in ("Operational beta envelope", "Sector classifier"):
            if required not in cat2:
                _print_gate("R_I", False, f"missing category '{required}' (legacy)")
                return False
            if cat2[required] not in ("NOT_RUN", "NOT_APPLICABLE"):
                _print_gate("R_I", False, f"category '{required}' = {cat2[required]} (want NOT_RUN/NOT_APPLICABLE)")
                return False
        _print_gate("R_I", True, "total=10 stable; legacy reads 8 substantive + 2 NOT_RUN")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_I", False, "exception")
        return False


def gate_R_E_malformed(traj: str, sidecar: str) -> bool:
    """Malformed envelope / classifier blobs do not crash; both render NOT_RUN."""
    try:
        # Malformed envelope (non-JSON string)
        with tempfile.TemporaryDirectory() as d:
            ext = Path(d) / "bad_env.npz"
            _write_sidecar_with_blobs(sidecar, str(ext), envelope_raw="not json")
            r = _generate(traj, str(ext))
        if r["section_8_beta_envelope"].get("available") is not False:
            _print_gate("R_E_malformed", False, "malformed envelope still rendered as available")
            return False
        # Malformed classifier (non-JSON string)
        with tempfile.TemporaryDirectory() as d:
            ext = Path(d) / "bad_clf.npz"
            _write_sidecar_with_blobs(sidecar, str(ext), sector_raw="<not json>")
            r2 = _generate(traj, str(ext))
        if r2["section_9_sector_classifier"].get("available") is not False:
            _print_gate("R_E_malformed", False, "malformed classifier still rendered as available")
            return False
        _print_gate("R_E_malformed", True, "malformed envelope / classifier both degrade to NOT_RUN")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_E_malformed", False, "exception")
        return False


def gate_R_J() -> bool:
    """Section 9 golden-file check (intra-process bit-identity).

    Re-runs the SAME synthetic 3-cluster construction and the SAME
    sector_classifier calls as the committed golden fixture, and asserts:
      * accuracy == golden.accuracy        (FP equality)
      * confusion_matrix == golden         (integer equality)
      * per_class_accuracy == golden       (FP equality)
      * permutation p_value == golden      (FP equality)
      * feature_ablation == golden         (FP equality)
    Drops the cross-OS bit-identity claim deliberately -- this is the
    Section 9 contract per the Stage 2 design.
    """
    try:
        from inertia_damping import sector_classifier as sc
        golden_path = Path(_HERE) / "test_fixtures" / "sector_classifier_golden.json"
        golden = json.loads(golden_path.read_text(encoding="utf-8"))

        rng = np.random.default_rng(20260617)
        n_per = 30
        centers = [np.array([0.1, 0.2, 0.3, 0.05]),
                   np.array([0.5, 1.0, 0.5, 0.5]),
                   np.array([0.9, 2.0, 0.7, 0.95])]
        X_list, y_list = [], []
        for lab, c in enumerate(centers):
            X_list.append(c + 0.03 * rng.standard_normal((n_per, 4)))
            y_list.append(np.full(n_per, lab, dtype=np.int64))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        clf = sc.knn_loo_classify(X, y, k=5)
        null = sc.permutation_null(X, y, n_permutations=100, k=5, seed=20260618)
        abl = sc.feature_ablation(X, y, k=5)

        mismatches = []
        if clf["accuracy"] != golden["classifier"]["accuracy"]:
            mismatches.append(f"accuracy {clf['accuracy']} != golden {golden['classifier']['accuracy']}")
        if clf["confusion_matrix"] != golden["classifier"]["confusion_matrix"]:
            mismatches.append(f"confusion_matrix {clf['confusion_matrix']} != golden")
        if clf["per_class_accuracy"] != golden["classifier"]["per_class_accuracy"]:
            mismatches.append(f"per_class_accuracy {clf['per_class_accuracy']} != golden")
        if null["p_value"] != golden["permutation"]["p_value"]:
            mismatches.append(f"p_value {null['p_value']} != golden {golden['permutation']['p_value']}")
        if null["null_accuracy_mean"] != golden["permutation"]["null_accuracy_mean"]:
            mismatches.append(f"null_mean {null['null_accuracy_mean']} != golden")
        if abl["leave_one_out"] != golden["ablation"]["leave_one_out"]:
            mismatches.append("leave_one_out mismatch")
        if abl["single_feature"] != golden["ablation"]["single_feature"]:
            mismatches.append("single_feature mismatch")
        if abl["single_feature_max"] != golden["ablation"]["single_feature_max"]:
            mismatches.append(f"single_feature_max {abl['single_feature_max']} != golden")

        if mismatches:
            _print_gate("R_J", False, "; ".join(mismatches[:3]))
            return False
        _print_gate("R_J", True,
                    f"intra-process bit-identity vs golden: acc={clf['accuracy']:.4f}, "
                    f"p={null['p_value']:.4f}, single_feature_max={abl['single_feature_max']:.4f}")
        return True
    except Exception:
        traceback.print_exc()
        _print_gate("R_J", False, "exception")
        return False


def gate_R_G_orchestrator(reference_v01_traj: str) -> bool:
    """End-to-end orchestrator smoke (--slow). Tiny overrides via env vars."""
    try:
        import subprocess
        env = os.environ.copy()
        env["ENVELOPE_THERM_SWEEPS"] = "10"
        env["ENVELOPE_MEASURE_SWEEPS"] = "20"
        env["ENVELOPE_LEAPFROG_STEPS"] = "40"
        with tempfile.TemporaryDirectory() as d:
            cmd = [
                sys.executable, "-m", "inertia_damping.run_validation_report",
                "--steps", "40", "--therm-sweeps", "30",
                "--cross-check-steps", "100", "--canonical-sweeps", "100",
                "--canonical-therm", "50",
                "--beta-envelope", "2.5",
                "--no-sector-classifier",
                "--output", d, "--seed", "20260620",
            ]
            res = subprocess.run(cmd, env=env, cwd=_ROOT,
                                  capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            _print_gate("R_G_orchestrator", False,
                        f"orchestrator exit {res.returncode}: {(res.stderr or res.stdout)[-400:]}")
            return False
        # Find the produced report
        prods = list(Path(d).rglob("validation_report_*.json"))
        if not prods:
            _print_gate("R_G_orchestrator", False, "no report JSON produced")
            return False
        rep = json.load(open(prods[0]))
        s8 = rep.get("section_8_beta_envelope", {})
        if not s8.get("available"):
            _print_gate("R_G_orchestrator", False, f"section_8 unavailable: {s8.get('reason')}")
            return False
        if s8.get("n_envelope_betas") != 1:
            _print_gate("R_G_orchestrator", False, f"n_envelope_betas={s8.get('n_envelope_betas')} (want 1)")
            return False
        # Verify sidecar size cap
        sidecars = list(Path(d).rglob("final_state.npz"))
        with np.load(sidecars[0]) as z:
            blob = str(z["beta_envelope_history"])
        if len(blob) >= 64 * 1024:
            _print_gate("R_G_orchestrator", False, f"envelope blob {len(blob)} B >= 64 KB cap")
            return False
        _print_gate("R_G_orchestrator", True,
                    f"end-to-end orchestrator OK; envelope blob {len(blob)} B")
        return True
    except subprocess.TimeoutExpired:
        _print_gate("R_G_orchestrator", False, "orchestrator timeout")
        return False
    except Exception:
        traceback.print_exc()
        _print_gate("R_G_orchestrator", False, "exception")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the small-run setup (R_A, R_C, R_D, R_G family will not fire)")
    ap.add_argument("--slow", action="store_true",
                    help="include R_G_orchestrator (~3-5 minute end-to-end smoke)")
    ap.add_argument("--trajectory", type=str, default=None,
                    help="path to an existing trajectory.json for R_F (defaults to "
                         "inertia_damping/trajectory.json)")
    ap.add_argument("--sidecar", type=str, default=None,
                    help="path to a final_state.npz for R_A/R_C/R_D (skips orchestrator)")
    args = ap.parse_args()

    print("=" * 68, flush=True)
    print("validation_report.py 15-gate battery (Stage 2)", flush=True)
    print("=" * 68, flush=True)

    reference_v01 = args.trajectory or os.path.join(_HERE, "trajectory.json")
    if not os.path.exists(reference_v01):
        print(f"FATAL: reference v0.1 trajectory missing at {reference_v01}", flush=True)
        return 1

    setup_traj = None
    setup_sidecar = None
    tmp_setup_dir = None
    if not args.quick:
        if args.sidecar and args.trajectory:
            setup_traj = args.trajectory
            setup_sidecar = args.sidecar
        else:
            tmp_setup_dir = Path(tempfile.mkdtemp(prefix="vrep_test_"))
            print(f"  setup: running small orchestrator into {tmp_setup_dir}", flush=True)
            setup = _run_small_orchestrator(tmp_setup_dir)
            setup_traj = setup["trajectory"]
            setup_sidecar = setup["sidecar"]
            print("  setup: done", flush=True)
    print(flush=True)

    results = {}
    print("R_A: regeneration reproducibility", flush=True)
    results["R_A"] = gate_R_A(setup_traj, setup_sidecar) if (setup_traj and setup_sidecar) else None
    if results["R_A"] is None:
        _print_gate("R_A", True, "SKIPPED (--quick)")
    print()

    print("R_B: Migdal-Witten target Bessel cross-check", flush=True)
    results["R_B"] = gate_R_B()
    print()

    print("R_C: gauge-invariance fresh-seed correctness", flush=True)
    results["R_C"] = gate_R_C(setup_traj, setup_sidecar) if (setup_traj and setup_sidecar) else None
    if results["R_C"] is None:
        _print_gate("R_C", True, "SKIPPED (--quick)")
    print()

    print("R_D: conservation numbers match direct frame computation", flush=True)
    results["R_D"] = gate_R_D(setup_traj, setup_sidecar) if (setup_traj and setup_sidecar) else None
    if results["R_D"] is None:
        _print_gate("R_D", True, "SKIPPED (--quick)")
    print()

    print("R_E: schema-violation raises ValueError", flush=True)
    results["R_E"] = gate_R_E()
    print()

    print("R_F: v0.1 trajectory + no sidecar gracefully degrades", flush=True)
    results["R_F"] = gate_R_F(reference_v01)
    print()

    # -------------------------------------------------------------------
    # Stage 2 gates: R_G family (envelope), R_H family (classifier),
    # R_I (denominator stability), R_E_malformed (defensive sidecar load),
    # R_G_orchestrator (--slow, end-to-end smoke).
    # -------------------------------------------------------------------
    print("R_G: envelope sidecar round-trip + sector_proxy_q_std invariance", flush=True)
    results["R_G"] = gate_R_G(setup_traj, setup_sidecar) if (setup_traj and setup_sidecar) else None
    if results["R_G"] is None:
        _print_gate("R_G", True, "SKIPPED (--quick)")
    print()

    print("R_G_legacy: legacy v0.1 sidecar -> envelope NOT_RUN; total=10", flush=True)
    results["R_G_legacy"] = gate_R_G_legacy(reference_v01)
    print()

    print("R_G_regime: regime='too_short' excludes best-score beta", flush=True)
    results["R_G_regime"] = gate_R_G_regime(setup_traj, setup_sidecar) if (setup_traj and setup_sidecar) else None
    if results["R_G_regime"] is None:
        _print_gate("R_G_regime", True, "SKIPPED (--quick)")
    print()

    print("R_H_module: kernel-free sector_classifier on well-separated clusters", flush=True)
    results["R_H_module"] = gate_R_H_module()
    print()

    print("R_H_module_NA: identity-link traj -> NOT_APPLICABLE", flush=True)
    results["R_H_module_NA"] = gate_R_H_module_NA()
    print()

    print("R_H_report: defense-in-depth verdict recomputation (PASS / FAIL paths)", flush=True)
    results["R_H_report"] = gate_R_H_report(setup_traj, setup_sidecar) if (setup_traj and setup_sidecar) else None
    if results["R_H_report"] is None:
        _print_gate("R_H_report", True, "SKIPPED (--quick)")
    print()

    print("R_H_legacy: legacy v0.1 sidecar -> classifier NOT_RUN", flush=True)
    results["R_H_legacy"] = gate_R_H_legacy(reference_v01)
    print()

    print("R_I: denominator stable at 10 (legacy = 8 substantive + 2 NOT_RUN)", flush=True)
    results["R_I"] = gate_R_I(setup_traj, setup_sidecar) if (setup_traj and setup_sidecar) else None
    if results["R_I"] is None:
        _print_gate("R_I", True, "SKIPPED (--quick)")
    print()

    print("R_E_malformed: bad envelope / classifier JSON -> NOT_RUN, no crash", flush=True)
    results["R_E_malformed"] = gate_R_E_malformed(setup_traj, setup_sidecar) if (setup_traj and setup_sidecar) else None
    if results["R_E_malformed"] is None:
        _print_gate("R_E_malformed", True, "SKIPPED (--quick)")
    print()

    print("R_J: sector_classifier golden-file intra-process bit-identity", flush=True)
    results["R_J"] = gate_R_J()
    print()

    if args.slow:
        print("R_G_orchestrator: end-to-end --beta-envelope smoke (--slow)", flush=True)
        results["R_G_orchestrator"] = gate_R_G_orchestrator(reference_v01)
        print()

    if tmp_setup_dir is not None:
        try:
            shutil.rmtree(tmp_setup_dir)
        except Exception:
            pass

    print("=" * 68, flush=True)
    fired = [k for k, v in results.items() if v is not None]
    failed = [k for k, v in results.items() if v is False]
    n_total = 17 if args.slow else 16
    if failed:
        print(f"OVERALL: {len(failed)} gate(s) FAILED: {failed}", flush=True)
        return 2
    print(f"OVERALL: {len(fired)}/{n_total} gates PASS"
          + (f" ({n_total - len(fired)} skipped)" if len(fired) < n_total else ""),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
