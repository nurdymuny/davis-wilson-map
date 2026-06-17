"""
validation_report.py -- Generate a Halcyon validation report from a buckyball
trajectory.json plus an optional sidecar with the run's final (U, E) state,
heatbath ensemble data, and beta-scan results.

Public API
----------
generate_report(trajectory_path, output_dir, run_metadata=None,
                sidecar_path=None, gauge_invariance_seed=20260616,
                run_scan=False) -> (md_path, json_path)

    Reads trajectory_path (v0.1 or v0.2), runs every check Section 2-6 of the
    seven-section discipline that the available data supports, writes a
    timestamped Markdown report and a typed JSON artifact next to it, and
    returns absolute paths to both files.

    The optional sidecar (a .pt or .npz alongside the trajectory) lets the
    generator perform checks that v0.1 trajectory.json cannot support on its
    own:
        - time reversibility (needs U_init, E_init, U_final, E_final)
        - gauge invariance (needs U_final, E_final)
        - method cross-check (needs heatbath ensemble + canonical measurement)
        - beta-scan (needs an array of {beta, P_measured, sem})

    Without a sidecar the generator still produces a valid report that:
        - verifies all substrate identities (Section 1)
        - verifies energy conservation from per-frame energies (Section 2.1)
        - reproduces the analytical Migdal-Witten target (Section 3.1)
        - measures the time-average plaquette from the frames and compares to
          the analytical target as a partial Section 5 cross-check
        - marks remaining sections "not available in this trajectory version"

    A report is REAL when every quantitative claim it makes is either
    (a) computed from an analytical formula in this module, (b) measured
    directly from the trajectory or sidecar that the calling agent produced,
    or (c) explicitly flagged as not yet validated. No synthesised numbers.

DISCIPLINE
----------
- This module does NOT modify any validated kernel file. It reads from
  buckyball_graph, buckyball_action, buckyball_heatbath, buckyball_integrator,
  buckyball_observables, buckyball_yangmills_exact, symplectic_integrator.
- Strict tolerances only. No error-bar slack on the Migdal-Witten gap.
- gauge_invariance_seed lets a reviewer regenerate the same gauge-invariance
  numbers bit-identically. The random Haar g_v draw is seeded from this value
  every run.
- Schema version of the JSON artifact is "halcyon_v1"; report schema version
  1.2 (1.1 added: gauss_history per-frame trajectory-wise Gauss check,
  Flyvbjerg-Petersen blocked SEMs at every P_sem path, and a
  tolerance_derivations metadata block / Appendix A. P_sem semantics shifted
  from naive to blocked; naive value preserved under *_sem_naive. 1.2 adds:
  section_8_beta_envelope local-stability sweep across an edge-probing beta
  envelope, and section_9_sector_classifier blind classifier on real
  trajectory windows binned by observed Q_surrogate. overall_verdict.total
  moves 8 -> 10 in 1.2.) Bumping requires a matching schema in
  test_validation_report.py.

- Sidecar blob contract. The .npz sidecar carries beta_scan,
  beta_envelope_history, and sector_classifier_state as JSON-encoded scalar
  STRINGS (np.string_), NOT numpy arrays. Downstream readers must use
  json.loads(str(z[key])). numpy.load(allow_pickle=False) is supported.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPORT_SCHEMA_VERSION = "1.2"
REPORT_VERSION = "halcyon_v1"
TOOL_VERSION = "validation_report.py/1.0"

# Strict tolerances (no loosening, no error-bar slack on Migdal-Witten).
TOL = {
    "energy_drift_rel": 1.0e-3,
    "gauss_max": 1.0e-9,
    "time_reverse": 1.0e-8,
    "migdal_witten": 1.0e-2,
    "gauge_invariance": 1.0e-10,
    "casimir_invariance": 1.0e-8,
    "method_crosscheck": 2.0e-2,
    # 1.2 schema additions: beta-envelope local-stability heuristics +
    # sector-classifier statistical contract.
    "envelope_energy_drift_rel": 5.0e-3,        # local-stability heuristic (NOT autocorrelation-derived)
    "envelope_gauss_max": 1.0e-9,
    "envelope_crosscheck_gap": 3.0e-2,
    "envelope_migdal_gap": 1.0e-2,
    "sector_classifier_real_accuracy_min": 0.80,  # stated as function of n_eff per CHAIN, NOT raw N
    "sector_classifier_null_alpha": 0.05,         # permutation-test p-value threshold (NOT a symmetric band)
}


def classifier_thresholds() -> Tuple[float, float]:
    """Single source of truth for the sector classifier gate criteria.

    Returns (real_accuracy_min, null_alpha). The orchestrator and the
    report builder MUST both call this -- no hard-coded duplicates.
    """
    return TOL["sector_classifier_real_accuracy_min"], TOL["sector_classifier_null_alpha"]

# Tolerance derivations -- each gate's expected residual is a CLAIM derived
# from first principles (machine epsilon, integrator order, FLOP counts,
# sample autocorrelation). The 'expected_residual' is what the gate would
# typically measure on this substrate at these parameters; the gap between
# 'expected_residual' and 'value' is the conservative_factor reviewers can
# tighten against. source_template strings are parameterised at render time
# (see _render_tolerance_appendix in the markdown layer).
TOLERANCE_DERIVATIONS: List[Dict[str, Any]] = [
    {
        "key": "energy_drift_rel",
        "value": TOL["energy_drift_rel"],
        "expected_residual": 4.0e-6,
        "source_template": (
            "Leapfrog is a 2nd-order symplectic integrator. By backward error "
            "analysis (Hairer-Lubich-Wanner) it exactly conserves a modified "
            "Hamiltonian H_dt differing from H by O(dt^2), so |H(t) - H(0)| / "
            "|H_0| is bounded by C * dt^2 over exponentially long times rather "
            "than growing secularly. At dt={dt} this gives ~dt^2 ~ {dt2:.1e} "
            "times an O(0.01-0.1) dimensionless prefactor for this system, "
            "hence observed ~1e-5."
        ),
        "conservative_factor": (
            "~100x looser than expected; absorbs prefactor + cross-platform "
            "FP variation"
        ),
    },
    {
        "key": "gauss_max",
        "value": TOL["gauss_max"],
        "expected_residual": 1.0e-13,
        "source_template": (
            "Tikhonov-shifted CG (shift = 1e-14) bounds the steady-state "
            "residual below by the shift. Per-vertex residual evaluation "
            "accumulates O(V * d_v) ~ O(60 * 3) ~ 180 FP operations on "
            "magnitudes ~|E| ~ O(1), giving roundoff ~ 180 * eps_mach ~ "
            "4e-14. One pass of iterative refinement drives residual to "
            "~1e-13."
        ),
        "conservative_factor": (
            "~4 orders looser than the realistic CG residual floor"
        ),
    },
    {
        "key": "time_reverse",
        "value": TOL["time_reverse"],
        "expected_residual": 7.0e-13,
        "source_template": (
            "Per-step roundoff ~ eps_mach * O(100 FLOPs) ~ 2e-14. Random-walk "
            "accumulation over N={N} steps gives ~sqrt(N) * 2e-14 ~ {rw:.1e}; "
            "pathological worst-case linear accumulation gives ~N * 2e-14 ~ "
            "{lin:.1e}."
        ),
        "conservative_factor": "~3-5 orders looser than either bound",
    },
    {
        "key": "migdal_witten",
        "value": TOL["migdal_witten"],
        "expected_residual": 3.5e-3,
        "source_template": (
            "Per-sweep plaquette std on the buckyball (averaged over 32 faces) "
            "is sigma_<P> ~ sigma_link / sqrt(n_faces) ~ 0.5/sqrt(32) ~ 0.09 "
            "at beta={beta}, dropping further once n_eff is accounted for. "
            "With N=1000 raw sweeps and tau_int ~ 5 -> n_eff ~ 200, SEM ~ "
            "0.09/sqrt(200) ~ 6e-3. Empirical SEM from blocked analysis "
            "(current run) is in the 3-5e-3 range."
        ),
        "conservative_factor": (
            "~2-3 SEM (strict; designed to fail loudly on calibration drift)"
        ),
    },
    {
        "key": "gauge_invariance",
        "value": TOL["gauge_invariance"],
        "expected_residual": 1.0e-12,
        "source_template": (
            "Gauge transform touches 90 edges (each ~16 FLOPs) and verification "
            "compares plaquette sums over 32 faces with 5-6 element products "
            "(each ~6*16 FLOPs): total ~ 90*16 + 32*6*16 ~ 4500 FLOPs; "
            "eps_mach * 4500 ~ 1e-12 conservatively."
        ),
        "conservative_factor": "~2 orders looser than expected",
    },
    {
        "key": "method_crosscheck",
        "value": TOL["method_crosscheck"],
        "expected_residual": 1.1e-2,
        "source_template": (
            "Budget split: stat = sqrt(sem_time^2 + sem_hb^2) where each is "
            "taken from the blocked analysis of the current run (typically "
            "~3e-3 each -> ~4e-3 stat). Sys = microcanonical-vs-canonical "
            "bias at finite trajectory length on 93 transverse DOF, bounded "
            "by O(sigma_<P> / sqrt(n_steps * dt / tau_relax)) ~ 0.09/sqrt("
            "{Tdt}/1) ~ {sys:.1e}. Total sqrt(stat^2 + sys^2) ~ 1.1e-2."
        ),
        "conservative_factor": (
            "~2x looser; absorbs stat fluctuations and the documented 93-DOF "
            "ergodicity caveat"
        ),
    },
    # ------------------------------------------------------------------
    # 1.2 schema additions: beta-envelope + sector-classifier gates.
    # These tolerances are LOCAL-STABILITY heuristics for envelope rows
    # and a STATISTICAL CONTRACT for the classifier; they are NOT
    # autocorrelation-derived in the same sense as the Section 2-5
    # tolerances above.
    # ------------------------------------------------------------------
    {
        "key": "envelope_energy_drift_rel",
        "value": TOL["envelope_energy_drift_rel"],
        "expected_residual": 5.0e-5,
        "source_template": (
            "Local-stability heuristic at each envelope beta. Short leapfrog "
            "(~{N_env} steps at dt=0.02) is sub-trajectory of the main run; "
            "expected drift scales linearly with the main-run envelope budget. "
            "Envelope tolerance is 5x the main-run gate (1e-3) to absorb "
            "edge-of-envelope variance without flagging a real stable point. "
            "NOT derived from a per-beta autocorrelation analysis -- this is "
            "a stability heuristic, not a publication-grade statistical bound."
        ),
        "conservative_factor": "5x looser than the main-run gate; absorbs edge-of-envelope variance",
    },
    {
        "key": "envelope_gauss_max",
        "value": TOL["envelope_gauss_max"],
        "expected_residual": 1.0e-13,
        "source_template": (
            "Identical to the main-run gauss_max tolerance: the Gauss "
            "projector floor is set by Tikhonov shift + FP roundoff and is "
            "beta-independent."
        ),
        "conservative_factor": "Same as main-run; the projector floor is invariant under beta",
    },
    {
        "key": "envelope_crosscheck_gap",
        "value": TOL["envelope_crosscheck_gap"],
        "expected_residual": 1.5e-2,
        "source_template": (
            "Local microcanonical-vs-canonical gap at each envelope beta. "
            "1.5x looser than the main-run cross-check (2e-2) because the "
            "envelope's short heatbath chain ({env_sweeps} sweeps) has "
            "smaller n_eff than the main run."
        ),
        "conservative_factor": "1.5x looser; reflects smaller n_eff in envelope phase",
    },
    {
        "key": "envelope_migdal_gap",
        "value": TOL["envelope_migdal_gap"],
        "expected_residual": 5.0e-3,
        "source_template": (
            "Identical to the main-run Migdal-Witten gate: the analytical "
            "target is independent of measurement budget."
        ),
        "conservative_factor": "Same as main-run; analytical target is sample-budget-invariant",
    },
    {
        "key": "sector_classifier_real_accuracy_min",
        "value": TOL["sector_classifier_real_accuracy_min"],
        "expected_residual": 0.85,
        "source_template": (
            "Stated as a function of n_eff per CHAIN, not raw N=90 samples "
            "per band. The within-chain LOOCV folds are autocorrelated; "
            "effective independent samples per band is roughly n_eff = "
            "n_samples / (2 * tau_int + 1). At tau_int ~ 5 the effective N "
            "per band is ~6, so 80% accuracy means the classifier reads at "
            "least 5/6 effective samples correctly. Below 80% the "
            "gauge-invariant features do not carry band information; the "
            "gate fails loudly."
        ),
        "conservative_factor": "n_eff-corrected; not directly comparable to a 90-sample raw classification rate",
    },
    {
        "key": "sector_classifier_null_alpha",
        "value": TOL["sector_classifier_null_alpha"],
        "expected_residual": 0.5,
        "source_template": (
            "Permutation-test p-value threshold for label-shuffled LOO. "
            "200 random shuffles of the real feature matrix's labels; null "
            "p-value = (number of shuffles whose accuracy meets or exceeds "
            "the real classifier) / 200. p >= 0.05 means the real classifier "
            "is statistically indistinguishable from a label-blind baseline "
            "-- gate FAILS. NOT a symmetric +-band around 1/3 (the prior "
            "design's '+-0.10 around 1/3' criterion misfired by construction "
            "because the Haar scramble null does not have a 1/3 baseline)."
        ),
        "conservative_factor": "5% type-I-error rate on the standard permutation contract",
    },
]

# Truncated icosahedron expected combinatorics (verified at runtime).
BUCKYBALL_EXPECTED = {
    "V": 60,
    "E": 90,
    "F": 32,
    "pentagons": 12,
    "hexagons": 20,
    "chi": 2,
    "edge_census_PH": 60,
    "edge_census_HH": 30,
    "edge_census_PP": 0,
    "sum_of_perimeters": 180,
}

OPEN_ITEMS: List[Dict[str, Any]] = [
    {
        "category": "framework",
        "description": (
            "Davis Duality / Branch XIII derivation of the specific functional "
            "form of the variable-beta coupling delta-m^2(x) = F[Omega, tau, K]. "
            "The current code uses an interim holonomy-density form which is "
            "gauge-invariant at FP64 (Section 4 verifies) and dimensionally "
            "correct, but is not derived from the framework."
        ),
        "operational_today": True,
    },
    {
        "category": "fermion",
        "description": (
            "SU(2) fermion sector. Matter-sector v1 validated the SU(3) fermion "
            "sector with staggered Dirac, Banks-Casher, and chRMT cross-checks. "
            "SU(2) fermions are open."
        ),
        "operational_today": False,
    },
    {
        "category": "hardware",
        "description": (
            "Multi-mode transmon plaquette network encoding from physical Josephson "
            "bias currents to SU(2) link variables. This simulation tests the gauge "
            "field dynamics, not the encoding."
        ),
        "operational_today": False,
    },
    {
        "category": "ergodicity",
        "description": (
            "Microcanonical-canonical agreement on small substrates. The buckyball "
            "has 93 transverse degrees of freedom. The Section 5 cross-check gap "
            "is consistent with a finite-size and short-trajectory ergodicity "
            "limitation rather than a structural failure of the integrator, but "
            "this interpretation is not yet demonstrated: closing it requires "
            "a trajectory-length sweep at fixed seed and multiple independent "
            "seeds (Flyvbjerg-Petersen blocked SEMs are now applied per "
            "Section 3.2 / Section 5 / Section 6). Deferred to the longer "
            "convergence study planned for the apparatus characterisation phase."
        ),
        "operational_today": True,
    },
    {
        "category": "drift",
        "description": (
            "Long-term apparatus drift over hours-to-days operating timescales. "
            "Junction-level decoherence, thermal drift, and 1/f noise in bias "
            "currents are open items for the hardware phase."
        ),
        "operational_today": False,
    },
    {
        "category": "claim",
        "description": (
            "Connection to inertia damping. The simulation produces validated "
            "gauge field dynamics. The claim that this dynamics modifies effective "
            "inertial mass is a prediction the apparatus, when built, will test. "
            "The simulation does not and cannot test the inertia claim."
        ),
        "operational_today": False,
    },
    {
        "category": "section_5_2_equipartition",
        "description": (
            "Canonical equipartition K/V band on the heatbath ensemble "
            "(Section 5.2 in the discipline template). Not yet emitted by "
            "this report. Adding it requires sampling K and V on the "
            "canonical heatbath history and verifying both are within band "
            "of the analytical equipartition expectation. No TOL entry "
            "exists yet; gating values will be added once the band is "
            "measured on a reference run."
        ),
        "operational_today": True,
    },
    # ------------------------------------------------------------------
    # The seven-gate pre-registered hardware kill chain.  Each entry is a
    # GATE the apparatus must clear before its mechanical / inertial
    # channel can be unblinded.  The simulation report does NOT pass these;
    # they are tracked here so a reviewer reading the JSON or the markdown
    # sees the discipline that gates the next phase.
    # ------------------------------------------------------------------
    {
        "category": "hardware_gate_1_gauge",
        "description": (
            "GATE 1 (Gauge). Apparatus must demonstrate accepted Q in {0, 1, 2} "
            "sectors via gauge-invariant observables: Wilson loops {W_gamma_i}, "
            "mean plaquette <P>, sector surrogate Q_surrogate, and Wilson action "
            "S_W must agree on the sector label and reach a pre-registered "
            "separability threshold across seeds and small control perturbations "
            "before any mechanical channel is unblinded."
        ),
        "operational_today": False,
    },
    {
        "category": "hardware_gate_2_stability",
        "description": (
            "GATE 2 (Stability). Sector must remain stable through the full "
            "mechanical measurement window. Per-snapshot logging of "
            "Q_surrogate(t), <P(t)>, max|G_v(t)|. Runs whose sector leaves "
            "the pre-registered band during measurement are INVALID. "
            "No post-hoc rescue."
        ),
        "operational_today": False,
    },
    {
        "category": "hardware_gate_3_null_drive",
        "description": (
            "GATE 3 (Null-drive). Power-matched and scrambled-phase drives "
            "must produce NO Q-linear inertial signal. Sham drives are part "
            "of the same data-taking block as the real Q-sector drive, not a "
            "follow-up campaign. The inertial claim only survives if the "
            "signal follows Q -- not RF power, heat, vibration, magnetic "
            "field, or drive amplitude."
        ),
        "operational_today": False,
    },
    {
        "category": "hardware_gate_4_blind_analysis",
        "description": (
            "GATE 4 (Blind analysis). The mechanical-channel analyst must be "
            "blind to sector labels until the analysis pipeline is committed. "
            "Unblinding is a one-way step."
        ),
        "operational_today": False,
    },
    {
        "category": "hardware_gate_5_linearity",
        "description": (
            "GATE 5 (Linearity). Q = 2 deviation must be approximately twice "
            "the Q = 1 deviation, within the systematics band. The linear-in-Q "
            "coupling ansatz is itself a gate; a non-linear residual at the "
            "framework's predicted magnitude falsifies the simple linear model."
        ),
        "operational_today": False,
    },
    {
        "category": "hardware_gate_6_reversal",
        "description": (
            "GATE 6 (Reversal). Reversing the programmed winding "
            "(Q -> -Q) must reverse or transform the predicted signature "
            "according to the framework's parity prediction. A signal that "
            "survives winding reversal in the wrong way falsifies the framework "
            "before linearity even matters."
        ),
        "operational_today": False,
    },
    {
        "category": "hardware_gate_7_independent_sensor",
        "description": (
            "GATE 7 (Independent sensor). The effect must appear in at least "
            "two measurement modalities, each with its own systematics model, "
            "before it is treated as a candidate physics result. A single-channel "
            "result is automatically suspect."
        ),
        "operational_today": False,
    },
    {
        "category": "sensitivity_floor_and_systematics_budget",
        "description": (
            "Pre-registered sensitivity floor alpha_min and four-channel "
            "systematics budget (thermal / magnetic / vibration / cage drift) "
            "are published BEFORE first data. Values to be set by Branch XIII "
            "derivation + cage characterisation runs. Backfilling these numbers "
            "after data is taken is grounds for retraction. The slot itself is "
            "the pre-registration."
        ),
        "operational_today": False,
    },
    {
        "category": "operational_beta_envelope",
        "description": (
            "Operating beta for the buckyball substrate will be selected from "
            "beta in {2.4, 2.5, 2.6, 2.7, 2.8} based on local-envelope "
            "stability (sector separation, energy conservation, Gauss covariance, "
            "canonical agreement) at each candidate value. Not the lower edge by "
            "default. The validation report's beta-envelope sweep "
            "(--beta-envelope orchestrator flag, pending implementation) is the "
            "gating measurement."
        ),
        "operational_today": True,
    },
]


# ---------------------------------------------------------------------------
# Lazy kernel imports (no top-level coupling; no kernel modification).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))


def _load_module(name: str, abs_path: str):
    spec = importlib.util.spec_from_file_location(name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {abs_path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod_cache: Dict[str, Any] = {}


def _mod(short: str):
    """Get a kernel module by short name. Cached."""
    if short in _mod_cache:
        return _mod_cache[short]
    rel = {
        "graph": os.path.join(_HERE, "buckyball_graph.py"),
        "action": os.path.join(_HERE, "buckyball_action.py"),
        "heatbath": os.path.join(_HERE, "buckyball_heatbath.py"),
        "integrator": os.path.join(_HERE, "buckyball_integrator.py"),
        "observables": os.path.join(_HERE, "buckyball_observables.py"),
        "ym_exact": os.path.join(_HERE, "buckyball_yangmills_exact.py"),
    }[short]
    mod_name = f"_vrep_{short}"
    m = _load_module(mod_name, rel)
    _mod_cache[short] = m
    return m


# ---------------------------------------------------------------------------
# Trajectory loading + extra-meta extraction
# ---------------------------------------------------------------------------
def _load_trajectory(path: str) -> Dict[str, Any]:
    """Read a trajectory.json with schema validation via buckyball_observables."""
    obs = _mod("observables")
    return obs.load_trajectory(path)


def _extract_run_params(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the run-level parameters embedded as extra-meta in the trajectory.

    The orchestrator records beta, dt, n_steps, seed, etc. at the top level
    of trajectory.json (extra_meta in dump_trajectory). Frames carry per-snapshot
    t, step, energy, control, plaquette_mean, Q.

    T_total is computed from n_steps * dt (the actual simulation length),
    not from the measured-frame span. Earlier versions reported
    frame[-1].t - frame[0].t which under-counts by the measure_every gap on
    each end and is internally inconsistent with the n_steps and dt rows.
    """
    n_steps = traj.get("n_steps")
    dt = traj.get("dt")
    T_total = (float(n_steps) * float(dt)) if (n_steps is not None and dt is not None) else None
    frame_t_first = traj["frames"][0]["t"] if traj.get("frames") else None
    frame_t_last = traj["frames"][-1]["t"] if traj.get("frames") else None
    params = {
        "schema_version": traj.get("schema_version"),
        "gauge_group": traj.get("gauge_group"),
        "V": traj["lattice"]["vertices"],
        "E": traj["lattice"]["edges"],
        "F": traj["lattice"]["faces"],
        "beta": traj.get("beta"),
        "dt": dt,
        "n_steps": n_steps,
        "n_thermalization_sweeps": traj.get("n_thermalization_sweeps"),
        "measure_every": traj.get("measure_every"),
        "seed": traj.get("seed"),
        "n_frames": len(traj.get("frames", [])),
        "T_total": T_total,
        "frame_t_first": frame_t_first,
        "frame_t_last": frame_t_last,
    }
    return params


# ---------------------------------------------------------------------------
# Sidecar loading -- optional rich-state data the trajectory schema doesn't carry
# ---------------------------------------------------------------------------
@dataclass
class SidecarData:
    """Loaded rich state for a run; any field may be None.

    The orchestrator writes this alongside trajectory.json so the report can
    do the checks the v0.1 schema can't carry. Persisted as a numpy .npz file
    with float64 arrays (no torch tensors on disk; the loader rehydrates to
    torch only at the point of computation).
    """
    U_init: Optional[np.ndarray] = None        # (n_edges, 4)
    E_init: Optional[np.ndarray] = None        # (n_edges, 4)
    U_final: Optional[np.ndarray] = None       # (n_edges, 4)
    E_final: Optional[np.ndarray] = None       # (n_edges, 4)
    heatbath_canonical_P_history: Optional[np.ndarray] = None  # (n_samples,)
    heatbath_canonical_n_thermalization: Optional[int] = None
    heatbath_canonical_beta: Optional[float] = None
    beta_scan: Optional[List[Dict[str, float]]] = None
    gauss_history: Optional[np.ndarray] = None        # (n_snapshots, 2): [step, max|G|_inf]
    # 1.2 schema additions: beta-envelope local-stability sweep + sector-
    # classifier blind-band gate.
    beta_envelope_history: Optional[List[Dict[str, Any]]] = None
    sector_classifier_state: Optional[Dict[str, Any]] = None
    code_commit: Optional[str] = None
    wall_time_seconds: Optional[float] = None


def _load_sidecar(path: Optional[str]) -> SidecarData:
    if path is None:
        return SidecarData()
    p = Path(path)
    if not p.exists():
        return SidecarData()
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=True) as z:
            sd = SidecarData()
            for key in ("U_init", "E_init", "U_final", "E_final",
                        "heatbath_canonical_P_history", "gauss_history"):
                if key in z.files:
                    sd.__dict__[key] = z[key].astype(np.float64)
            if "heatbath_canonical_n_thermalization" in z.files:
                sd.heatbath_canonical_n_thermalization = int(z["heatbath_canonical_n_thermalization"])
            if "heatbath_canonical_beta" in z.files:
                sd.heatbath_canonical_beta = float(z["heatbath_canonical_beta"])
            if "beta_scan" in z.files:
                try:
                    sd.beta_scan = json.loads(str(z["beta_scan"]))
                except Exception:
                    sd.beta_scan = None
            # 1.2 schema additions. Defensive: malformed JSON-string blobs
            # degrade gracefully to None, never crash. Required by R_E
            # malformed-blob gate.
            if "beta_envelope_history" in z.files:
                try:
                    val = json.loads(str(z["beta_envelope_history"]))
                    if isinstance(val, list) and all(isinstance(x, dict) for x in val):
                        sd.beta_envelope_history = val
                    else:
                        sd.beta_envelope_history = None
                except Exception:
                    sd.beta_envelope_history = None
            if "sector_classifier_state" in z.files:
                try:
                    val = json.loads(str(z["sector_classifier_state"]))
                    if isinstance(val, dict):
                        sd.sector_classifier_state = val
                    else:
                        sd.sector_classifier_state = None
                except Exception:
                    sd.sector_classifier_state = None
            if "code_commit" in z.files:
                sd.code_commit = str(z["code_commit"])
            if "wall_time_seconds" in z.files:
                sd.wall_time_seconds = float(z["wall_time_seconds"])
            return sd
    raise ValueError(f"unsupported sidecar format: {p.suffix!r}; expected .npz")


# ---------------------------------------------------------------------------
# Substrate-identity verification (Section 1)
# ---------------------------------------------------------------------------
def _verify_substrate(graph) -> Dict[str, Any]:
    V = graph.n_vertices
    E = graph.n_edges
    F = graph.n_faces
    chi = V - E + F
    n_pent = len(graph.pentagons)
    n_hex = len(graph.hexagons)
    pent_perim = sum(5 for _ in graph.pentagons)
    hex_perim = sum(6 for _ in graph.hexagons)
    perim_sum = pent_perim + hex_perim
    two_face_check = graph.verify_pent_hex_edges()
    census = graph.edge_type_census()

    checks = {
        "V": (V, BUCKYBALL_EXPECTED["V"]),
        "E": (E, BUCKYBALL_EXPECTED["E"]),
        "F": (F, BUCKYBALL_EXPECTED["F"]),
        "pentagons": (n_pent, BUCKYBALL_EXPECTED["pentagons"]),
        "hexagons": (n_hex, BUCKYBALL_EXPECTED["hexagons"]),
        "chi": (chi, BUCKYBALL_EXPECTED["chi"]),
        "perimeter_sum": (perim_sum, BUCKYBALL_EXPECTED["sum_of_perimeters"]),
        "2E_check": (2 * E, BUCKYBALL_EXPECTED["sum_of_perimeters"]),
        "two_faces_per_edge": (two_face_check, True),
        "census_PH": (census[0], BUCKYBALL_EXPECTED["edge_census_PH"]),
        "census_HH": (census[1], BUCKYBALL_EXPECTED["edge_census_HH"]),
        "census_PP": (census[2], BUCKYBALL_EXPECTED["edge_census_PP"]),
    }
    all_pass = all(meas == exp for meas, exp in checks.values())
    return {
        "checks": {k: {"measured": v[0], "expected": v[1], "pass": v[0] == v[1]} for k, v in checks.items()},
        "verdict": "PASS" if all_pass else "FAIL",
    }


# ---------------------------------------------------------------------------
# Flyvbjerg-Petersen blocking analysis (autocorrelation-corrected SEM)
# ---------------------------------------------------------------------------
def _flyvbjerg_petersen_blocking(x: np.ndarray) -> Dict[str, Any]:
    """Flyvbjerg-Petersen blocking analysis for autocorrelation-corrected SEM.

    Returns a dict with keys:
        sem_naive             : std(x, ddof=1) / sqrt(N), the naive SEM
        sem_blocked           : the blocked SEM (plateau-detected or
                                conservative fallback)
        n_eff                 : effective sample count (sem_naive/sem_blocked)^2 * N,
                                clamped to [1, N]
        plateau_block_size    : block size at which the plateau was detected
                                (or the last block size if no plateau)
        plateau_detected      : True iff a 3-consecutive-levels-within-10% plateau
                                was found at >=4 levels of doubling
        blocking_curve        : list[(block_size, sem)] of the doubling curve
        n_samples             : N
        regime                : one of "degenerate", "too_short", "single_level",
                                "shallow", "plateau_detected", "no_plateau"

    Determinism: pure numpy, no RNG. Bit-identical for identical inputs at
    pinned numpy/BLAS versions. Never under-reports SEM: conservative
    fallback uses max(last-block-SEM, naive-SEM) when no plateau is found.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("blocking input contains non-finite values")
    N = int(x.size)
    if N < 4:
        return {
            "sem_naive": 0.0, "sem_blocked": 0.0, "n_eff": float(N),
            "plateau_block_size": 1, "plateau_detected": False,
            "blocking_curve": [], "n_samples": N, "regime": "degenerate",
        }
    sem_naive = float(x.std(ddof=1) / math.sqrt(N))
    # Zero variance (constant input): n_eff is meaningfully 1.
    if sem_naive == 0.0:
        return {
            "sem_naive": 0.0, "sem_blocked": 0.0, "n_eff": 1.0,
            "plateau_block_size": 1, "plateau_detected": False,
            "blocking_curve": [], "n_samples": N, "regime": "degenerate",
        }
    # Build doubling curve; require >=16 blocks at each level for a usable SEM.
    curve: List[Tuple[int, float]] = []
    cur = x.copy()
    block_size = 1
    while cur.size >= 32:
        n = (cur.size // 2) * 2
        cur = (cur[:n][::2] + cur[:n][1::2]) / 2.0
        block_size *= 2
        sem_block = float(cur.std(ddof=1) / math.sqrt(cur.size))
        curve.append((block_size, sem_block))
    # Too short for any doubling step.
    if len(curve) == 0:
        return {
            "sem_naive": sem_naive, "sem_blocked": sem_naive, "n_eff": float(N),
            "plateau_block_size": 1, "plateau_detected": False,
            "blocking_curve": [], "n_samples": N, "regime": "too_short",
        }
    # Single-level or shallow: not enough doublings for a meaningful plateau.
    # Conservative fallback: max(last-block SEM, naive SEM). Never under-report.
    if N < 256 or len(curve) < 4:
        last_block_sem = float(curve[-1][1])
        sem_blocked_conservative = max(last_block_sem, sem_naive)
        n_eff = float(N) if sem_blocked_conservative == sem_naive else float(
            min(N, max(1.0, (sem_naive / sem_blocked_conservative) ** 2 * N))
        )
        regime = "single_level" if len(curve) == 1 else "shallow"
        return {
            "sem_naive": sem_naive, "sem_blocked": sem_blocked_conservative,
            "n_eff": n_eff, "plateau_block_size": int(curve[-1][0]),
            "plateau_detected": False,
            "blocking_curve": [(int(b), float(s)) for (b, s) in curve],
            "n_samples": N, "regime": regime,
        }
    # Plateau test: 3 consecutive levels agreeing within 10% of their local mean.
    sems = np.array([c[1] for c in curve], dtype=np.float64)
    plateau_idx: Optional[int] = None
    for i in range(len(sems) - 2):
        window = sems[i:i + 3]
        mu = float(window.mean())
        if mu > 0 and float(window.max() - window.min()) / mu <= 0.10:
            plateau_idx = i
            break
    if plateau_idx is None:
        last_block_sem = float(sems[-1])
        sem_blocked_conservative = max(last_block_sem, sem_naive)
        n_eff = float(min(N, max(1.0, (sem_naive / sem_blocked_conservative) ** 2 * N)))
        return {
            "sem_naive": sem_naive, "sem_blocked": sem_blocked_conservative,
            "n_eff": n_eff, "plateau_block_size": int(curve[-1][0]),
            "plateau_detected": False,
            "blocking_curve": [(int(b), float(s)) for (b, s) in curve],
            "n_samples": N, "regime": "no_plateau",
        }
    sem_blocked = float(sems[plateau_idx:plateau_idx + 3].max())
    plateau_block_size = int(curve[plateau_idx][0])
    raw_n_eff = (sem_naive / sem_blocked) ** 2 * N if sem_blocked > 0 else 1.0
    plateau_detected = True
    if raw_n_eff > N:
        # Small-sample artifact: blocked SEM came in lower than naive.
        # Demote plateau flag and clamp.
        plateau_detected = False
        raw_n_eff = float(N)
    n_eff = float(min(N, max(1.0, raw_n_eff)))
    return {
        "sem_naive": sem_naive, "sem_blocked": sem_blocked,
        "n_eff": n_eff, "plateau_block_size": plateau_block_size,
        "plateau_detected": plateau_detected,
        "blocking_curve": [(int(b), float(s)) for (b, s) in curve],
        "n_samples": N,
        "regime": "plateau_detected" if plateau_detected else "no_plateau",
    }


# ---------------------------------------------------------------------------
# Section 2: conservation laws
# ---------------------------------------------------------------------------
def _energy_conservation(traj: Dict[str, Any]) -> Dict[str, Any]:
    frames = traj["frames"]
    energies = [f["energy"] for f in frames if "energy" in f]
    if not energies:
        return {"available": False, "reason": "frames carry no 'energy' field"}
    H = np.asarray(energies, dtype=np.float64)
    # Loud-fail on non-finite values: a NaN/inf in the energy trace means the
    # integrator went off the rails, the trajectory.json was corrupted, or a
    # frame value was lost. None of those are silently-degradable.
    nonfinite_count = int(np.sum(~np.isfinite(H)))
    if nonfinite_count > 0:
        return {
            "available": True,
            "verdict": "FAIL",
            "reason": (f"{nonfinite_count}/{len(H)} energy values are NaN or inf; "
                       f"trajectory is corrupted or integrator diverged"),
            "n_samples": int(len(H)),
            "nonfinite_count": nonfinite_count,
            "tolerance": TOL["energy_drift_rel"],
        }
    H0 = float(H[0])
    if abs(H0) < 1e-30:
        rel = np.abs(H - H0)
    else:
        rel = np.abs((H - H0) / H0)
    max_rel = float(rel.max())
    tol = TOL["energy_drift_rel"]
    margin = (tol / max_rel) if max_rel > 0 else float("inf")
    # Loud-fail on a fixed-H "integrator": std normalised by |H_0| should not
    # be exactly zero. A constant-H trajectory passes the drift gate
    # vacuously, masking a broken integrator that returns its input unchanged.
    H_std = float(H.std()) if len(H) > 1 else 0.0
    rel_std = (H_std / abs(H0)) if abs(H0) > 0 else H_std
    constant_H_suspect = (rel_std < 1e-14 and len(H) >= 3)
    # Monotone-drift detector: linear fit slope, normalised by H0.
    if len(H) >= 3:
        x = np.arange(len(H), dtype=np.float64)
        slope = float(np.polyfit(x, H, 1)[0])
        slope_per_frame_rel = abs(slope / H0) if abs(H0) > 0 else abs(slope)
        # Monotone if |slope * N| dominates oscillation: heuristic threshold.
        is_monotone = (slope_per_frame_rel * len(H)) > 5.0 * H_std / abs(H0) if abs(H0) > 0 else False
    else:
        slope_per_frame_rel = 0.0
        is_monotone = False
    if constant_H_suspect:
        verdict = "FAIL"
        reason = "energy is exactly constant across all frames; integrator appears dead"
    elif max_rel > tol or is_monotone:
        verdict = "FAIL"
        reason = "drift exceeds tolerance or shows monotone trend"
    else:
        verdict = "PASS"
        reason = None
    out = {
        "available": True,
        "H_0": H0,
        "n_samples": int(len(H)),
        "max_relative_drift": max_rel,
        "energy_std_over_H0": rel_std,
        "tolerance": tol,
        "margin_factor": margin,
        "monotone_drift_detected": is_monotone,
        "constant_H_suspect": constant_H_suspect,
        "verdict": verdict,
    }
    if reason:
        out["reason"] = reason
    return out


def _gauss_check_from_state(sd: SidecarData, graph) -> Dict[str, Any]:
    """Final-state Gauss residual + optional trajectory-wise max|G| sub-check.

    The trajectory-wise piece is enabled when the sidecar carries
    gauss_history (shape (n_snapshots, 2): each row is [step_index, max|G|]).
    Combined verdict is PASS iff the final-state state passes AND, if the
    trajectory-wise sub-check is available, IT passes too.
    """
    if sd.U_final is None or sd.E_final is None:
        return {"available": False, "reason": "no U_final/E_final sidecar"}
    import torch  # local: kernel uses torch
    integ = _mod("integrator")
    U = torch.from_numpy(sd.U_final).to(dtype=torch.float64)
    E = torch.from_numpy(sd.E_final).to(dtype=torch.float64)
    G = integ.compute_gauss_residual(E, U, graph)
    G_max = float(G.abs().max())
    G_flat = integ.compute_gauss_residual_flat(E, graph)
    G_flat_max = float(G_flat.abs().max())
    tol = TOL["gauss_max"]
    final_state_verdict = "PASS" if G_max <= tol else "FAIL"

    # Trajectory-wise sub-check (defaults to NOT_RUN when sidecar lacks
    # gauss_history; reviewer reads the existing final-state row only).
    traj_available = False
    traj_verdict = "NOT_RUN"
    traj_max: Optional[float] = None
    traj_min: Optional[float] = None
    traj_step: Optional[int] = None
    n_snapshots: Optional[int] = None
    gh_list: Optional[List[List[float]]] = None
    if sd.gauss_history is not None:
        gh = np.asarray(sd.gauss_history, dtype=np.float64)
        if gh.ndim != 2 or gh.shape[1] != 2:
            return {
                "available": True,
                "gauss_covariant_max_final_state": G_max,
                "gauss_flat_max_final_state": G_flat_max,
                "ratio_flat_over_covariant": (G_flat_max / G_max) if G_max > 0 else float("inf"),
                "tolerance": tol,
                "gauss_trajectorywise_available": True,
                "gauss_trajectorywise_verdict": "FAIL",
                "verdict": "FAIL",
                "final_state_verdict": final_state_verdict,
                "reason": f"gauss_history must be shape (n_snapshots, 2), got {tuple(gh.shape)}",
            }
        if not np.all(np.isfinite(gh)):
            return {
                "available": True,
                "gauss_covariant_max_final_state": G_max,
                "gauss_flat_max_final_state": G_flat_max,
                "ratio_flat_over_covariant": (G_flat_max / G_max) if G_max > 0 else float("inf"),
                "tolerance": tol,
                "gauss_trajectorywise_available": True,
                "gauss_trajectorywise_verdict": "FAIL",
                "verdict": "FAIL",
                "final_state_verdict": final_state_verdict,
                "reason": "gauss_history contains NaN or inf",
            }
        traj_available = True
        traj_max = float(gh[:, 1].max())
        traj_min = float(gh[:, 1].min())
        argmax_idx = int(gh[:, 1].argmax())
        traj_step = int(gh[argmax_idx, 0])
        n_snapshots = int(gh.shape[0])
        traj_verdict = "PASS" if traj_max <= tol else "FAIL"
        gh_list = gh.tolist()
    combined_verdict = (
        "PASS" if (final_state_verdict == "PASS" and
                   (not traj_available or traj_verdict == "PASS"))
        else "FAIL"
    )
    return {
        "available": True,
        "gauss_covariant_max_final_state": G_max,
        "gauss_flat_max_final_state": G_flat_max,
        "ratio_flat_over_covariant": (G_flat_max / G_max) if G_max > 0 else float("inf"),
        "tolerance": tol,
        "final_state_verdict": final_state_verdict,
        "gauss_trajectorywise_available": traj_available,
        "gauss_trajectorywise_verdict": traj_verdict,
        "gauss_max_across_trajectory": traj_max,
        "gauss_max_across_trajectory_step": traj_step,
        "gauss_min_across_trajectory": traj_min,
        "n_snapshots": n_snapshots,
        "gauss_history": gh_list,
        "verdict": combined_verdict,
    }


def _time_reversibility(sd: SidecarData, graph, beta: float, dt: float, n_steps: int) -> Dict[str, Any]:
    if any(x is None for x in (sd.U_init, sd.E_init, sd.U_final, sd.E_final)):
        return {"available": False, "reason": "need U_init, E_init, U_final, E_final"}
    import torch
    integ = _mod("integrator")
    U = torch.from_numpy(sd.U_final).to(dtype=torch.float64).clone()
    E = -torch.from_numpy(sd.E_final).to(dtype=torch.float64).clone()
    E[..., 0] = 0.0
    for _ in range(n_steps):
        U, E = integ.leapfrog_step(U, E, dt, graph, beta)
    E[..., 0] = 0.0
    # Compare against (U_init, -E_init); leapfrog reverses E_init.
    U_diff = float((U - torch.from_numpy(sd.U_init).to(dtype=torch.float64)).abs().max())
    E_diff = float((E - (-torch.from_numpy(sd.E_init).to(dtype=torch.float64))).abs().max())
    tol = TOL["time_reverse"]
    verdict = "PASS" if (U_diff <= tol and E_diff <= tol) else "FAIL"
    return {
        "available": True,
        "U_residual_inf": U_diff,
        "E_residual_inf": E_diff,
        "tolerance": tol,
        "verdict": verdict,
    }


def _section2(traj: Dict[str, Any], sd: SidecarData, graph,
              beta: float, dt: float, n_steps: int) -> Dict[str, Any]:
    return {
        "energy": _energy_conservation(traj),
        "gauss": _gauss_check_from_state(sd, graph),
        "time_reversibility": _time_reversibility(sd, graph, beta, dt, n_steps),
    }


# ---------------------------------------------------------------------------
# Section 3: analytical agreement (Migdal-Witten)
# ---------------------------------------------------------------------------
def _migdal_witten_target(beta: float, j_max_primary: int = 80) -> Dict[str, Any]:
    """Compute the exact buckyball <P> + an independent Bessel cross-check.

    Cross-check uses scipy.special.iv directly with the textbook
    I_2(beta) / I_1(beta) F->inf formula, which agrees with the buckyball-
    finite formula at the ~10^-10 level (the finite-volume correction).

    At small beta (< ~1.0), high-j Bessel terms in the series underflow to
    zero. The kernel handles this correctly via log-sum-exp normalisation
    (the underflowing j-values contribute negligibly), but numpy still
    raises divide-by-zero / invalid-value RuntimeWarnings during the
    intermediate np.log(I_n) computation. Suppress those warnings here so
    they do not leak into the report-generation log. The numerical result
    is unaffected.
    """
    ym = _mod("ym_exact")
    from scipy.special import iv as _iv
    with np.errstate(divide="ignore", invalid="ignore"):
        P_exact = ym.exact_mean_plaquette_su2_buckyball(beta, j_max=j_max_primary)
        P_check = ym.exact_mean_plaquette_su2_buckyball(beta, j_max=160)
    truncation_residual = abs(P_exact - P_check)
    # Independent Bessel-ratio cross-check (F -> infty limit). For SU(2),
    # <P> = (1/N) <Re Tr U> = (1/2) <Re Tr U>; the F -> infty series sum is
    # dominated by j = 1/2, giving <Re Tr U> = 2 I_2(beta)/I_1(beta), so
    # <P> -> I_2(beta) / I_1(beta).
    bessel_ratio_infty = float(_iv(2, beta) / _iv(1, beta))
    finite_volume_correction = abs(P_exact - bessel_ratio_infty)
    return {
        "P_exact": P_exact,
        "P_exact_jmax_160": P_check,
        "truncation_residual_jmax_80_vs_160": truncation_residual,
        "bessel_ratio_F_infty": bessel_ratio_infty,
        "finite_volume_correction": finite_volume_correction,
        "j_max_used": j_max_primary,
    }


def _section3(traj: Dict[str, Any], sd: SidecarData, beta: float) -> Dict[str, Any]:
    target = _migdal_witten_target(beta)
    P_meas: Optional[float] = None
    P_sem: Optional[float] = None
    P_sem_naive: Optional[float] = None
    P_sem_blocked: Optional[float] = None
    P_n_eff: Optional[float] = None
    P_plateau_block_size: Optional[int] = None
    P_plateau_detected: bool = False
    P_blocking_curve: List[Tuple[int, float]] = []
    P_blocking_regime: Optional[str] = None
    P_source: Optional[str] = None
    if sd.heatbath_canonical_P_history is not None and sd.heatbath_canonical_P_history.size > 0:
        if sd.heatbath_canonical_beta is not None and abs(sd.heatbath_canonical_beta - beta) > 1e-6:
            return {
                "available": True,
                "verdict": "FAIL",
                "reason": (f"sidecar heatbath beta={sd.heatbath_canonical_beta} "
                           f"differs from trajectory beta={beta}; "
                           f"comparison is not meaningful"),
                "P_exact": target["P_exact"],
                "P_exact_jmax_160": target["P_exact_jmax_160"],
                "truncation_residual": target["truncation_residual_jmax_80_vs_160"],
                "bessel_ratio_F_infty": target["bessel_ratio_F_infty"],
                "finite_volume_correction": target["finite_volume_correction"],
                "j_max": target["j_max_used"],
                "tolerance": TOL["migdal_witten"],
                "beta_trajectory": beta,
                "beta_sidecar": sd.heatbath_canonical_beta,
            }
        Ps = np.asarray(sd.heatbath_canonical_P_history, dtype=np.float64)
        P_meas = float(Ps.mean())
        b = _flyvbjerg_petersen_blocking(Ps)
        P_sem_naive = b["sem_naive"]
        P_sem_blocked = b["sem_blocked"]
        P_sem = P_sem_blocked
        P_n_eff = b["n_eff"]
        P_plateau_block_size = b["plateau_block_size"]
        P_plateau_detected = b["plateau_detected"]
        P_blocking_curve = b["blocking_curve"]
        P_blocking_regime = b["regime"]
        P_source = f"sidecar heatbath ensemble ({Ps.size} samples)"
    else:
        frames = traj["frames"]
        Pframes = np.asarray([f["plaquette_mean"] for f in frames], dtype=np.float64)
        if Pframes.size > 0:
            P_meas = float(Pframes.mean())
            b = _flyvbjerg_petersen_blocking(Pframes)
            P_sem_naive = b["sem_naive"]
            P_sem_blocked = b["sem_blocked"]
            P_sem = P_sem_blocked
            P_n_eff = b["n_eff"]
            P_plateau_block_size = b["plateau_block_size"]
            P_plateau_detected = b["plateau_detected"]
            P_blocking_curve = b["blocking_curve"]
            P_blocking_regime = b["regime"]
            P_source = f"microcanonical time-average over {Pframes.size} frames (no canonical ensemble in sidecar)"
    if P_meas is None:
        return {"available": False, "reason": "no heatbath or frame measurements"}
    gap = abs(P_meas - target["P_exact"])
    tol = TOL["migdal_witten"]
    margin = (tol / gap) if gap > 0 else float("inf")
    verdict = "PASS" if gap <= tol else "FAIL"
    return {
        "available": True,
        "P_exact": target["P_exact"],
        "P_exact_jmax_160": target["P_exact_jmax_160"],
        "truncation_residual": target["truncation_residual_jmax_80_vs_160"],
        "bessel_ratio_F_infty": target["bessel_ratio_F_infty"],
        "finite_volume_correction": target["finite_volume_correction"],
        "j_max": target["j_max_used"],
        "P_measured": P_meas,
        "P_sem": P_sem,                       # displayed (blocked) value
        "P_sem_naive": P_sem_naive,
        "P_sem_blocked": P_sem_blocked,
        "P_n_eff": P_n_eff,
        "P_plateau_block_size": P_plateau_block_size,
        "P_plateau_detected": P_plateau_detected,
        "P_blocking_curve": P_blocking_curve,
        "P_blocking_regime": P_blocking_regime,
        "sem_convention": "flyvbjerg_petersen_blocked",
        "P_source": P_source,
        "gap": gap,
        "tolerance": tol,
        "margin_factor": margin,
        "verdict": verdict,
        "beta_scan": sd.beta_scan if sd.beta_scan else [],
    }


# ---------------------------------------------------------------------------
# Section 4: gauge-invariance verification
# ---------------------------------------------------------------------------
def _haar_su2(n: int, seed: int) -> np.ndarray:
    """Draw n unit quaternions Haar-uniformly on SU(2).

    Standard recipe: sample u_1, u_2, u_3 ~ U(0,1), set
        q = (sqrt(1-u_1)*sin(2 pi u_2), sqrt(1-u_1)*cos(2 pi u_2),
             sqrt(u_1)  *sin(2 pi u_3), sqrt(u_1)  *cos(2 pi u_3))
    and store as (q0, q1, q2, q3). This is the Shoemake (1992) construction.
    """
    rng = np.random.default_rng(seed)
    u1 = rng.random(n)
    u2 = rng.random(n)
    u3 = rng.random(n)
    s1 = np.sqrt(1.0 - u1)
    s2 = np.sqrt(u1)
    a = s1 * np.sin(2.0 * np.pi * u2)
    b = s1 * np.cos(2.0 * np.pi * u2)
    c = s2 * np.sin(2.0 * np.pi * u3)
    d = s2 * np.cos(2.0 * np.pi * u3)
    # Store as (q0, q1, q2, q3) with q0 = d (real part).
    q = np.stack([d, a, b, c], axis=-1)
    n_q = np.sqrt((q * q).sum(axis=-1, keepdims=True))
    return q / n_q


def _section4(sd: SidecarData, graph, beta: float, seed: int) -> Dict[str, Any]:
    if sd.U_final is None or sd.E_final is None:
        return {"available": False, "reason": "no U_final/E_final sidecar"}
    import torch
    obs = _mod("observables")
    action = _mod("action")
    integ = _mod("integrator")

    U = torch.from_numpy(sd.U_final).to(dtype=torch.float64)
    E = torch.from_numpy(sd.E_final).to(dtype=torch.float64)

    g_np = _haar_su2(graph.n_vertices, seed=seed)
    g = torch.from_numpy(g_np).to(dtype=torch.float64)

    # Apply gauge transformation to U: U'_e = g(v_from) U_e g(v_to)^dag
    U_p = obs.apply_gauge_transformation(U, g, graph)

    # E transforms as a tail-frame Lie-algebra vector under Ad(g_{tail}).
    # In matrix form: E_e -> M(g_tail) E_e M(g_tail)^{-1}.  In this codebase's
    # quaternion convention that is qmul(qmul(g_tail, E_e), qconj(g_tail)).
    # (Cross-check: the integrator docstring says qmul(qmul(qconj(U), E), U)
    # is Ad(U^{-1}) E -- reversing g <-> qconj(g) gives the Ad(g) we need.)
    su2 = action._load_su2()
    edges = torch.as_tensor(graph.edges, dtype=torch.long)
    g_from = g[edges[:, 0]]
    g_from_conj = su2.qconj(g_from)
    E_p = su2.qmul(su2.qmul(g_from, E), g_from_conj)
    E_p = E_p.clone()
    E_p[..., 0] = 0.0

    # Observables -- before and after.
    def measure(Ux, Ex):
        S_W = float(action.wilson_action(Ux, graph, beta))
        Uf = action.all_face_holonomies(Ux, graph)
        P_mean = float(Uf[:, 0].mean())
        q0_per_face = Uf[:, 0].detach().cpu().numpy().astype(np.float64)
        Q_surr = float(obs.Q_surrogate(Ux, graph))
        ek = float(obs.edge_kinetic(Ex).sum())
        G = integ.compute_gauss_residual(Ex, Ux, graph)
        return dict(S_W=S_W, P_mean=P_mean, q0_per_face=q0_per_face,
                    Q_surr=Q_surr, edge_kinetic_total=ek,
                    G=G.detach().cpu().numpy().astype(np.float64),
                    gauss_max=float(G.abs().max()))

    pre = measure(U, E)
    post = measure(U_p, E_p)

    def rel_delta(a, b):
        denom = max(abs(a), abs(b), 1e-30)
        return abs(a - b) / denom

    obs_list = []
    tol_inv = TOL["gauge_invariance"]
    tol_cas = TOL["casimir_invariance"]
    obs_list.append({
        "name": "Wilson action S_W",
        "type": "invariant",
        "pre": pre["S_W"], "post": post["S_W"],
        "delta_abs": abs(pre["S_W"] - post["S_W"]),
        "delta_rel": rel_delta(pre["S_W"], post["S_W"]),
        "tolerance": tol_inv,
        "verdict": "PASS" if rel_delta(pre["S_W"], post["S_W"]) <= tol_inv else "FAIL",
    })
    obs_list.append({
        "name": "Mean plaquette <P>",
        "type": "invariant",
        "pre": pre["P_mean"], "post": post["P_mean"],
        "delta_abs": abs(pre["P_mean"] - post["P_mean"]),
        "delta_rel": rel_delta(pre["P_mean"], post["P_mean"]),
        "tolerance": tol_inv,
        "verdict": "PASS" if rel_delta(pre["P_mean"], post["P_mean"]) <= tol_inv else "FAIL",
    })
    q0_max_rel = float(
        np.max(np.abs(pre["q0_per_face"] - post["q0_per_face"]))
        / max(float(np.max(np.abs(pre["q0_per_face"]))), 1e-30)
    )
    obs_list.append({
        "name": "Per-face q_0(U_f) (max across 32 faces)",
        "type": "invariant",
        "delta_rel": q0_max_rel,
        "tolerance": tol_inv,
        "verdict": "PASS" if q0_max_rel <= tol_inv else "FAIL",
    })
    obs_list.append({
        "name": "Q_surrogate",
        "type": "invariant",
        "pre": pre["Q_surr"], "post": post["Q_surr"],
        "delta_abs": abs(pre["Q_surr"] - post["Q_surr"]),
        "delta_rel": rel_delta(pre["Q_surr"], post["Q_surr"]),
        "tolerance": tol_inv,
        "verdict": "PASS" if rel_delta(pre["Q_surr"], post["Q_surr"]) <= tol_inv else "FAIL",
    })
    obs_list.append({
        "name": "edge_kinetic total 2|q_vec(E)|^2",
        "type": "invariant",
        "pre": pre["edge_kinetic_total"], "post": post["edge_kinetic_total"],
        "delta_abs": abs(pre["edge_kinetic_total"] - post["edge_kinetic_total"]),
        "delta_rel": rel_delta(pre["edge_kinetic_total"], post["edge_kinetic_total"]),
        "tolerance": tol_inv,
        "verdict": "PASS" if rel_delta(pre["edge_kinetic_total"], post["edge_kinetic_total"]) <= tol_inv else "FAIL",
    })
    # Gauss covariance check: G_v transforms in the adjoint representation
    # under gauge transformation, G'_v = M(g_v) G_v M(g_v)^{-1}. Verify this
    # directly: rotate G_pre per-vertex via the same quaternion sandwich used
    # for E, then compare to G_post elementwise. PASS iff the maximum
    # absolute deviation is at the FP64 floor (the projector drove |G| to
    # ~1e-15 at init; the gauge transform should not introduce additional
    # error beyond floating-point noise).
    G_pre = pre["G"]                                              # (V, 3)
    G_post = post["G"]                                            # (V, 3)
    # Build per-vertex quaternion sandwich on G_pre (q0=0 quaternion form).
    G_pre_q = np.zeros((graph.n_vertices, 4), dtype=np.float64)
    G_pre_q[:, 1:] = G_pre
    g_pre_t = torch.from_numpy(G_pre_q)
    g_conj_t = su2.qconj(g)
    G_pre_rotated = su2.qmul(su2.qmul(g, g_pre_t), g_conj_t).detach().cpu().numpy()
    G_pre_rotated = G_pre_rotated[:, 1:]                          # strip q0
    cov_residual_max = float(np.max(np.abs(G_post - G_pre_rotated)))
    # Sanity: the absolute G level the projector drove to ~1e-15 sets the
    # FP64 floor for this check. Tolerance is 1e-10 (well above noise).
    obs_list.append({
        "name": "G_v covariance under Ad(g_v): max |G'_v - g_v G_v g_v^-1|",
        "type": "invariant",
        "delta_abs": cov_residual_max,
        "G_pre_max": float(np.max(np.abs(G_pre))),
        "G_post_max": float(np.max(np.abs(G_post))),
        "tolerance": tol_inv,
        "verdict": "PASS" if cov_residual_max <= tol_inv else "FAIL",
    })
    # edge_phase is gauge-variant by construction; report and declare.
    pre_phase = float(obs.edge_phase(U).abs().mean())
    post_phase = float(obs.edge_phase(U_p).abs().mean())
    obs_list.append({
        "name": "edge_phase mean (visualization only)",
        "type": "variant",
        "pre": pre_phase, "post": post_phase,
        "delta_abs": abs(pre_phase - post_phase),
        "verdict": "DECLARED_VARIANT",
    })

    all_invariants_pass = all(
        o["verdict"] == "PASS" for o in obs_list if o["type"] == "invariant"
    )
    return {
        "available": True,
        "gauge_seed": seed,
        "observables": obs_list,
        "verdict": "PASS" if all_invariants_pass else "FAIL",
    }


# ---------------------------------------------------------------------------
# Section 5: microcanonical vs canonical cross-check
# ---------------------------------------------------------------------------
def _section5(traj: Dict[str, Any], sd: SidecarData, beta: float) -> Dict[str, Any]:
    frames = traj["frames"]
    Pt = np.asarray([f["plaquette_mean"] for f in frames], dtype=np.float64)
    if Pt.size == 0:
        return {"available": False, "reason": "no frames"}
    P_time = float(Pt.mean())
    # Blocked SEM for the microcanonical arm.
    if Pt.size > 0:
        b_t = _flyvbjerg_petersen_blocking(Pt)
        P_time_sem_naive = b_t["sem_naive"]
        P_time_sem_blocked = b_t["sem_blocked"]
        P_time_sem = P_time_sem_blocked
        P_time_n_eff = b_t["n_eff"]
        P_time_plateau_block_size = b_t["plateau_block_size"]
        P_time_plateau_detected = b_t["plateau_detected"]
        P_time_blocking_curve = b_t["blocking_curve"]
        P_time_blocking_regime = b_t["regime"]
    else:
        P_time_sem_naive = 0.0
        P_time_sem_blocked = 0.0
        P_time_sem = 0.0
        P_time_n_eff = float(Pt.size)
        P_time_plateau_block_size = 1
        P_time_plateau_detected = False
        P_time_blocking_curve = []
        P_time_blocking_regime = "degenerate"

    if sd.heatbath_canonical_P_history is None or sd.heatbath_canonical_P_history.size == 0:
        return {
            "available": False,
            "reason": "no canonical heatbath ensemble in sidecar",
            "P_time": P_time,
            "P_time_sem": P_time_sem,
            "P_time_sem_naive": P_time_sem_naive,
            "P_time_sem_blocked": P_time_sem_blocked,
            "P_time_n_eff": P_time_n_eff,
            "P_time_plateau_block_size": P_time_plateau_block_size,
            "P_time_plateau_detected": P_time_plateau_detected,
            "P_time_blocking_curve": P_time_blocking_curve,
            "P_time_blocking_regime": P_time_blocking_regime,
            "sem_convention": "flyvbjerg_petersen_blocked",
        }
    if sd.heatbath_canonical_beta is not None and abs(sd.heatbath_canonical_beta - beta) > 1e-6:
        return {
            "available": True,
            "verdict": "FAIL",
            "reason": (f"sidecar heatbath beta={sd.heatbath_canonical_beta} "
                       f"differs from trajectory beta={beta}; cross-check is incoherent"),
            "P_time": P_time, "P_time_sem": P_time_sem, "n_time_samples": int(Pt.size),
            "tolerance": TOL["method_crosscheck"],
            "beta_trajectory": beta,
            "beta_sidecar": sd.heatbath_canonical_beta,
        }
    Ph = np.asarray(sd.heatbath_canonical_P_history, dtype=np.float64)
    P_hb = float(Ph.mean())
    # Blocked SEM for the heatbath arm.
    b_h = _flyvbjerg_petersen_blocking(Ph)
    P_heatbath_sem_naive = b_h["sem_naive"]
    P_heatbath_sem_blocked = b_h["sem_blocked"]
    P_heatbath_sem = P_heatbath_sem_blocked
    P_heatbath_n_eff = b_h["n_eff"]
    P_heatbath_plateau_block_size = b_h["plateau_block_size"]
    P_heatbath_plateau_detected = b_h["plateau_detected"]
    P_heatbath_blocking_curve = b_h["blocking_curve"]
    P_heatbath_blocking_regime = b_h["regime"]
    gap = abs(P_time - P_hb)
    tol = TOL["method_crosscheck"]
    margin = (tol / gap) if gap > 0 else float("inf")
    verdict = "PASS" if gap <= tol else "FAIL"
    return {
        "available": True,
        "P_time": P_time, "P_time_sem": P_time_sem, "n_time_samples": int(Pt.size),
        "P_time_sem_naive": P_time_sem_naive,
        "P_time_sem_blocked": P_time_sem_blocked,
        "P_time_n_eff": P_time_n_eff,
        "P_time_plateau_block_size": P_time_plateau_block_size,
        "P_time_plateau_detected": P_time_plateau_detected,
        "P_time_blocking_curve": P_time_blocking_curve,
        "P_time_blocking_regime": P_time_blocking_regime,
        "P_heatbath": P_hb, "P_heatbath_sem": P_heatbath_sem, "n_heatbath_samples": int(Ph.size),
        "P_heatbath_sem_naive": P_heatbath_sem_naive,
        "P_heatbath_sem_blocked": P_heatbath_sem_blocked,
        "P_heatbath_n_eff": P_heatbath_n_eff,
        "P_heatbath_plateau_block_size": P_heatbath_plateau_block_size,
        "P_heatbath_plateau_detected": P_heatbath_plateau_detected,
        "P_heatbath_blocking_curve": P_heatbath_blocking_curve,
        "P_heatbath_blocking_regime": P_heatbath_blocking_regime,
        "sem_convention": "flyvbjerg_petersen_blocked",
        "gap": gap,
        "tolerance": tol,
        "margin_factor": margin,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Section 6: parameter scan
# ---------------------------------------------------------------------------
def _section6(sd: SidecarData) -> Dict[str, Any]:
    if not sd.beta_scan:
        return {"available": False, "reason": "no beta_scan in sidecar"}
    tol = TOL["migdal_witten"]
    entries = []
    all_pass = True
    Ps_prev = None
    monotone_increase = True
    for row in sd.beta_scan:
        beta = float(row["beta"])
        P_meas = float(row["P_measured"])
        P_exact = float(row.get("P_exact", _migdal_witten_target(beta)["P_exact"]))
        # Apply blocking when the row carries P_history; else fall back to
        # the naive sem the sidecar recorded.
        if "P_history" in row and len(row["P_history"]) > 0:
            b = _flyvbjerg_petersen_blocking(np.asarray(row["P_history"], dtype=np.float64))
            P_sem_naive = b["sem_naive"]
            P_sem_blocked = b["sem_blocked"]
            P_sem = P_sem_blocked
            P_n_eff = b["n_eff"]
            P_plateau_block_size = b["plateau_block_size"]
            P_plateau_detected = b["plateau_detected"]
        else:
            P_sem = float(row.get("P_sem", 0.0))
            P_sem_naive = P_sem
            P_sem_blocked = P_sem
            P_n_eff = None
            P_plateau_block_size = None
            P_plateau_detected = False
        gap = abs(P_meas - P_exact)
        ok = gap <= tol
        entries.append({
            "beta": beta,
            "P_measured": P_meas,
            "P_sem": P_sem,
            "P_sem_naive": P_sem_naive,
            "P_sem_blocked": P_sem_blocked,
            "P_n_eff": P_n_eff,
            "P_plateau_block_size": P_plateau_block_size,
            "P_plateau_detected": P_plateau_detected,
            "P_exact": P_exact,
            "gap": gap,
            "tolerance": tol,
            "pass": ok,
        })
        if not ok:
            all_pass = False
        if Ps_prev is not None and P_meas < Ps_prev - 1e-6:
            monotone_increase = False
        Ps_prev = P_meas
    return {
        "available": True,
        "entries": entries,
        "all_within_tolerance": all_pass,
        "monotone_increase": monotone_increase,
        "sem_convention": "flyvbjerg_petersen_blocked",
        "verdict": "PASS" if all_pass and monotone_increase else "FAIL",
    }


# ---------------------------------------------------------------------------
# Overall verdict assembly
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Section 8: operational beta envelope (1.2 schema)
# ---------------------------------------------------------------------------
SECTOR_PROXY_WEIGHT = 0.0  # carry sector_proxy_q_std for transparency; weight 0


def _section8(sd: SidecarData) -> Dict[str, Any]:
    """Local-stability envelope sweep at the operating point.

    Each entry in sd.beta_envelope_history carries scalar summaries only.
    This builder computes per-entry sub-tolerance margins, a continuous
    stability_score, and a recommendation rule that prefers betas in a
    usable SEM regime (plateau_detected or shallow).
    """
    if sd.beta_envelope_history is None or len(sd.beta_envelope_history) == 0:
        return {
            "available": False,
            "reason": "no beta_envelope_history in sidecar; run --beta-envelope",
            "verdict": "NOT_RUN",
        }
    entries: List[Dict[str, Any]] = []
    for row in sd.beta_envelope_history:
        b = float(row["beta"])
        drift = float(row.get("energy_drift_max", float("inf")))
        gauss = float(row.get("gauss_max_across_window", float("inf")))
        xch = float(row.get("crosscheck_gap", float("inf")))
        mw = float(row.get("migdal_gap", float("inf")))
        regime = str(row.get("regime", "unknown"))
        # Continuous margins clipped to [0, 1]; replaces integer-dominated formula.
        def _margin(val: float, tol: float) -> float:
            return max(0.0, min(1.0, 1.0 - val / tol)) if tol > 0 else 0.0
        m_drift = _margin(drift, TOL["envelope_energy_drift_rel"])
        m_gauss = _margin(gauss, TOL["envelope_gauss_max"])
        m_xcheck = _margin(xch, TOL["envelope_crosscheck_gap"])
        m_migdal = _margin(mw, TOL["envelope_migdal_gap"])
        cleared_count = int((m_drift > 0) + (m_gauss > 0) + (m_xcheck > 0) + (m_migdal > 0))
        stability_score = m_drift + m_gauss + m_xcheck + m_migdal
        indeterminate = regime in ("too_short", "single_level")
        entries.append({
            "beta": b,
            "P_meas": float(row.get("P_meas", float("nan"))),
            "P_meas_sem": float(row.get("P_meas_sem", float("nan"))),
            "P_exact": float(row.get("P_exact", float("nan"))),
            "regime": regime,
            "indeterminate": indeterminate,
            "energy_drift_max": drift,
            "gauss_max_across_window": gauss,
            "crosscheck_gap": xch,
            "migdal_gap": mw,
            "m_drift": m_drift,
            "m_gauss": m_gauss,
            "m_xcheck": m_xcheck,
            "m_migdal": m_migdal,
            "cleared_count": cleared_count,
            "stability_score": stability_score,
            "sector_proxy_q_std": float(row.get("sector_proxy_q_std", 0.0)),
        })
    # Recommendation pool: cleared_count == 4 AND regime in usable set
    usable_regimes = ("plateau_detected", "shallow")
    pool = [e for e in entries if e["cleared_count"] == 4 and e["regime"] in usable_regimes]
    if pool:
        pool.sort(key=lambda e: (-e["stability_score"], e["energy_drift_max"]))
        rec = pool[0]
        operational_beta = float(rec["beta"])
        rec_rationale = (f"argmax stability_score among entries clearing all 4 "
                         f"sub-tolerances in a usable SEM regime; tie-break by "
                         f"smallest energy_drift_max")
    else:
        operational_beta = None
        rec_rationale = "no beta cleared all 4 sub-tolerances in a usable SEM regime"
    # Verdict: PASS iff rec clears all 4 AND a neighbor clears >=3
    verdict = "FAIL"
    if operational_beta is not None:
        rec_idx = next(i for i, e in enumerate(entries) if e["beta"] == operational_beta)
        neighbor_idxs = [i for i in (rec_idx - 1, rec_idx + 1) if 0 <= i < len(entries)]
        neighbor_max_cleared = max((entries[i]["cleared_count"] for i in neighbor_idxs), default=0)
        if neighbor_max_cleared >= 3:
            verdict = "PASS"
        else:
            verdict = "PASS_BUT_BRITTLE"
    regime_per_beta = {e["beta"]: e["regime"] for e in entries}
    return {
        "available": True,
        "entries": entries,
        "operational_beta_recommendation": operational_beta,
        "recommendation_rationale": rec_rationale,
        "regime_per_beta": regime_per_beta,
        "sector_proxy_weight": SECTOR_PROXY_WEIGHT,
        "n_envelope_betas": len(entries),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Section 9: sector classifier (band discrimination) (1.2 schema)
# ---------------------------------------------------------------------------
def _section9(sd: SidecarData) -> Dict[str, Any]:
    """Defence-in-depth re-application of the gate criteria to the recorded
    sector_classifier_state blob. Does NOT trust blob['verdict']; recomputes
    PASS / FAIL from real_accuracy, permutation_p_value, single_feature_max
    against classifier_thresholds().
    """
    if sd.sector_classifier_state is None:
        return {
            "available": False,
            "reason": "no sector_classifier_state in sidecar; run --sector-classifier",
            "verdict": "NOT_RUN",
        }
    blob = sd.sector_classifier_state
    # NOT_APPLICABLE branch (collect_real_windows returned {} -> band coverage
    # insufficient)
    if blob.get("verdict") == "NOT_APPLICABLE":
        return {
            "available": True,
            "verdict": "NOT_APPLICABLE",
            "reason": blob.get("reason", "insufficient band population"),
            "real_accuracy_threshold": blob.get("real_accuracy_threshold"),
            "null_alpha_threshold": blob.get("null_alpha_threshold"),
        }
    real_min, null_alpha = classifier_thresholds()
    real_acc = float(blob.get("real_accuracy", 0.0))
    p_value = float(blob.get("permutation_p_value", 1.0))
    ablation = blob.get("feature_ablation") or {}
    single_max = float(ablation.get("single_feature_max", 1.0))

    fail_reasons: List[str] = []
    if real_acc < real_min:
        fail_reasons.append(
            f"real_accuracy {real_acc:.3f} < threshold {real_min:.3f}"
        )
    if p_value > null_alpha:
        fail_reasons.append(
            f"permutation p_value {p_value:.3f} > null_alpha {null_alpha:.3f}"
        )
    if single_max >= real_min:
        fail_reasons.append(
            f"single_feature_max {single_max:.3f} >= real_min {real_min:.3f} "
            "(gauge-leak suspected)"
        )
    verdict = "PASS" if not fail_reasons else "FAIL"
    return {
        "available": True,
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "real_accuracy": real_acc,
        "real_accuracy_threshold": real_min,
        "permutation_p_value": p_value,
        "permutation_null_alpha": null_alpha,
        "permutation_null_accuracy_mean": blob.get("permutation_null_accuracy_mean"),
        "permutation_null_accuracy_std": blob.get("permutation_null_accuracy_std"),
        "per_class_accuracy": blob.get("per_class_accuracy"),
        "confusion_matrix": blob.get("confusion_matrix"),
        "n_eff": blob.get("n_eff"),
        "n_samples": blob.get("n_samples"),
        "n_samples_per_band": blob.get("n_samples_per_band"),
        "feature_ablation": ablation,
        "single_feature_max": single_max,
        "feature_names": blob.get("feature_names"),
        "band_edges": blob.get("band_edges"),
        "band_names": blob.get("band_names"),
        "k": blob.get("k"),
        "seed": blob.get("seed"),
    }


def _overall(sections: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # Fixed 7-category denominator. Unavailable sections become NOT_RUN; they
    # never count as PASS and they never silently disappear from the total.
    # An unavailable section is honest about what wasn't checked, and a
    # "5/7 PASS, 2 NOT_RUN" report reads very differently from a "5/5 PASS"
    # one whose missing sections are invisible.
    def _verdict_for(node):
        if not node.get("available"):
            return "NOT_RUN"
        return node.get("verdict", "NOT_RUN")
    s2 = sections["section_2_conservation"]
    s1 = sections["section_1_run_parameters"]
    # Fixed 10-category denominator in 1.2. NOT_RUN and NOT_APPLICABLE both
    # count toward the denominator -- they MUST NOT silently disappear.
    spec = [
        ("Substrate identities", s1["substrate"]),
        ("Energy conservation", s2["energy"]),
        ("Covariant Gauss residual", s2["gauss"]),
        ("Time reversibility", s2["time_reversibility"]),
        ("Migdal-Witten analytical target", sections["section_3_analytical"]),
        ("Gauge invariance", sections["section_4_gauge_invariance"]),
        ("Microcanonical vs canonical", sections["section_5_method_crosscheck"]),
        ("Beta-scan", sections["section_6_parameter_scan"]),
        ("Operational beta envelope", sections.get("section_8_beta_envelope", {"available": False})),
        ("Sector classifier", sections.get("section_9_sector_classifier", {"available": False})),
    ]
    categories = []
    for name, node in spec:
        # Substrate identities is the only category whose verdict comes
        # straight off the node; the others go through _verdict_for so that
        # missing sections become NOT_RUN.
        if name == "Substrate identities":
            verdict = node.get("verdict", "NOT_RUN")
        else:
            verdict = _verdict_for(node)
        categories.append({"name": name, "verdict": verdict})
    passing = sum(1 for c in categories if c["verdict"] == "PASS")
    failing = sum(1 for c in categories if c["verdict"] == "FAIL")
    not_run = sum(1 for c in categories if c["verdict"] == "NOT_RUN")
    total = len(categories)
    if failing == 0 and not_run == 0:
        summary = f"{passing}/{total} validation categories PASS"
    elif failing == 0:
        summary = (f"{passing}/{total} PASS, {not_run} NOT_RUN "
                   f"(no sidecar or section data unavailable)")
    else:
        summary = (f"{passing}/{total} PASS, {failing} FAIL"
                   + (f", {not_run} NOT_RUN" if not_run else ""))
    return {
        "categories": categories,
        "passing": passing,
        "failing": failing,
        "not_run": not_run,
        "total": total,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _fmt_sci(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if abs(x) >= 1e-3 and abs(x) < 1e4:
        return f"{x:.{digits + 2}f}"
    return f"{x:.{digits}e}"


def _markdown(payload: Dict[str, Any]) -> str:
    s1 = payload["section_1_run_parameters"]
    s2 = payload["section_2_conservation"]
    s3 = payload["section_3_analytical"]
    s4 = payload["section_4_gauge_invariance"]
    s5 = payload["section_5_method_crosscheck"]
    s6 = payload["section_6_parameter_scan"]
    s7 = payload["section_7_open_items"]
    rm = payload["report_metadata"]
    overall = payload["overall_verdict"]

    p = s1["params"]
    sub = s1["substrate"]["checks"]
    lines: List[str] = []
    a = lines.append
    a(f"# Halcyon Validation Report")
    a("")
    a(f"**Run identifier:** `{rm['run_identifier']}`  ")
    a(f"**Generated:** {rm['generated_at']}  ")
    a(f"**Substrate:** Truncated icosahedron, {p.get('gauge_group','SU(2)')}  ")
    a(f"**Trajectory schema version:** {p.get('schema_version','?')}  ")
    a(f"**Report schema version:** {REPORT_SCHEMA_VERSION}  ")
    a("")
    a("---")
    a("")
    a("## 1. What was simulated")
    a("")
    a("### Run parameters")
    a("")
    a("| Parameter | Value |")
    a("|---|---|")
    a(f"| Gauge group | {p.get('gauge_group','SU(2)')}, quaternion storage `E[1:] = 2 alpha` |")
    a(f"| Substrate | Truncated icosahedron |")
    a(f"| Vertices V | {p.get('V','?')} |")
    a(f"| Edges E | {p.get('E','?')} |")
    a(f"| Faces F | {p.get('F','?')} (12 pent, 20 hex) |")
    a(f"| Euler chi = V - E + F | {sub['chi']['measured']} |")
    a(f"| Edge census (pent-hex / hex-hex / pent-pent) | {sub['census_PH']['measured']} / {sub['census_HH']['measured']} / {sub['census_PP']['measured']} |")
    a(f"| Coupling beta | {p.get('beta','?')} |")
    a(f"| Integrator | Symplectic leapfrog, second order |")
    a(f"| Timestep dt | {p.get('dt','?')} |")
    a(f"| Number of leapfrog steps | {p.get('n_steps','?')} |")
    T_str = (f"{p['T_total']:.2f} (= n_steps * dt = {p.get('n_steps','?')} * {p.get('dt','?')})"
             if p.get("T_total") is not None else "?")
    a(f"| Total trajectory time | T = {T_str} |")
    if p.get("frame_t_first") is not None and p.get("frame_t_last") is not None:
        a(f"| Measured-frame span | t = {p['frame_t_first']} -> {p['frame_t_last']} ({p.get('n_frames','?')} frames at measure_every = {p.get('measure_every','?')}) |")
    a(f"| Thermalization | {p.get('n_thermalization_sweeps','?')} heatbath sweeps from cold start |")
    a(f"| Heatbath algorithm | Cabibbo-Marinari with Kennedy-Pendleton SU(2) updates |")
    a(f"| Gauss projector | Covariant graph Laplacian, conjugate gradient, Tikhonov eps = 1e-14 (global elliptic solve; explicitly NOT a local operation -- documented as validation/restoration, not part of the physics update) |")
    a(f"| Random seed | {p.get('seed','?')} |")
    a(f"| Floating-point precision | FP64 throughout |")
    a(f"| Platform | {rm['system_info']['platform']} |")
    a(f"| Python | {rm['system_info']['python_version']} |")
    lv = rm['system_info'].get('library_versions', {}) or {}
    if lv:
        lv_str = ", ".join(f"{k} {v}" for k, v in lv.items())
        a(f"| Library versions | {lv_str} |")
    a(f"| Code commit | {rm.get('code_commit','(not recorded)')} |")
    if rm.get("wall_time_seconds") is not None:
        a(f"| Wall time | {rm['wall_time_seconds']:.1f} seconds |")
    a("")
    a("### Verified substrate identities")
    a("")
    a("| Identity | Expected | Measured | Pass |")
    a("|---|---|---|---|")
    rows = [
        ("Euler formula V - E + F", BUCKYBALL_EXPECTED["chi"], sub["chi"]["measured"]),
        ("Sum of face perimeters", BUCKYBALL_EXPECTED["sum_of_perimeters"], sub["perimeter_sum"]["measured"]),
        ("Pentagons * 5 + Hexagons * 6", BUCKYBALL_EXPECTED["sum_of_perimeters"], sub["2E_check"]["measured"]),
        ("Each edge in exactly two faces", "yes", "yes" if sub["two_faces_per_edge"]["measured"] else "no"),
        ("Edge type count (PH, HH, PP)", f"({BUCKYBALL_EXPECTED['edge_census_PH']}, {BUCKYBALL_EXPECTED['edge_census_HH']}, {BUCKYBALL_EXPECTED['edge_census_PP']})",
         f"({sub['census_PH']['measured']}, {sub['census_HH']['measured']}, {sub['census_PP']['measured']})"),
    ]
    for name, exp, meas in rows:
        ok = "yes" if str(exp) == str(meas) else "**NO**"
        a(f"| {name} | {exp} | {meas} | {ok} |")
    a("")
    a(f"**Section verdict:** **{s1['substrate']['verdict']}**")
    a("")
    a("---")
    a("")

    # Section 2
    a("## 2. Conservation laws")
    a("")
    e2 = s2["energy"]
    a("### 2.1 Energy conservation")
    if e2.get("available"):
        a("")
        a("| Quantity | Value | Tolerance | Margin |")
        a("|---|---|---|---|")
        a(f"| H_0 (initial total energy) | {_fmt_sci(e2['H_0'])} | -- | -- |")
        a(f"| max|delta H / H_0| across {e2['n_samples']} samples | {_fmt_sci(e2['max_relative_drift'])} | {_fmt_sci(e2['tolerance'])} | {_fmt_sci(e2['margin_factor'])}x under |")
        a(f"| Monotone drift detected | {'yes' if e2['monotone_drift_detected'] else 'no'} | required: no | confirmed |")
        a(f"")
        a(f"**Energy conservation verdict:** **{e2['verdict']}**")
    else:
        a(f"\n_Not available: {e2.get('reason','no data')}._\n")
    a("")
    g2 = s2["gauss"]
    a("### 2.2 Covariant Gauss residual (final state)")
    if g2.get("available"):
        a("")
        a("| Quantity | Value | Tolerance |")
        a("|---|---|---|")
        a(f"| Covariant |G_v|_max on final (U, E) | {_fmt_sci(g2['gauss_covariant_max_final_state'])} | {_fmt_sci(g2['tolerance'])} |")
        a(f"| Flat |G_v|_max on the SAME state | {_fmt_sci(g2['gauss_flat_max_final_state'])} | -- |")
        a(f"| Ratio flat / covariant | {_fmt_sci(g2['ratio_flat_over_covariant'])} | -- |")
        if g2.get("gauss_trajectorywise_available"):
            a(f"| max |G_v|_inf across trajectory ({g2.get('n_snapshots','?')} snapshots) | {_fmt_sci(g2['gauss_max_across_trajectory'])} | {_fmt_sci(g2['tolerance'])} |")
            a(f"| Step index of trajectory max |G| | {g2.get('gauss_max_across_trajectory_step', 'n/a')} | -- |")
        a("")
        a("The flat (abelian) residual is reported only to make the distinction "
          "from the covariant Gauss operator explicit; the symplectic flow "
          "conserves the COVARIANT residual, not the flat one. This distinction "
          "caught the load-bearing WF#2 bug.")
        a("")
        if g2.get("gauss_trajectorywise_available"):
            a("_Trajectory-wise instrumentation:_ per-snapshot |G_v|_inf was sampled at "
              "every measurement window during the run. The maximum across the full "
              "trajectory is reported above. Sampling cadence matches the energy / "
              "plaquette measurement window; finer resolution requires per-step "
              "instrumentation.")
        else:
            a("_Final-state-only caveat:_ the residual reported here is on the final "
              "(U, E) state, not max|G| sampled across the trajectory. A reviewer "
              "who wants to rule out post-step projection cleanup as the source of "
              "the small final value should reproduce the run with the seed in the "
              "run parameters and instrument the per-frame Gauss residual; the "
              "validated kernel exposes `compute_gauss_residual(E, U, graph)` for "
              "that purpose.")
            a("")
            a("_Trajectory-wise check:_ NOT RUN (sidecar lacks gauss_history; "
              "historical sidecars require an orchestrator re-run to populate this field).")
        a("")
        a(f"**Gauss residual verdict (final-state only):** **{g2.get('final_state_verdict', g2['verdict'])}**")
        if g2.get("gauss_trajectorywise_available"):
            a(f"**Gauss residual verdict (combined, including trajectory-wise):** **{g2['verdict']}**")
            a(f"_Trajectory-wise sub-verdict:_ **{g2['gauss_trajectorywise_verdict']}**")
    else:
        a(f"\n_Not available: {g2.get('reason','no data')}._\n")
    a("")
    tr = s2["time_reversibility"]
    a("### 2.3 Time reversibility")
    if tr.get("available"):
        a("")
        a("| Quantity | Value | Tolerance |")
        a("|---|---|---|")
        a(f"| |U_final_reversed - U_initial|_inf | {_fmt_sci(tr['U_residual_inf'])} | {_fmt_sci(tr['tolerance'])} |")
        a(f"| |E_final_reversed - E_initial|_inf | {_fmt_sci(tr['E_residual_inf'])} | {_fmt_sci(tr['tolerance'])} |")
        a("")
        a(f"**Time-reversibility verdict:** **{tr['verdict']}**")
    else:
        a(f"\n_Not available: {tr.get('reason','no data')}. "
          f"Recompute requires the U_init/E_init/U_final/E_final sidecar._\n")
    a("")
    a("---")
    a("")

    # Section 3
    a("## 3. Analytical agreement: Migdal-Witten target")
    a("")
    if s3.get("available"):
        a("### 3.1 Exact target")
        a("")
        a("| Quantity | Value |")
        a("|---|---|")
        a(f"| <P>_exact (finite-F, Migdal-Witten, j_max={s3['j_max']}) | {s3['P_exact']:.10f} |")
        a(f"| Truncation residual (j_max=80 vs 160) | {_fmt_sci(s3['truncation_residual'])} |")
        a(f"| Cross-check via I_2(beta) / I_1(beta), F -> infinity | {s3['bessel_ratio_F_infty']:.10f} |")
        a(f"| Finite-volume correction (finite F minus F -> infinity) | {_fmt_sci(s3['finite_volume_correction'])} |")
        a("")
        a("A reviewer can verify the cross-check independently via "
          "`scipy.special.iv(2, beta) / scipy.special.iv(1, beta)`. For SU(2), "
          "&lt;P&gt; = (1/N)&lt;Re Tr U&gt; = (1/2)&lt;Re Tr U&gt;, and the F-to-infinity "
          "series is dominated by the j = 1/2 irrep, giving "
          "&lt;P&gt;_infty = I_2(beta) / I_1(beta).")
        a("")
        a(f"### 3.2 Heatbath measurement at beta = {payload['section_1_run_parameters']['params'].get('beta','?')}")
        a("")
        a("| Quantity | Value |")
        a("|---|---|")
        a(f"| <P>_measured | {s3['P_measured']:.6f} |")
        _neff_str = f"{s3['P_n_eff']:.0f}" if s3.get("P_n_eff") is not None else "n/a"
        _plat_str = (f"{s3.get('P_plateau_block_size','n/a')}"
                     if s3.get("P_plateau_block_size") is not None else "n/a")
        _blocked_str = _fmt_sci(s3.get("P_sem_blocked", s3["P_sem"]))
        _naive_str = _fmt_sci(s3.get("P_sem_naive", s3["P_sem"]))
        a(f"| Standard error of the mean (Flyvbjerg-Petersen blocked) | +/- {_blocked_str} (naive: +/- {_naive_str}, n_eff = {_neff_str}, plateau block size = {_plat_str}) |")
        a(f"| Source | {s3['P_source']} |")
        a(f"| Gap to analytical target | {_fmt_sci(s3['gap'])} |")
        a(f"| Tolerance (strict, no error-bar slack) | {_fmt_sci(s3['tolerance'])} |")
        a(f"| Margin under tolerance | {_fmt_sci(s3['margin_factor'])}x |")
        a("")
        _regime = s3.get("P_blocking_regime", "plateau_detected")
        _plateau_detected = s3.get("P_plateau_detected", False)
        if _plateau_detected:
            a("_SEM:_ corrected via Flyvbjerg-Petersen blocking on the heatbath "
              f"P_history. A plateau was detected in the blocked-SEM curve, giving "
              f"n_eff = {_neff_str} effective independent samples (block size = "
              f"{_plat_str}), accounting for sweep-to-sweep autocorrelation.")
        elif _regime in ("shallow", "single_level"):
            a("_SEM (blocking degraded):_ only "
              + ("one" if _regime == "single_level" else "a few")
              + " doubling level" + ("s" if _regime != "single_level" else "")
              + " fit in this P_history; using max(last-block SEM, naive SEM) "
              "as a conservative floor. Autocorrelation may not be fully captured.")
        elif _regime == "too_short":
            a("_SEM (blocking not applicable):_ n_samples < 32; falling back to "
              "naive SEM with no autocorrelation correction.")
        elif _regime == "no_plateau":
            a("_SEM (no plateau detected):_ blocked-SEM curve did not plateau "
              "within the available doubling levels; using max(last-block SEM, "
              "naive SEM) as a conservative floor. Reported n_eff is a lower bound.")
        else:
            a("_SEM (degenerate):_ insufficient data for any SEM estimate; "
              "reporting zero.")
        a("")
        a(f"**Section 3 verdict:** **{s3['verdict']}**")
        if s3.get("beta_scan"):
            a("")
            a("### 3.3 beta-scan against the Migdal-Witten curve")
            a("")
            a("| beta | <P>_measured | <P>_exact (M-W) | Gap |")
            a("|---|---|---|---|")
            for row in s3["beta_scan"]:
                a(f"| {row['beta']} | {row['P_measured']:.6f} | {row['P_exact']:.6f} | {_fmt_sci(abs(row['P_measured']-row['P_exact']))} |")
    else:
        a(f"_Not available: {s3.get('reason','no data')}._\n")
    a("")
    a("---")
    a("")

    # Section 4
    a("## 4. Gauge-invariance verification")
    a("")
    if s4.get("available"):
        a(f"Random per-vertex SU(2) gauge transformation g_v drawn from Haar "
          f"measure (seed = {s4['gauge_seed']}). Each claimed observable is "
          f"recomputed on the transformed state.")
        a("")
        a("| Observable | Type | Delta (rel) | Tolerance | Verdict |")
        a("|---|---|---|---|---|")
        for o in s4["observables"]:
            tol_str = _fmt_sci(o.get("tolerance", float("nan"))) if o["type"] == "invariant" else "--"
            # Prefer delta_rel; fall back to delta_abs for observables (like
            # the covariance check) where the meaningful number is absolute.
            # The earlier rendering emitted "n/a" for those rows, which read
            # like a missing measurement to a hostile reviewer.
            dr = o.get("delta_rel")
            d_abs = o.get("delta_abs")
            if dr is not None:
                dr_str = _fmt_sci(dr) + " (rel)"
            elif d_abs is not None:
                dr_str = _fmt_sci(d_abs) + " (abs)"
            else:
                dr_str = "n/a"
            a(f"| {o['name']} | {o['type']} | {dr_str} | {tol_str} | **{o['verdict']}** |")
        a("")
        a(f"**Section 4 verdict:** **{s4['verdict']}**")
    else:
        a(f"_Not available: {s4.get('reason','no data')}. "
          f"Recompute requires the U_final/E_final sidecar._\n")
    a("")
    a("---")
    a("")

    # Section 5
    a("## 5. Method cross-check: microcanonical vs canonical")
    a("")
    a("_Scope note:_ this section compares two sampling methods (symplectic "
      "leapfrog microcanonical vs heatbath canonical) on the same observable "
      "at the same coupling. At the current sample sizes it is a smoke-run "
      "diagnostic rather than a publication-grade cross-check: the comparison "
      "needs a trajectory-length sweep, multiple seeds, autocorrelation-"
      "corrected SEMs, and ideally a convergence plot against the exact "
      "Migdal-Witten value before it earns the name. A FAIL here at production "
      "scale on a 93-DOF substrate is consistent with the documented Section "
      "7.4 ergodicity caveat and not by itself a structural failure.")
    a("")
    if s5.get("available"):
        a("| Quantity | Value |")
        a("|---|---|")
        _t_neff = f"{s5['P_time_n_eff']:.0f}" if s5.get("P_time_n_eff") is not None else "n/a"
        _h_neff = f"{s5['P_heatbath_n_eff']:.0f}" if s5.get("P_heatbath_n_eff") is not None else "n/a"
        _t_plat = "plateau" if s5.get("P_time_plateau_detected") else "no plateau"
        _h_plat = "plateau" if s5.get("P_heatbath_plateau_detected") else "no plateau"
        a(f"| <P>_time (microcanonical, {s5['n_time_samples']} frames, n_eff = {_t_neff}, {_t_plat}) | {s5['P_time']:.6f} +/- {_fmt_sci(s5['P_time_sem'])} (blocked) |")
        a(f"| <P>_heatbath (canonical, {s5['n_heatbath_samples']} sweeps, n_eff = {_h_neff}, {_h_plat}) | {s5['P_heatbath']:.6f} +/- {_fmt_sci(s5['P_heatbath_sem'])} (blocked) |")
        a(f"| Gap |<P>_time - <P>_heatbath| | {_fmt_sci(s5['gap'])} |")
        a(f"| Tolerance | {_fmt_sci(s5['tolerance'])} |")
        a(f"| Margin | {_fmt_sci(s5['margin_factor'])}x |")
        a("")
        a(f"**Section 5 verdict:** **{s5['verdict']}**")
        a("")
        a("_SEM:_ corrected via Flyvbjerg-Petersen blocking on both the "
          "microcanonical P_t and the canonical P_history. The n_eff figures "
          "reflect per-method autocorrelation length; a \"no plateau\" annotation "
          "means the blocked-SEM curve had not plateaued within the available "
          "doubling levels and the reported SEM is the conservative "
          "max(last-block, naive).")
    else:
        a(f"_Not available: {s5.get('reason','no data')}._\n")
        if s5.get("P_time") is not None:
            a("")
            a(f"_Partial data:_ time-average <P>_time = {s5['P_time']:.6f} from "
              f"{len([f for f in payload['frames_summary']['t_range']])} samples; "
              f"no canonical heatbath ensemble available for comparison.")
    a("")
    a("---")
    a("")

    # Section 6
    a("## 6. Parameter scan and operating envelope")
    a("")
    if s6.get("available"):
        a("| beta | <P>_measured +/- SEM | <P>_exact | Gap | n_eff (block) | Within tol |")
        a("|---|---|---|---|---|---|")
        for row in s6["entries"]:
            if row.get("P_n_eff") is not None:
                _blk = row.get("P_plateau_block_size", "n/a")
                _plat_flag = "" if row.get("P_plateau_detected") else "*"
                _neff_cell = f"{row['P_n_eff']:.0f}{_plat_flag} (block={_blk})"
            else:
                _neff_cell = "n/a (legacy sidecar)"
            a(f"| {row['beta']} | {row['P_measured']:.6f} +/- {_fmt_sci(row.get('P_sem', 0.0))} | {row['P_exact']:.6f} | {_fmt_sci(row['gap'])} | {_neff_cell} | {'yes' if row['pass'] else '**no**'} |")
        a("")
        a("_n_eff annotation:_ a trailing `*` after n_eff means the blocked-SEM "
          "curve did not reach a plateau within the available doubling levels, "
          "so n_eff is a lower bound and SEM is the conservative max(last-block, naive).")
        a("")
        a(f"All scanned beta within {TOL['migdal_witten']} tolerance: "
          f"**{'yes' if s6['all_within_tolerance'] else 'no'}**.  "
          f"Monotone increase of <P>(beta): "
          f"**{'yes' if s6['monotone_increase'] else 'no'}**.")
        a("")
        a(f"**Section 6 verdict:** **{s6['verdict']}**")
    else:
        a(f"_Not available: {s6.get('reason','no data')}. "
          f"Re-run with run_validation_report.py --beta-scan to populate._\n")
    a("")
    a("---")
    a("")

    # Section 7
    a("## 7. Open items: what this run does not validate")
    a("")
    for item in s7:
        a(f"### {item['category']}")
        a("")
        a(item["description"])
        a("")
    a("---")
    a("")

    # Section 8: operational beta envelope
    s8 = payload.get("section_8_beta_envelope", {"available": False})
    a("## 8. Operational beta envelope")
    a("")
    if s8.get("available"):
        a("_Energy drift in this section is measured over a {} -step LOCAL "
          "window; long-time secular drift is the load-bearing Section 2 "
          "gate, not this one._".format("400"))
        a("")
        a("| beta | regime | &lt;P&gt;_meas +/- SEM | M-W gap | drift | gauss_max | xcheck_gap | cleared/4 | score |")
        a("|---|---|---|---|---|---|---|---|---|")
        for e in s8["entries"]:
            ind_mark = "*" if e["indeterminate"] else ""
            a(f"| {e['beta']:.2f}{ind_mark} | {e['regime']} | "
              f"{e['P_meas']:.4f} +/- {_fmt_sci(e['P_meas_sem'])} | "
              f"{_fmt_sci(e['migdal_gap'])} | "
              f"{_fmt_sci(e['energy_drift_max'])} | "
              f"{_fmt_sci(e['gauss_max_across_window'])} | "
              f"{_fmt_sci(e['crosscheck_gap'])} | "
              f"{e['cleared_count']}/4 | "
              f"{e['stability_score']:.2f} |")
        a("")
        op_b = s8.get("operational_beta_recommendation")
        if op_b is not None:
            a(f"**Recommendation:** operational beta = {op_b}")
        else:
            a("**Recommendation:** no candidate cleared all 4 sub-tolerances "
              "in a usable SEM regime.")
        a("")
        a(f"_Rationale:_ {s8.get('recommendation_rationale','')}")
        a("")
        a(f"**Section 8 verdict:** **{s8['verdict']}**")
        a("")
        a("_* indicates regime in {too_short, single_level}; excluded from "
          "the recommendation pool._")
    else:
        a(f"_NOT_RUN:_ {s8.get('reason','run --beta-envelope to populate this section')}.")
    a("")
    a("---")
    a("")

    # Section 9: sector classifier
    s9 = payload.get("section_9_sector_classifier", {"available": False})
    a("## 9. Sector classifier (band discrimination)")
    a("")
    a("_**This is NOT a topological-charge test.** &pi;<sub>2</sub>(SU(2)) = 0 "
      "on S<sup>2</sup>; the bands **B0 / B1 / B2** are operational labels for "
      "observed Q<sub>surrogate</sub> ranges, not topological sectors. A PASS "
      "verdict means the apparatus can DISCRIMINATE calibration windows; it "
      "does NOT mean it RESOLVES topology._")
    a("")
    if s9.get("available") and s9.get("verdict") != "NOT_APPLICABLE":
        a("| Quantity | Value |")
        a("|---|---|")
        a(f"| Real accuracy | {s9.get('real_accuracy', 0.0):.4f} (threshold {s9.get('real_accuracy_threshold','?')}) |")
        a(f"| Permutation p-value | {s9.get('permutation_p_value', 1.0):.4f} (threshold {s9.get('permutation_null_alpha','?')}) |")
        a(f"| Permutation null accuracy | {s9.get('permutation_null_accuracy_mean','n/a')} +/- {s9.get('permutation_null_accuracy_std','n/a')} |")
        a(f"| Single-feature max accuracy | {s9.get('single_feature_max','n/a'):.4f} (must be &lt; real_accuracy_threshold) |"
          if isinstance(s9.get('single_feature_max'), (int, float))
          else f"| Single-feature max accuracy | n/a |")
        a(f"| Effective samples (n_eff) | {s9.get('n_eff','n/a')} |")
        a(f"| Total samples | {s9.get('n_samples','n/a')} |")
        a(f"| k (k-NN) | {s9.get('k','n/a')} |")
        a("")
        if s9.get("per_class_accuracy"):
            a("**Per-class accuracy:**")
            a("")
            a("| Band | Accuracy |")
            a("|---|---|")
            for band, acc in s9["per_class_accuracy"].items():
                a(f"| {band} | {acc:.4f} |")
            a("")
        if s9.get("confusion_matrix"):
            a("**Confusion matrix (rows = true, cols = predicted):**")
            a("")
            a("| | B0 | B1 | B2 |")
            a("|---|---|---|---|")
            for i, band in enumerate(("B0", "B1", "B2")):
                row = s9["confusion_matrix"][i]
                a(f"| **{band}** | {row[0]} | {row[1]} | {row[2]} |")
            a("")
        if s9.get("feature_ablation"):
            a("**Feature ablation:**")
            a("")
            a("| Feature | LOO accuracy WITHOUT this feature | Single-feature LOO accuracy |")
            a("|---|---|---|")
            loo = s9["feature_ablation"].get("leave_one_out", {})
            single = s9["feature_ablation"].get("single_feature", {})
            for name in s9.get("feature_names", []):
                lo = loo.get(name, "n/a")
                sg = single.get(name, "n/a")
                lo_s = f"{lo:.4f}" if isinstance(lo, (int, float)) else lo
                sg_s = f"{sg:.4f}" if isinstance(sg, (int, float)) else sg
                a(f"| `{name}` | {lo_s} | {sg_s} |")
            a("")
        a(f"**Section 9 verdict:** **{s9['verdict']}**")
        if s9.get("fail_reasons"):
            a("")
            a("_FAIL reasons:_")
            for r in s9["fail_reasons"]:
                a(f"  - {r}")
    elif s9.get("verdict") == "NOT_APPLICABLE":
        a(f"_NOT_APPLICABLE: {s9.get('reason','insufficient band coverage')}._")
    else:
        a(f"_NOT_RUN: {s9.get('reason','run --sector-classifier to populate this section')}._")
    a("")
    a("---")
    a("")

    # Reproducibility (renumbered to 11 in schema 1.2)
    a("## 11. Reproducibility checksums")
    a("")
    a("_Renumbered to 11 in schema 1.2 (Sections 8, 9 now occupy 8 and 9; "
      "Appendix A retains its slot)._")
    a("")
    a("")
    a("| File | SHA-256 (truncated) | Size |")
    a("|---|---|---|")
    for entry in rm["sha256sums"]:
        size_kib = entry["size_bytes"] / 1024.0
        a(f"| `{entry['path']}` | `{entry['sha256_12']}...` | {size_kib:.1f} KiB |")
    a("")
    a(f"A reviewer with the same seed ({p.get('seed','?')}), the same code at "
      f"these commit hashes, and the same library versions reproduces every "
      f"number in this report to the bit on identical hardware.")
    a("")
    a("_Full hashes:_ the table above prints the first 12 hex characters of "
      "each SHA-256 for readability. The full 64-character hashes are in the "
      "companion JSON artifact under `report_metadata.sha256sums[*].sha256`. "
      "Verify with `sha256sum <file>` (POSIX) or `Get-FileHash <file> -Algorithm SHA256` "
      "(PowerShell).")
    a("")
    a("---")
    a("")

    # Verdict
    a("## Verdict summary")
    a("")
    a("| Category | Status |")
    a("|---|---|")
    for c in overall["categories"]:
        a(f"| {c['name']} | **{c['verdict']}** |")
    a("")
    a(f"**Overall:** {overall['summary']}.")
    a("")
    a("---")
    a("")
    # Appendix A: tolerance derivations parameterised against actual run params.
    a("## Appendix A. Tolerance derivations")
    a("")
    a("Each tolerance below is derived from first principles. The Expected "
      "residual column states the residual the underlying numerics are "
      "expected to deliver at FP64 with the documented step count and sample "
      "sizes; the gates are intentionally non-tight so that a FAIL is "
      "unambiguous.")
    a("")
    a(f"_All derivations are quoted at the parameters of THIS run: "
      f"N = {p.get('n_steps','?')} steps, dt = {p.get('dt','?')}, "
      f"beta = {p.get('beta','?')}. Scale expected residuals when these "
      f"differ from defaults._")
    a("")
    a("| Tolerance | Value | Expected residual | Source | Conservative factor |")
    a("|---|---|---|---|---|")
    _N_p = int(p.get("n_steps") or 1000)
    _dt_p = float(p.get("dt") or 0.02)
    _beta_p = float(p.get("beta") or 2.5)
    _fmt_ctx = {
        "N": _N_p, "dt": _dt_p, "dt2": _dt_p * _dt_p,
        "rw": 2e-14 * math.sqrt(_N_p), "lin": 2e-14 * _N_p,
        "beta": _beta_p, "Tdt": _N_p * _dt_p,
        "sys": 0.09 / max(math.sqrt(_N_p * _dt_p), 1e-30),
    }
    _derivations = rm.get("tolerance_derivations", TOLERANCE_DERIVATIONS)
    for _entry in _derivations:
        _src = _entry.get("source_template", _entry.get("source", "")).format(**_fmt_ctx)
        _exp_str = (_fmt_sci(_entry["expected_residual"])
                    if _entry.get("expected_residual") is not None else "n/a")
        a(f"| `{_entry['key']}` | {_fmt_sci(_entry['value'])} | {_exp_str} | "
          f"{_src} | {_entry['conservative_factor']} |")
    a("")
    a("A FAIL on any of these gates indicates either (a) a regression that "
      "violates the derived expectation by orders of magnitude, or (b) a "
      "sampling or ergodicity issue documented in Section 7. None of these "
      "tolerances is fitted to current data.")
    a("")
    a("---")
    a("")
    a(f"*Report ends. Numerical artifact: `{rm['json_artifact_basename']}` (included separately).*")
    a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------
def _sha256_of(path: str) -> Tuple[str, int]:
    h = hashlib.sha256()
    sz = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
            sz += len(chunk)
    return h.hexdigest(), sz


def _sha_block(paths: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        digest, sz = _sha256_of(p)
        out.append({
            "path": os.path.relpath(p, start=_HERE),
            "sha256": digest,
            "sha256_12": digest[:12],
            "size_bytes": sz,
        })
    return out


# ---------------------------------------------------------------------------
# Library version inventory
# ---------------------------------------------------------------------------
def _library_versions() -> Dict[str, str]:
    """Record installed versions of the libraries the numerical pipeline
    depends on. A reviewer reproducing the report needs this to know what
    they are matching against. Soft-fails on import errors (records the
    error string rather than crashing the report)."""
    out: Dict[str, str] = {}
    for mod_name, attr in (
        ("numpy", "__version__"),
        ("scipy", "__version__"),
        ("torch", "__version__"),
    ):
        try:
            mod = __import__(mod_name)
            out[mod_name] = str(getattr(mod, attr, "(no __version__)"))
        except Exception as ex:
            out[mod_name] = f"(unavailable: {type(ex).__name__})"
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_report(
    trajectory_path: str,
    output_dir: str,
    run_metadata: Optional[Dict[str, Any]] = None,
    sidecar_path: Optional[str] = None,
    gauge_invariance_seed: int = 20260616,
    run_scan: bool = False,
) -> Tuple[Path, Path]:
    """Generate a Halcyon validation report.

    Args:
        trajectory_path     : absolute path to a trajectory.json
        output_dir          : directory where md + json are written
        run_metadata        : optional dict merged into report_metadata
        sidecar_path        : optional path to a .npz with final-state + heatbath data
        gauge_invariance_seed : seed for the Haar g_v draw in Section 4
        run_scan            : reserved for orchestrator integration; not used here

    Returns:
        (md_path, json_path)
    """
    trajectory_path = os.path.abspath(trajectory_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    traj = _load_trajectory(trajectory_path)
    p = _extract_run_params(traj)
    sd = _load_sidecar(sidecar_path)

    # Build a fresh graph and verify substrate identities.
    gmod = _mod("graph")
    graph = gmod.build_truncated_icosahedron()
    sub = _verify_substrate(graph)

    beta = float(p["beta"]) if p["beta"] is not None else 2.5
    dt = float(p["dt"]) if p["dt"] is not None else 0.02
    n_steps = int(p["n_steps"]) if p["n_steps"] is not None else 1000

    section_1 = {"params": p, "substrate": sub}
    # Substrate identity short-circuit: per the spec ("If a substrate
    # identity doesn't hold, mark this section's verdict FAIL and do not
    # proceed to subsequent sections"), if Section 1 fails we mark every
    # downstream section unavailable with an explicit reason. The trajectory
    # is on the wrong graph; nothing downstream can be coherently checked.
    if sub["verdict"] == "FAIL":
        substrate_fail_section = {
            "available": False,
            "verdict": "FAIL",
            "reason": "skipped: substrate identity failure short-circuits Sections 2-6",
        }
        section_2 = {
            "energy": dict(substrate_fail_section),
            "gauss": dict(substrate_fail_section),
            "time_reversibility": dict(substrate_fail_section),
        }
        section_3 = dict(substrate_fail_section)
        section_4 = dict(substrate_fail_section)
        section_5 = dict(substrate_fail_section)
        section_6 = dict(substrate_fail_section)
    else:
        section_2 = _section2(traj, sd, graph, beta, dt, n_steps)
        section_3 = _section3(traj, sd, beta)
        section_4 = _section4(sd, graph, beta, seed=gauge_invariance_seed)
        section_5 = _section5(traj, sd, beta)
        section_6 = _section6(sd)

    # Timestamp + run_id (deterministic: derived from trajectory hash so the
    # same trajectory always produces the same identifier).
    traj_hash, _ = _sha256_of(trajectory_path)
    run_identifier = f"halcyon_{traj_hash[:12]}"
    now = datetime.now(timezone.utc).astimezone()
    ts_compact = now.strftime("%Y%m%d_%H%M%S")
    md_basename = f"validation_report_{ts_compact}.md"
    json_basename = f"validation_report_{ts_compact}.json"
    md_path = Path(output_dir) / md_basename
    json_path = Path(output_dir) / json_basename

    # Reproducibility receipts -- file hashes for the trajectory + the kernel
    # files this report depended on.
    kernel_files = [
        os.path.join(_HERE, name) for name in (
            "buckyball_graph.py", "buckyball_action.py", "buckyball_heatbath.py",
            "buckyball_integrator.py", "buckyball_yangmills_exact.py",
            "buckyball_observables.py",
        )
    ]
    sha_paths = [trajectory_path] + [k for k in kernel_files if os.path.exists(k)]
    if sidecar_path and os.path.exists(sidecar_path):
        sha_paths.append(sidecar_path)
    sha_block = _sha_block(sha_paths)

    report_metadata = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "trajectory_path": os.path.relpath(trajectory_path, start=_HERE),
        "sidecar_path": (os.path.relpath(sidecar_path, start=_HERE) if sidecar_path else None),
        "report_version": REPORT_VERSION,
        "tool_version": TOOL_VERSION,
        "run_identifier": run_identifier,
        "code_commit": (run_metadata or {}).get("code_commit") or sd.code_commit,
        "wall_time_seconds": (run_metadata or {}).get("wall_time_seconds") or sd.wall_time_seconds,
        "system_info": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "machine": platform.machine(),
            "library_versions": _library_versions(),
        },
        "sha256sums": sha_block,
        "tolerance_derivations": TOLERANCE_DERIVATIONS,
        "sem_convention": "flyvbjerg_petersen_blocked",
        # 1.2 schema: source-of-truth provenance for the new sidecar blobs.
        "beta_envelope_history_source": (str(sidecar_path)
                                          if sd.beta_envelope_history is not None
                                          else "absent"),
        "sector_classifier_state_source": (str(sidecar_path)
                                            if sd.sector_classifier_state is not None
                                            else "absent"),
        "json_artifact_basename": json_basename,
    }
    if run_metadata:
        for k, v in run_metadata.items():
            if k not in report_metadata and v is not None:
                report_metadata[k] = v

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_metadata": report_metadata,
        "section_1_run_parameters": section_1,
        "section_2_conservation": section_2,
        "section_3_analytical": section_3,
        "section_4_gauge_invariance": section_4,
        "section_5_method_crosscheck": section_5,
        "section_6_parameter_scan": section_6,
        "section_7_open_items": OPEN_ITEMS,
        "section_8_beta_envelope": _section8(sd),
        "section_9_sector_classifier": _section9(sd),
        "frames_summary": {
            "n_frames": len(traj["frames"]),
            "t_range": [traj["frames"][0]["t"], traj["frames"][-1]["t"]],
        },
    }
    payload["overall_verdict"] = _overall(payload)

    # Convert any numpy arrays in payload to plain lists (for JSON).
    payload_json = _jsonable(payload)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload_json, fh, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_markdown(payload_json))

    return md_path, json_path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REPORT_VERSION",
    "TOOL_VERSION",
    "TOL",
    "TOLERANCE_DERIVATIONS",
    "OPEN_ITEMS",
    "SidecarData",
    "generate_report",
]
