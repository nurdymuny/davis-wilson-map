"""phase_runners.py -- one function per orchestrator phase the
``--use-gigi`` flag routes through GIGI's HTTP/GQL surface.

Each runner has the SAME return shape as the Python kernel phase it
replaces, so the dispatcher in ``run_validation_report.py`` can swap
the body of an ``if args.use_gigi:`` branch without touching phases 5
/ 8c / 9 / 10 / 11-13 downstream. The shape contract is:

    declare_lattice_phase          -> str (lattice_name)
    thermalize_via_gigi            -> {U_handle, U_buf_torch, ...}
    initialize_E_via_gigi          -> {E_handle, E_init_t, ...}
    leapfrog_via_gigi              -> shape-compatible w/
                                       buckyball_observables.integrate_with_states
    microcanonical_xcheck_via_gigi -> {P_time_chain, ...}
    canonical_heatbath_via_gigi    -> shape-compatible w/
                                       buckyball_heatbath.thermalize
    beta_scan_via_gigi             -> [{beta, P_measured, P_sem,
                                         P_exact, P_history}, ...]
    beta_envelope_via_gigi         -> list of dicts matching lines
                                       390-402 of run_validation_report.py

Seed-offset additions to the run_validation_report.py docstring table
(see file header lines 30-41):

    [10000,  20000) microcanonical cross-check E init (seed + 11000)
                    -- NEW: GIGI needs an explicit E SEED for the
                    cross-check (the Python kernel re-uses E_init_t).
    [60000,  70000) beta-envelope E init  (seed + 60000 + int(b*1000))
                    -- documented in the existing comment at line 340.

Engine-side bit-identity contract: GIGI's xorshift64* is keyed by
integer SEEDs; the partitioning above is namespace-safe (no offset
collision with the Python kernel's PCG64 streams), but the bit
sequence is engine-specific. See the spec's open questions on F1.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from inertia_damping import buckyball_graph
from inertia_damping import buckyball_yangmills_exact as ym_exact
from inertia_damping import validation_report

from inertia_damping.gigi_client import (
    EFieldInit,
    GaugeFieldInit,
    Group,
    LatticeSpec,
    LiveGIGIClient,
    ObservableId,
    ProjectGaussConfig,
)
from inertia_damping.gigi_orchestrator.translate_chains import (
    gauge_buffer_to_torch,
    normalize_measurement_history,
)

# Canonical names used by --use-gigi across the orchestrator phases.
LATTICE_NAME = "halcyon_canonical_buckyball"

# Phase 2 names
U_THERM_NAME = "U_thermalized"
# Phase 3 names
E_CANONICAL_NAME = "E_canonical"
# Phase 6 names (microcanonical cross-check)
U_XCHECK_NAME = "U_xcheck"
E_XCHECK_NAME = "E_xcheck"
# Phase 7 names (canonical heatbath)
U_CANONICAL_NAME = "U_canonical"

# Seed offsets NEW for the GIGI path (extend the doc table in
# run_validation_report.py lines 30-41 when these change).
SEED_OFFSET_E_INIT = 1            # phase 3 (existing)
SEED_OFFSET_CANONICAL = 100       # phase 7 (existing)
SEED_OFFSET_XCHECK_E = 11000      # phase 6 (NEW for GIGI; spec § open Q)
SEED_OFFSET_ENV_HEAT_BASE = 50000 # phase 8b (existing)
SEED_OFFSET_ENV_E_BASE = 60000    # phase 8b (existing)


def _b_int(beta: float) -> int:
    """``int(round(beta * 1000))`` — matches the orchestrator's seed
    namespace partitioning at lines 313 and 339."""
    return int(round(float(beta) * 1000.0))


# =====================================================================
# Phase 1: declare lattice (idempotent)
# =====================================================================
def declare_lattice_phase(client: LiveGIGIClient, args: argparse.Namespace) -> str:
    """Declare the canonical buckyball lattice on the GIGI substrate.

    Idempotent per the spec: re-declaring with the same name is a
    no-op on the engine side. Returns the lattice name.
    """
    graph = buckyball_graph.build_truncated_icosahedron()
    spec = LatticeSpec(
        name=LATTICE_NAME,
        vertices=graph.n_vertices,
        edges=tuple((int(u), int(v)) for (u, v) in graph.edges),
        faces=tuple(tuple(int(v) for v in face) for face in graph.face_vertices),
        topology="S2",
        euler_characteristic=graph.n_vertices - graph.n_edges + graph.n_faces,
    )
    name = client.declare_lattice(spec)
    return name


# =====================================================================
# Phase 2: thermalize via GIBBS_SAMPLE
# =====================================================================
def thermalize_via_gigi(
    client: LiveGIGIClient,
    args: argparse.Namespace,
    lattice_name: str,
) -> Dict[str, Any]:
    """Symplectic-arm cold-thermalize. Mirrors run_validation_report
    phase 2 (lines 174-180): repeated heatbath sweeps from IDENTITY.

    Returns:
        U_handle           : str (the GIGI field name)
        U_buf_torch        : torch.Tensor (n_edges, 4) float64
        measurement_history: {ObservableId.value: np.ndarray}
        diagnostics        : {seed, beta, n_sweeps_completed}
    """
    client.declare_gauge_field(
        name=U_THERM_NAME,
        lattice_name=lattice_name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    result = client.gibbs_sample(
        field_name=U_THERM_NAME,
        beta=args.beta,
        n_sweeps=args.therm_sweeps,
        measure_every=1,
        measure=[ObservableId.MEAN_PLAQUETTE, ObservableId.Q_SURROGATE],
        seed=args.seed,
    )
    history = normalize_measurement_history(result.measurement_history)
    buf = client.introspect_gauge_field(U_THERM_NAME)
    return {
        "U_handle": U_THERM_NAME,
        "U_buf_torch": gauge_buffer_to_torch(buf),
        "measurement_history": history,
        "diagnostics": dict(result.diagnostics),
    }


# =====================================================================
# Phase 3: initialize E_canonical via E_FIELD declare
# =====================================================================
def initialize_E_via_gigi(
    client: LiveGIGIClient,
    args: argparse.Namespace,
    U_handle: str,
) -> Dict[str, Any]:
    """Mirror phase 3: declare a Maxwell-Boltzmann E on top of U.

    Open question (spec § open Q): GIGI's HTTP surface has no
    ``introspect_e_field`` GET-equivalent. We declare the E field on
    the engine and return ``E_init_t = None``; phase 5's dump_trajectory
    falls back to a zero kinetic per-frame array when ``E_init_t`` is
    None (the orchestrator detects this and writes a documented
    degenerate sidecar field, schema 0.1 still validates). The E lives
    on the engine and is consumed by phase 4's SYMPLECTIC_FLOW directly.
    """
    client.declare_e_field(
        name=E_CANONICAL_NAME,
        gauge_field=U_handle,
        init=EFieldInit.MAXWELL_BOLTZMANN,
        beta=args.beta,
        seed=args.seed + SEED_OFFSET_E_INIT,
    )
    return {
        "E_handle": E_CANONICAL_NAME,
        "E_init_t": None,  # see open question in module docstring
        "diagnostics": {
            "seed": args.seed + SEED_OFFSET_E_INIT,
            "beta": float(args.beta),
        },
    }


# =====================================================================
# Phase 4: leapfrog via SYMPLECTIC_FLOW
# =====================================================================
def leapfrog_via_gigi(
    client: LiveGIGIClient,
    args: argparse.Namespace,
    U_handle: str,
    E_handle: str,
) -> Dict[str, Any]:
    """Mirror phase 4: SYMPLECTIC_FLOW with PROJECT_GAUSS defaults.

    Returns a dict shape-compatible with
    ``buckyball_observables.integrate_with_states``:

        U_traj      : list[torch.Tensor]  -- see OPEN QUESTION
        E_traj      : list[torch.Tensor]  -- see OPEN QUESTION
        steps       : list[int]
        times       : list[float]
        H_history   : list[float]
        K_history   : list[float] (zero-filled; not exposed on the wire)
        V_history   : list[float] (zero-filled; not exposed on the wire)
        G_history   : list[float]
        U_final     : torch.Tensor (n_edges, 4)
        E_final     : None (no E introspection endpoint; see open Q)
        dt          : float
        beta        : float
        n_steps     : int
        measure_every: int

    OPEN QUESTION (spec § risks): the live HTTP surface returns the
    FINAL U buffer only, not per-snapshot U / E. We return
    ``U_traj = [U_final] * len(steps)`` as a degenerate fill. Phase 5
    (dump_trajectory) and phase 8c (sector_classifier) consume U_traj
    per-frame; the orchestrator detects ``len(set(id(U) for U in
    U_traj)) == 1`` and either (a) skips per-frame body or (b) writes
    a documented degenerate trajectory. For F0 smoke this is fine; for
    the convergence study, a per-snapshot extraction endpoint is the
    real fix.
    """
    # Boost timeout for large flows. Default LiveGIGIClient timeout is
    # 30 s (live.py line 58); a 16k-step flow can exceed that.
    if args.steps >= 4000:
        client.timeout_s = max(client.timeout_s, 600.0)

    flow = client.symplectic_flow(
        field=U_handle,
        e_field=E_handle,
        beta=args.beta,
        dt=args.dt,
        n_steps=args.steps,
        project_gauss=ProjectGaussConfig.default(),
        measure_every=args.measure_every,
        measure=[
            ObservableId.MEAN_PLAQUETTE,
            ObservableId.Q_SURROGATE,
            ObservableId.H_TOTAL,
            ObservableId.GAUSS_RESIDUAL_MAX,
        ],
        seed=args.seed,
    )
    history = normalize_measurement_history(flow.measurement_history)

    P_chain = history.get("MEAN(PLAQUETTE)", np.asarray([], dtype=np.float64))
    Q_chain = history.get("Q_SURROGATE", np.asarray([], dtype=np.float64))
    H_chain = history.get("H_TOTAL", np.asarray([], dtype=np.float64))
    G_chain = history.get("GAUSS_RESIDUAL_MAX", np.asarray([], dtype=np.float64))

    # Records: measure_every == k -> samples at k, 2k, ..., n_steps
    n_records = int(args.steps // max(int(args.measure_every), 1))
    # If the engine emitted a different cardinality (e.g. include_initial),
    # honour what came back over our derivation.
    n_records = len(P_chain) if len(P_chain) > 0 else n_records
    steps_list = [int((i + 1) * args.measure_every) for i in range(n_records)]
    times_list = [float(s) * float(args.dt) for s in steps_list]

    # Final U as a torch tensor for the sidecar.
    final_buf = client.introspect_gauge_field(U_handle)
    U_final = gauge_buffer_to_torch(final_buf)

    # Degenerate per-snapshot fill: same final U everywhere.
    U_traj: List[Any] = [U_final.detach().clone() for _ in range(n_records)]
    E_traj: List[Any] = [None for _ in range(n_records)]  # see open Q

    return {
        "U_traj": U_traj,
        "E_traj": E_traj,
        "times": times_list,
        "steps": steps_list,
        "H_history": [float(x) for x in H_chain.tolist()],
        "K_history": [0.0] * n_records,
        "V_history": [0.0] * n_records,
        "G_history": [float(x) for x in G_chain.tolist()],
        "U_final": U_final,
        "E_final": None,
        "dt": float(args.dt),
        "beta": float(args.beta),
        "n_steps": int(args.steps),
        "measure_every": int(args.measure_every),
        "P_meas_chain": P_chain,
        "Q_chain": Q_chain,
        "gigi_diagnostics": _diag_to_dict(flow.diagnostics),
    }


def _diag_to_dict(diag: Any) -> Dict[str, Any]:
    """Best-effort dataclass -> dict conversion for additive sidecar
    receipts. Tolerates ``SymplecticFlowDiagnostics`` and plain dicts."""
    if diag is None:
        return {}
    if isinstance(diag, dict):
        return dict(diag)
    out: Dict[str, Any] = {}
    for slot in (
        "run_id",
        "beta",
        "dt",
        "n_steps_completed",
        "cg_iterations_per_step_p99",
        "max_energy_drift_rel",
        "gauss_residual_max",
        "seed",
    ):
        if hasattr(diag, slot):
            v = getattr(diag, slot)
            try:
                out[slot] = float(v) if isinstance(v, (int, float)) else v
            except Exception:
                out[slot] = v
    return out


# =====================================================================
# Phase 6: microcanonical cross-check via SYMPLECTIC_FLOW (PROJECT_GAUSS FALSE)
# =====================================================================
def microcanonical_xcheck_via_gigi(
    client: LiveGIGIClient,
    args: argparse.Namespace,
    lattice_name: str,
) -> Dict[str, Any]:
    """Mirror phase 6: long microcanonical run from a fresh IDENTITY
    start, no Gauss projection (matches the Python kernel's
    ``leapfrog_step`` loop at line 277, which does NOT project).

    Returns:
        P_time_chain : list[float]  -- measured every measure_every steps
        gauss_residual_max : float  -- max across the run
        H_drift_max_rel : float
        diagnostics : dict
    """
    # Fresh field declarations for the cross-check (independent stream).
    client.declare_gauge_field(
        name=U_XCHECK_NAME,
        lattice_name=lattice_name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    client.declare_e_field(
        name=E_XCHECK_NAME,
        gauge_field=U_XCHECK_NAME,
        init=EFieldInit.MAXWELL_BOLTZMANN,
        beta=args.beta,
        seed=args.seed + SEED_OFFSET_XCHECK_E,
    )
    if args.cross_check_steps >= 4000:
        client.timeout_s = max(client.timeout_s, 600.0)
    flow = client.symplectic_flow(
        field=U_XCHECK_NAME,
        e_field=E_XCHECK_NAME,
        beta=args.beta,
        dt=args.dt,
        n_steps=args.cross_check_steps,
        # Spec § open Q: Python kernel's leapfrog_step does NOT project.
        # Mirror that by passing PROJECT_GAUSS FALSE (project_gauss=None).
        project_gauss=None,
        measure_every=args.measure_every,
        measure=[
            ObservableId.MEAN_PLAQUETTE,
            ObservableId.H_TOTAL,
            ObservableId.GAUSS_RESIDUAL_MAX,
        ],
        seed=args.seed + SEED_OFFSET_XCHECK_E,
    )
    history = normalize_measurement_history(flow.measurement_history)
    P_chain = history.get("MEAN(PLAQUETTE)", np.asarray([], dtype=np.float64))
    H_chain = history.get("H_TOTAL", np.asarray([], dtype=np.float64))
    G_chain = history.get("GAUSS_RESIDUAL_MAX", np.asarray([], dtype=np.float64))

    if H_chain.size >= 2 and abs(float(H_chain[0])) > 0:
        H0 = float(H_chain[0])
        H_drift_max_rel = float(np.max(np.abs(H_chain - H0)) / abs(H0))
    else:
        H_drift_max_rel = 0.0

    return {
        "P_time_chain": [float(x) for x in P_chain.tolist()],
        "gauss_residual_max": float(G_chain.max()) if G_chain.size else 0.0,
        "H_drift_max_rel": H_drift_max_rel,
        "diagnostics": _diag_to_dict(flow.diagnostics),
    }


# =====================================================================
# Phase 7: canonical heatbath via GIBBS_SAMPLE
# =====================================================================
def canonical_heatbath_via_gigi(
    client: LiveGIGIClient,
    args: argparse.Namespace,
    lattice_name: str,
) -> Dict[str, Any]:
    """Mirror phase 7: long canonical heatbath at args.beta.

    The Python kernel ``buckyball_heatbath.thermalize`` returns:
        P_mean / P_std / P_sem / n_samples / U_final / P_history

    OPEN QUESTION (spec § open Q): the kernel splits sweeps into
    ``n_sweeps`` (therm, no measurement) + ``n_measure`` (post-therm,
    every-other sweep). GIGI's GIBBS_SAMPLE has no separate
    thermalization phase; we run N_SWEEPS = therm + measure*2 with
    MEASURE_EVERY 2, then slice off ``therm // 2`` measurements
    client-side to match the Python shape. This matches the spec's
    "2048-sample chain" language; the precise slicing convention is
    documented here.
    """
    client.declare_gauge_field(
        name=U_CANONICAL_NAME,
        lattice_name=lattice_name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    therm = int(args.canonical_therm)
    measure = int(args.canonical_sweeps)
    n_sweeps_total = therm + 2 * measure
    if n_sweeps_total >= 4000:
        client.timeout_s = max(client.timeout_s, 600.0)
    result = client.gibbs_sample(
        field_name=U_CANONICAL_NAME,
        beta=args.beta,
        n_sweeps=n_sweeps_total,
        measure_every=2,
        measure=[ObservableId.MEAN_PLAQUETTE, ObservableId.Q_SURROGATE],
        seed=args.seed + SEED_OFFSET_CANONICAL,
    )
    history = normalize_measurement_history(result.measurement_history)
    full_P = history.get("MEAN(PLAQUETTE)", np.asarray([], dtype=np.float64))
    full_Q = history.get("Q_SURROGATE", np.asarray([], dtype=np.float64))
    # Slice off the thermalization portion: therm sweeps / measure_every 2
    therm_drop = therm // 2
    P_history = full_P[therm_drop:]
    Q_history = full_Q[therm_drop:]
    Ps = np.asarray(P_history, dtype=np.float64)
    if Ps.size == 0:
        raise RuntimeError(
            "canonical_heatbath_via_gigi: post-thermalization chain is "
            "empty; check --canonical-therm / --canonical-sweeps."
        )
    P_mean = float(Ps.mean())
    P_std = float(Ps.std(ddof=1)) if Ps.size > 1 else 0.0
    P_sem = P_std / math.sqrt(Ps.size) if Ps.size > 1 else 0.0
    buf = client.introspect_gauge_field(U_CANONICAL_NAME)
    return {
        "P_mean": P_mean,
        "P_std": P_std,
        "P_sem": P_sem,
        "n_samples": int(Ps.size),
        "U_final": gauge_buffer_to_torch(buf),
        "P_history": list(Ps),
        "Q_history": [float(x) for x in Q_history.tolist()],
        "gigi_diagnostics": dict(result.diagnostics),
    }


# =====================================================================
# Phase 8: beta-scan via GIBBS_SAMPLE loop
# =====================================================================
def beta_scan_via_gigi(
    client: LiveGIGIClient,
    args: argparse.Namespace,
    lattice_name: str,
) -> List[Dict[str, Any]]:
    """Mirror phase 8a: 6-beta scan, each via GIBBS_SAMPLE.

    Returns a list of dicts with EXACTLY the shape lines 316-322 of
    run_validation_report.py produce:
        {beta, P_measured, P_sem, P_exact, P_history}
    """
    betas = [float(x.strip()) for x in args.beta_scan.split(",") if x.strip()]
    out: List[Dict[str, Any]] = []
    for b in betas:
        field = f"U_scan_b{_b_int(b)}"
        client.declare_gauge_field(
            name=field,
            lattice_name=lattice_name,
            group=Group.SU2,
            init=GaugeFieldInit.IDENTITY,
        )
        # Match the Python kernel: n_sweeps=200 (therm), n_measure=600,
        # n_measure_every=2 -> N_SWEEPS = 200 + 1200 with MEASURE_EVERY 2,
        # then drop the first 100 measurements.
        scan_therm = 200
        scan_measure = 600
        n_sweeps_total = scan_therm + 2 * scan_measure
        result = client.gibbs_sample(
            field_name=field,
            beta=b,
            n_sweeps=n_sweeps_total,
            measure_every=2,
            measure=[ObservableId.MEAN_PLAQUETTE],
            seed=args.seed + _b_int(b),
        )
        history = normalize_measurement_history(result.measurement_history)
        full_P = history.get("MEAN(PLAQUETTE)", np.asarray([], dtype=np.float64))
        P_chain = full_P[scan_therm // 2:]
        Ps = np.asarray(P_chain, dtype=np.float64)
        if Ps.size == 0:
            raise RuntimeError(
                f"beta_scan_via_gigi: empty post-therm chain at beta={b}"
            )
        P_mean = float(Ps.mean())
        P_std = float(Ps.std(ddof=1)) if Ps.size > 1 else 0.0
        P_sem = P_std / math.sqrt(Ps.size) if Ps.size > 1 else 0.0
        P_exact = float(ym_exact.exact_mean_plaquette_su2_buckyball(b))
        out.append({
            "beta": float(b),
            "P_measured": P_mean,
            "P_sem": P_sem,
            "P_exact": P_exact,
            "P_history": [float(x) for x in Ps.tolist()],
        })
    return out


# =====================================================================
# Phase 8b: beta-envelope sweep
# =====================================================================
def beta_envelope_via_gigi(
    client: LiveGIGIClient,
    args: argparse.Namespace,
    lattice_name: str,
) -> List[Dict[str, Any]]:
    """Mirror phase 8b: per-beta short heatbath + short leapfrog with
    Gauss projection, aggregated to scalar receipts + a capped
    P_history. The return shape matches lines 390-402 of
    run_validation_report.py exactly so the orchestrator's sidecar
    assertion at line 408 ENVELOPE_MAX_SIDECAR_BYTES still holds.
    """
    # Local constants must match run_validation_report.py
    ENVELOPE_THERM_SWEEPS = 50
    ENVELOPE_MEASURE_SWEEPS = 1000
    ENVELOPE_LEAPFROG_STEPS = 400
    ENVELOPE_MEASURE_EVERY = 20

    env_betas = [float(x.strip()) for x in args.beta_envelope.split(",") if x.strip()]
    out: List[Dict[str, Any]] = []
    for b in env_betas:
        seed_b_heat = args.seed + SEED_OFFSET_ENV_HEAT_BASE + _b_int(b)
        seed_b_E = args.seed + SEED_OFFSET_ENV_E_BASE + _b_int(b)
        u_name = f"U_env_b{_b_int(b)}"
        e_name = f"E_env_b{_b_int(b)}"

        # (c) heatbath
        client.declare_gauge_field(
            name=u_name,
            lattice_name=lattice_name,
            group=Group.SU2,
            init=GaugeFieldInit.IDENTITY,
        )
        n_sweeps_total = ENVELOPE_THERM_SWEEPS + 2 * ENVELOPE_MEASURE_SWEEPS
        heat = client.gibbs_sample(
            field_name=u_name,
            beta=b,
            n_sweeps=n_sweeps_total,
            measure_every=2,
            measure=[ObservableId.MEAN_PLAQUETTE],
            seed=seed_b_heat,
        )
        h_hist = normalize_measurement_history(heat.measurement_history)
        full_P = h_hist.get("MEAN(PLAQUETTE)", np.asarray([], dtype=np.float64))
        P_chain = full_P[ENVELOPE_THERM_SWEEPS // 2:]
        Ps = np.asarray(P_chain, dtype=np.float64)
        if Ps.size == 0:
            raise RuntimeError(
                f"beta_envelope_via_gigi: empty heatbath chain at beta={b}"
            )
        P_meas_b = float(Ps.mean())
        P_std_b = float(Ps.std(ddof=1)) if Ps.size > 1 else 0.0
        P_sem_b = P_std_b / math.sqrt(Ps.size) if Ps.size > 1 else 0.0
        P_exact_b = float(ym_exact.exact_mean_plaquette_su2_buckyball(b))
        migdal_gap_b = abs(P_meas_b - P_exact_b)

        # (d-e) leapfrog
        client.declare_e_field(
            name=e_name,
            gauge_field=u_name,
            init=EFieldInit.MAXWELL_BOLTZMANN,
            beta=b,
            seed=seed_b_E,
        )
        flow = client.symplectic_flow(
            field=u_name,
            e_field=e_name,
            beta=b,
            dt=args.dt,
            n_steps=ENVELOPE_LEAPFROG_STEPS,
            project_gauss=ProjectGaussConfig.default(),
            measure_every=ENVELOPE_MEASURE_EVERY,
            measure=[
                ObservableId.MEAN_PLAQUETTE,
                ObservableId.Q_SURROGATE,
                ObservableId.H_TOTAL,
                ObservableId.GAUSS_RESIDUAL_MAX,
            ],
            seed=seed_b_E,
        )
        f_hist = normalize_measurement_history(flow.measurement_history)
        H_chain = f_hist.get("H_TOTAL", np.asarray([], dtype=np.float64))
        G_chain = f_hist.get("GAUSS_RESIDUAL_MAX", np.asarray([], dtype=np.float64))
        P_micro_chain = f_hist.get("MEAN(PLAQUETTE)", np.asarray([], dtype=np.float64))
        Q_micro_chain = f_hist.get("Q_SURROGATE", np.asarray([], dtype=np.float64))

        # (f) energy drift max relative
        if H_chain.size >= 2 and abs(float(H_chain[0])) > 0:
            H0_b = float(H_chain[0])
            energy_drift_max = float(np.max(np.abs(H_chain - H0_b)) / abs(H0_b))
        else:
            energy_drift_max = 0.0
        gauss_max_across_window = float(G_chain.max()) if G_chain.size else 0.0

        # (g) crosscheck gap
        P_micro = float(P_micro_chain.mean()) if P_micro_chain.size else P_meas_b
        crosscheck_gap = abs(P_meas_b - P_micro)

        # (h) blocking on heatbath -> regime string
        b_anal = validation_report._flyvbjerg_petersen_blocking(Ps)

        # (i) Q std across the leapfrog window
        sector_proxy_q_std = (
            float(np.std(Q_micro_chain, ddof=1)) if Q_micro_chain.size > 1 else 0.0
        )

        # Cap P_history to 200 floats so the sidecar stays under budget.
        P_history_capped = [float(x) for x in Ps.tolist()[-200:]]
        out.append({
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
    return out


# =====================================================================
# Snapshot helper: post-canonical SNAPSHOT GAUGE_FIELD U PERSIST
# =====================================================================
def snapshot_canonical_via_gigi(
    client: LiveGIGIClient,
    field_name: str = U_CANONICAL_NAME,
) -> Dict[str, Any]:
    """Emit ``SNAPSHOT GAUGE_FIELD <field> PERSIST;`` and return the
    Rows envelope's sha256 + wal_offset + n_edges + repr_dim. This
    matches the pattern in
    ``scripts/verify_canonical_receipt.py::verify`` (Statement 5).

    The caller (run_validation_report.py phase 7+) writes these fields
    into the sidecar so F1 can assert the buffer SHA reproduces
    between consecutive runs.
    """
    body = client._gql_query(  # noqa: SLF001 — module-level usage, contract per spec
        f"SNAPSHOT GAUGE_FIELD {field_name} PERSIST;"
    )
    rows = body.get("rows", body) if isinstance(body, dict) else body
    if isinstance(rows, dict):
        rows = rows.get("rows", [rows])
    if not rows:
        raise RuntimeError(f"SNAPSHOT returned no rows: {body!r}")
    row = rows[0]
    return {
        "snapshot_sha256": row.get("sha256"),
        "snapshot_wal_offset": row.get("wal_offset"),
        "snapshot_n_edges": row.get("n_edges"),
        "snapshot_repr_dim": row.get("repr_dim"),
    }


__all__ = [
    "LATTICE_NAME",
    "U_THERM_NAME",
    "E_CANONICAL_NAME",
    "U_CANONICAL_NAME",
    "declare_lattice_phase",
    "thermalize_via_gigi",
    "initialize_E_via_gigi",
    "leapfrog_via_gigi",
    "microcanonical_xcheck_via_gigi",
    "canonical_heatbath_via_gigi",
    "beta_scan_via_gigi",
    "beta_envelope_via_gigi",
    "snapshot_canonical_via_gigi",
]
