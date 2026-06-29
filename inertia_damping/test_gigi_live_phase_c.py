"""Phase C live tests — fire SYMPLECTIC_FLOW on real gigi-stream.

Part IV is live in production as of 2026-06-18 (deploy
``a6d8efa`` after the LBrace/RBrace hotfix for prod features).

THIS is the architectural completion: every phase of
``inertia_damping/run_validation_report.py`` now has a GQL home.
The leapfrog runs on GIGI's Rust engine with PROJECT_GAUSS struct
clause exposed (per Q3 — tikhonov / cg_tol / cg_max_iter knobs).

Phase C receipts:
  G_LIVE_C0   Tiny SYMPLECTIC_FLOW smoke test — wire shape +
              measurement chain return.
  G_LIVE_C1   PROJECT_GAUSS TRUE delivers max |dH/H_0| < 1e-3 on
              50 steps at dt=0.02, β=2.5.
  G_LIVE_C2   PROJECT_GAUSS struct override accepted by the engine
              (Q3 receipt).
  G_LIVE_C3   Same-seed bit-identity over the H_TOTAL chain.

Skip cleanly when gigi-stream isn't reachable.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from inertia_damping import buckyball_graph
from inertia_damping.gigi_client import (
    EFieldInit,
    GaugeFieldInit,
    Group,
    LatticeSpec,
    LiveGIGIClient,
    ObservableId,
    ProjectGaussConfig,
)


@pytest.fixture(scope="module")
def live_client() -> LiveGIGIClient:
    url = os.environ.get("GIGI_URL", "http://localhost:3142")
    try:
        return LiveGIGIClient(base_url=url, ping=True)
    except (ConnectionError, RuntimeError) as ex:
        pytest.skip(f"gigi-stream not reachable: {ex}")


@pytest.fixture(scope="module")
def buckyball_spec() -> LatticeSpec:
    g = buckyball_graph.build_truncated_icosahedron()
    return LatticeSpec(
        name="halcyon_phase_c_buckyball",
        vertices=g.n_vertices,
        edges=tuple((int(u), int(v)) for (u, v) in g.edges),
        faces=tuple(tuple(int(v) for v in face) for face in g.face_vertices),
        topology="S2",
        euler_characteristic=2,
    )


def _thermalize(client: LiveGIGIClient, spec: LatticeSpec, u_name: str) -> None:
    client.declare_lattice(spec)
    client.declare_gauge_field(
        name=u_name, lattice_name=spec.name,
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    client.gibbs_sample(
        u_name, beta=2.5, n_sweeps=50, seed=20260616, measure=[],
    )


# ---------------------------------------------------------------------
# G_LIVE_C0 — Smoke
# ---------------------------------------------------------------------
def test_G_LIVE_C0_symplectic_smoke(
    live_client: LiveGIGIClient, buckyball_spec: LatticeSpec,
):
    _thermalize(live_client, buckyball_spec, "U_c0")
    live_client.declare_e_field(
        "E_c0", "U_c0", EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617,
    )
    res = live_client.symplectic_flow(
        field="U_c0", e_field="E_c0",
        beta=2.5, dt=0.02, n_steps=20,
        project_gauss=ProjectGaussConfig.default(),
        measure_every=5,
        measure=[ObservableId.H_TOTAL, ObservableId.MEAN_PLAQUETTE,
                 ObservableId.GAUSS_RESIDUAL_MAX],
        seed=20260617,
    )
    H = res.measurement_history[ObservableId.H_TOTAL.value]
    P = res.measurement_history[ObservableId.MEAN_PLAQUETTE.value]
    G = res.measurement_history[ObservableId.GAUSS_RESIDUAL_MAX.value]
    assert H.shape == (4,)
    assert P.shape == (4,)
    assert G.shape == (4,)
    assert np.isfinite(H).all()
    assert np.isfinite(P).all()
    assert np.isfinite(G).all()
    assert res.diagnostics.n_steps_completed == 20
    assert res.diagnostics.beta == pytest.approx(2.5, abs=1e-12)
    assert res.diagnostics.dt == pytest.approx(0.02, abs=1e-12)


# ---------------------------------------------------------------------
# G_LIVE_C1 — Energy conservation on the live engine
# ---------------------------------------------------------------------
def test_G_LIVE_C1_energy_drift_below_tolerance(
    live_client: LiveGIGIClient, buckyball_spec: LatticeSpec,
):
    """Section 2 energy-drift gate fired against the LIVE engine.
    max |dH/H_0| < 1e-3 over 50 leapfrog steps at dt=0.02, β=2.5."""
    _thermalize(live_client, buckyball_spec, "U_c1")
    live_client.declare_e_field(
        "E_c1", "U_c1", EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617,
    )
    res = live_client.symplectic_flow(
        field="U_c1", e_field="E_c1",
        beta=2.5, dt=0.02, n_steps=50,
        project_gauss=ProjectGaussConfig.default(),
        measure_every=10, measure=[], seed=20260617,
    )
    drift = res.diagnostics.max_energy_drift_rel
    assert drift < 1e-3, (
        f"LIVE engine max |dH/H_0| = {drift:.3e}; symplecticness contract "
        f"violated on production. Cross-reference with the III.8a release-"
        f"profile receipts; if the live engine drifts > 1e-3, either "
        f"SYMPLECTIC_FLOW is not symplectic or PROJECT_GAUSS is injecting "
        f"energy via the constraint correction."
    )


def test_G_LIVE_C1_gauss_residual_machine_epsilon(
    live_client: LiveGIGIClient, buckyball_spec: LatticeSpec,
):
    """Section 2 Gauss-residual gate fired against the LIVE engine.
    With PROJECT_GAUSS enabled, residual stays at machine epsilon."""
    _thermalize(live_client, buckyball_spec, "U_c1g")
    live_client.declare_e_field(
        "E_c1g", "U_c1g", EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617,
    )
    res = live_client.symplectic_flow(
        field="U_c1g", e_field="E_c1g",
        beta=2.5, dt=0.02, n_steps=50,
        project_gauss=ProjectGaussConfig.default(),
        measure_every=50, measure=[], seed=20260617,
    )
    gres = res.diagnostics.gauss_residual_max
    assert gres < 1e-9, (
        f"LIVE engine Gauss residual = {gres:.3e}; covariant Gauss law "
        f"violated. CG projector failed to converge within cg_max_iter=200 "
        f"or PROJECT_GAUSS struct override wasn't honored."
    )


# ---------------------------------------------------------------------
# G_LIVE_C2 — PROJECT_GAUSS struct override accepted (Q3 receipt)
# ---------------------------------------------------------------------
def test_G_LIVE_C2_project_gauss_struct_override(
    live_client: LiveGIGIClient, buckyball_spec: LatticeSpec,
):
    """Q3 contract: PROJECT_GAUSS { tikhonov, cg_tol, cg_max_iter }
    struct sugar is accepted by the engine. Pass non-default values
    to confirm the override path is wired."""
    _thermalize(live_client, buckyball_spec, "U_c2")
    live_client.declare_e_field(
        "E_c2", "U_c2", EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617,
    )
    cfg = ProjectGaussConfig(tikhonov=1e-12, cg_tol=1e-8, cg_max_iter=500)
    res = live_client.symplectic_flow(
        field="U_c2", e_field="E_c2",
        beta=2.5, dt=0.02, n_steps=20,
        project_gauss=cfg,
        measure_every=20, measure=[], seed=20260617,
    )
    # Even with looser cg_tol the energy drift should stay below the
    # production tolerance; the receipt is that the override was
    # accepted (no parse error, leapfrog completes).
    assert res.diagnostics.n_steps_completed == 20
    assert res.diagnostics.max_energy_drift_rel < 1e-2, (
        f"with loosened PROJECT_GAUSS (tikhonov=1e-12, cg_tol=1e-8) the "
        f"energy drift = {res.diagnostics.max_energy_drift_rel:.3e}; "
        f"either the struct override wasn't applied or the tolerance "
        f"relaxation broke symplecticness more than expected."
    )


# ---------------------------------------------------------------------
# G_LIVE_C3 — A2 same-seed bit-identity on the H_TOTAL chain
# ---------------------------------------------------------------------
def test_G_LIVE_C3_same_seed_bit_identity(
    live_client: LiveGIGIClient, buckyball_spec: LatticeSpec,
):
    """A2: same-seed SYMPLECTIC_FLOW in the same process reproduces
    bit-identical H_TOTAL chains. The contract Halcyon's harness
    cross-pins from MockGIGIClient into the real engine."""
    def _run(suffix: str) -> np.ndarray:
        u_name = f"U_c3_{suffix}"
        e_name = f"E_c3_{suffix}"
        _thermalize(live_client, buckyball_spec, u_name)
        live_client.declare_e_field(
            e_name, u_name, EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617,
        )
        res = live_client.symplectic_flow(
            field=u_name, e_field=e_name,
            beta=2.5, dt=0.02, n_steps=30,
            project_gauss=ProjectGaussConfig.default(),
            measure_every=1, measure=[ObservableId.H_TOTAL], seed=20260617,
        )
        return res.measurement_history[ObservableId.H_TOTAL.value]

    H1 = _run("a")
    H2 = _run("b")
    np.testing.assert_array_equal(H1, H2, err_msg=(
        "Same-seed SYMPLECTIC_FLOW on the LIVE engine produced different "
        "H_TOTAL chains. A2 same-process bit-identity contract broken on "
        "the engine side."
    ))
