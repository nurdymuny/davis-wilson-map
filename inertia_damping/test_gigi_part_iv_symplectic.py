"""Part IV gate test — E_FIELD + SYMPLECTIC_FLOW + PROJECT_GAUSS.

Locks the Part IV contracts that GIGI shipped on 2026-06-18:

  G4.A   E_FIELD declaration: INIT ZERO, INIT MAXWELL_BOLTZMANN (seed
         + beta), INIT FROM_FIELD round-trip.
  G4.B   ProjectGaussConfig defaults match the GIGI release-profile
         numbers (tikhonov=1e-14, cg_tol=1e-10, cg_max_iter=200) and
         struct-literal overrides take.
  G4.C   SYMPLECTIC_FLOW energy conservation: max |dH/H_0| < 1e-3
         on a 50-step leapfrog at dt=0.02, β=2.5 (the production
         Section 2 energy-drift gate).
  G4.D   SYMPLECTIC_FLOW Gauss residual: covariant Gauss residual
         stays at machine epsilon (≤ 1e-9) when PROJECT_GAUSS is
         enabled. This is the production Section 2 Gauss gate.
  G4.E   SYMPLECTIC_FLOW measurement chains carry H_TOTAL,
         MEAN_PLAQUETTE, Q_SURROGATE, GAUSS_RESIDUAL_MAX of the
         right shape and finite values.
  G4.F   SYMPLECTIC_FLOW intra-process bit-identity at same seed.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from inertia_damping import buckyball_graph
from inertia_damping.gigi_client import (
    EFieldInit,
    GaugeFieldInit,
    Group,
    LatticeSpec,
    MockGIGIClient,
    ObservableId,
    ProjectGaussConfig,
)


@pytest.fixture(scope="module")
def buckyball_spec() -> LatticeSpec:
    g = buckyball_graph.build_truncated_icosahedron()
    return LatticeSpec(
        name="buckyball",
        vertices=g.n_vertices,
        edges=tuple((int(u), int(v)) for (u, v) in g.edges),
        faces=tuple(tuple(int(v) for v in face) for face in g.face_vertices),
        topology="S2",
        euler_characteristic=g.n_vertices - g.n_edges + g.n_faces,
    )


def _client_with_thermalized_field(spec: LatticeSpec) -> Tuple[MockGIGIClient, str]:
    """Build a client with a 50-sweep thermalized U field at β=2.5."""
    c = MockGIGIClient()
    c.declare_lattice(spec)
    c.declare_gauge_field("U", "buckyball", Group.SU2, GaugeFieldInit.IDENTITY)
    c.gibbs_sample("U", beta=2.5, n_sweeps=50, seed=20260616, measure=[])
    return c, "U"


# ---------------------------------------------------------------------
# G4.A — E_FIELD declarations
# ---------------------------------------------------------------------
def test_G4_A_e_field_zero(buckyball_spec: LatticeSpec):
    c, u = _client_with_thermalized_field(buckyball_spec)
    handle = c.declare_e_field("E0", u, EFieldInit.ZERO)
    assert handle.init_kind == EFieldInit.ZERO
    buf = c.introspect_e_field("E0")
    assert buf.shape == (90, 4)
    np.testing.assert_allclose(buf, np.zeros((90, 4)), atol=1e-14)


def test_G4_A_e_field_maxwell_boltzmann_requires_seed(buckyball_spec: LatticeSpec):
    c, u = _client_with_thermalized_field(buckyball_spec)
    with pytest.raises(ValueError, match="SEED"):
        c.declare_e_field("E_bad", u, EFieldInit.MAXWELL_BOLTZMANN, beta=2.5)


def test_G4_A_e_field_maxwell_boltzmann_q0_zero(buckyball_spec: LatticeSpec):
    """Maxwell-Boltzmann samples are su(2) Lie elements: q0=0 per row."""
    c, u = _client_with_thermalized_field(buckyball_spec)
    c.declare_e_field("E", u, EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617)
    buf = c.introspect_e_field("E")
    np.testing.assert_allclose(buf[:, 0], np.zeros(90), atol=1e-12, err_msg=(
        "MAXWELL_BOLTZMANN E field has nonzero q0 components — Lie algebra "
        "convention violated."
    ))


def test_G4_A_e_field_maxwell_boltzmann_reproducible(buckyball_spec: LatticeSpec):
    """Same seed -> bit-identical E buffer (A2 contract)."""
    c, u = _client_with_thermalized_field(buckyball_spec)
    c.declare_e_field("E1", u, EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617)
    c.declare_e_field("E2", u, EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617)
    b1 = c.introspect_e_field("E1")
    b2 = c.introspect_e_field("E2")
    np.testing.assert_array_equal(b1, b2)


# ---------------------------------------------------------------------
# G4.B — ProjectGaussConfig defaults + overrides
# ---------------------------------------------------------------------
def test_G4_B_project_gauss_defaults_match_release_profile():
    """Defaults are the GIGI release-profile production-canonical values."""
    cfg = ProjectGaussConfig.default()
    assert cfg.tikhonov == 1e-14
    assert cfg.cg_tol == 1e-10
    assert cfg.cg_max_iter == 200


def test_G4_B_project_gauss_struct_override():
    cfg = ProjectGaussConfig(tikhonov=1e-12, cg_tol=1e-8, cg_max_iter=500)
    assert cfg.tikhonov == 1e-12
    assert cfg.cg_tol == 1e-8
    assert cfg.cg_max_iter == 500


# ---------------------------------------------------------------------
# G4.C — Energy conservation receipt
# ---------------------------------------------------------------------
def test_G4_C_energy_drift_below_production_tolerance(buckyball_spec: LatticeSpec):
    """Max |dH/H_0| < 1e-3 over 50 leapfrog steps at dt=0.02, β=2.5.

    This is the Section 2 energy-drift gate from
    ``HALCYON_TO_GIGI_REPLY § A2`` — the symplecticness receipt for
    SYMPLECTIC_FLOW. Production typically delivers ~5e-5 (20× under
    the gate); the gate is intentionally loose.
    """
    c, u = _client_with_thermalized_field(buckyball_spec)
    c.declare_e_field("E", u, EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617)
    res = c.symplectic_flow(
        u, "E", beta=2.5, dt=0.02, n_steps=50,
        project_gauss=ProjectGaussConfig.default(),
        measure_every=10, measure=[ObservableId.H_TOTAL], seed=20260617,
    )
    drift = res.diagnostics.max_energy_drift_rel
    assert drift < 1e-3, (
        f"max |dH/H_0| = {drift:.3e}; Section 2 energy-drift gate broken. "
        f"Either SYMPLECTIC_FLOW is not symplectic or PROJECT_GAUSS is "
        f"injecting too much energy via the constraint correction."
    )


# ---------------------------------------------------------------------
# G4.D — Gauss residual receipt (PROJECT_GAUSS ON)
# ---------------------------------------------------------------------
def test_G4_D_gauss_residual_at_machine_epsilon_with_projection(buckyball_spec: LatticeSpec):
    """With PROJECT_GAUSS enabled, the covariant Gauss residual stays
    at machine epsilon (1e-9 envelope tol). Section 2 Gauss gate."""
    c, u = _client_with_thermalized_field(buckyball_spec)
    c.declare_e_field("E", u, EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617)
    res = c.symplectic_flow(
        u, "E", beta=2.5, dt=0.02, n_steps=50,
        project_gauss=ProjectGaussConfig.default(),
        measure_every=10, measure=[], seed=20260617,
    )
    gres = res.diagnostics.gauss_residual_max
    assert gres < 1e-9, (
        f"Gauss residual = {gres:.3e}; covariant Gauss law violated. "
        f"Either PROJECT_GAUSS is disabled when it should be on, or the "
        f"CG projector failed to converge within cg_max_iter=200."
    )


# ---------------------------------------------------------------------
# G4.E — Measurement chains shape + finiteness
# ---------------------------------------------------------------------
def test_G4_E_measurement_chains_shape_and_finite(buckyball_spec: LatticeSpec):
    c, u = _client_with_thermalized_field(buckyball_spec)
    c.declare_e_field("E", u, EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617)
    res = c.symplectic_flow(
        u, "E", beta=2.5, dt=0.02, n_steps=40,
        project_gauss=ProjectGaussConfig.default(),
        measure_every=5,
        measure=[ObservableId.H_TOTAL, ObservableId.MEAN_PLAQUETTE,
                 ObservableId.Q_SURROGATE, ObservableId.GAUSS_RESIDUAL_MAX],
        seed=20260617,
    )
    # 40 steps / measure_every=5 -> 8 records
    expected_n = 40 // 5
    for obs in (ObservableId.H_TOTAL, ObservableId.MEAN_PLAQUETTE,
                ObservableId.Q_SURROGATE, ObservableId.GAUSS_RESIDUAL_MAX):
        chain = res.measurement_history[obs.value]
        assert chain.shape == (expected_n,), (
            f"observable {obs} chain shape {chain.shape}, expected ({expected_n},)"
        )
        assert np.isfinite(chain).all(), f"observable {obs} chain has non-finite values"


# ---------------------------------------------------------------------
# G4.F — Same-seed bit-identity (A2 same-process)
# ---------------------------------------------------------------------
def test_G4_F_symplectic_flow_same_seed_bit_identical(buckyball_spec: LatticeSpec):
    """Two SYMPLECTIC_FLOW calls in the same process at the same seed
    must produce bit-identical measurement chains. A2 contract."""
    def _run() -> np.ndarray:
        c, u = _client_with_thermalized_field(buckyball_spec)
        c.declare_e_field("E", u, EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617)
        res = c.symplectic_flow(
            u, "E", beta=2.5, dt=0.02, n_steps=30,
            project_gauss=ProjectGaussConfig.default(),
            measure_every=1, measure=[ObservableId.H_TOTAL], seed=20260617,
        )
        return res.measurement_history[ObservableId.H_TOTAL.value]

    H1 = _run()
    H2 = _run()
    np.testing.assert_array_equal(H1, H2, err_msg=(
        "Same-seed SYMPLECTIC_FLOW produced different H_TOTAL chains. "
        "Either the leapfrog is reading external state, or the PROJECT_GAUSS "
        "CG iteration count diverged in a way that broke bit-identity."
    ))
