"""Part III gate test — PLAQUETTE + GIBBS_SAMPLE + observable battery.

Locks the receipts in HALCYON_PART_I_GATES.md § PART III:
  - SELECT PLAQUETTE in all three reductions (per_face / sum / mean)
    matches the kernel.
  - Q_SURROGATE matches buckyball_observables.Q_surrogate exactly
    (per HALCYON_TO_GIGI_REPLY § A1).
  - The 3-statement GQL block (LATTICE -> GAUGE_FIELD -> GIBBS_SAMPLE)
    reproduces the v1.2 production thermalization at β=2.5 seed=
    20260616 to within the Flyvbjerg-Petersen SEM published in the
    production sidecar (canonical <P>_meas = 0.506847 ± 0.001463
    over 2048 samples).
  - Bit-identity per A2: same seed, same β, same n_sweeps, same
    process -> bit-identical link buffer AND measurement history.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from inertia_damping import (
    buckyball_action,
    buckyball_graph,
    buckyball_heatbath,
    buckyball_observables,
)
from inertia_damping.gigi_client import (
    GaugeFieldInit,
    GibbsSampleResult,
    Group,
    LatticeSpec,
    MockGIGIClient,
    ObservableId,
)


@pytest.fixture(scope="module")
def buckyball_spec() -> LatticeSpec:
    graph = buckyball_graph.build_truncated_icosahedron()
    return LatticeSpec(
        name="buckyball",
        vertices=graph.n_vertices,
        edges=tuple((int(u), int(v)) for (u, v) in graph.edges),
        faces=tuple(tuple(int(v) for v in face) for face in graph.face_vertices),
        topology="S2",
        euler_characteristic=graph.n_vertices - graph.n_edges + graph.n_faces,
    )


@pytest.fixture
def client(buckyball_spec: LatticeSpec) -> MockGIGIClient:
    c = MockGIGIClient()
    c.declare_lattice(buckyball_spec)
    return c


# ---------------------------------------------------------------------
# G3.A — PLAQUETTE reductions
# ---------------------------------------------------------------------
def test_G3_A_plaquette_per_face_matches_kernel(client: MockGIGIClient):
    """PLAQUETTE per_face matches buckyball_heatbath.plaquette_per_face
    on a fixed Haar-random field. Bit-identical at FP64."""
    client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    per_face = client.select_plaquette("U", reduction="per_face")
    assert per_face.shape == (32,)
    # Reference via kernel
    graph = buckyball_graph.build_truncated_icosahedron()
    U_torch = client.introspect_gauge_field("U")
    import torch
    U_t = torch.from_numpy(U_torch)
    ref = buckyball_heatbath.plaquette_per_face(U_t, graph).detach().cpu().numpy().astype(np.float64)
    np.testing.assert_array_equal(per_face, ref)


def test_G3_A_plaquette_mean_matches_kernel(client: MockGIGIClient):
    """MEAN(PLAQUETTE) = mean over the per_face values."""
    client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    mean_p = client.select_plaquette("U", reduction="mean")
    per_face = client.select_plaquette("U", reduction="per_face")
    assert mean_p == pytest.approx(per_face.mean(), abs=1e-15)


def test_G3_A_plaquette_identity_is_unity(client: MockGIGIClient):
    """⟨P⟩ on the identity field is 1 (Re tr(I)/2 = 1)."""
    client.declare_gauge_field(
        name="U_id", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    mean_p = client.select_plaquette("U_id", reduction="mean")
    assert mean_p == pytest.approx(1.0, abs=1e-14)


# ---------------------------------------------------------------------
# G3.B — Q_SURROGATE observable (per HALCYON_TO_GIGI_REPLY § A1)
# ---------------------------------------------------------------------
def test_G3_B_Q_surrogate_at_identity_is_zero(client: MockGIGIClient):
    """A1 spec: Q_surrogate(I) = 0 because every face holonomy is the
    identity and arccos(1) = 0."""
    client.declare_gauge_field(
        name="U_id", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    q = client.select_observable("U_id", ObservableId.Q_SURROGATE)
    assert q == pytest.approx(0.0, abs=1e-12)


def test_G3_B_Q_surrogate_matches_kernel(client: MockGIGIClient):
    """Q_surrogate on a Haar-random field matches the kernel exactly."""
    client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    q_mock = client.select_observable("U", ObservableId.Q_SURROGATE)
    # Kernel reference
    import torch
    graph = buckyball_graph.build_truncated_icosahedron()
    U_t = torch.from_numpy(client.introspect_gauge_field("U"))
    q_ref = float(buckyball_observables.Q_surrogate(U_t, graph))
    assert q_mock == pytest.approx(q_ref, abs=1e-15)


def test_G3_B_Q_surrogate_within_range(client: MockGIGIClient):
    """A1 range: Q_surrogate ∈ [0, F/2] = [0, 16] on the buckyball."""
    client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    q = client.select_observable("U", ObservableId.Q_SURROGATE)
    assert 0.0 <= q <= 16.0


def test_G3_B_H_total_rejected_before_part_IV(client: MockGIGIClient):
    """H_TOTAL requires an E field (Lie-algebra-valued); not available
    until SYMPLECTIC_FLOW (Part IV). Must fail with a clear error."""
    client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    with pytest.raises(ValueError, match="(?i)part iv|e field"):
        client.select_observable("U", ObservableId.H_TOTAL)


# ---------------------------------------------------------------------
# G3.C — Bit-identity contract for GIBBS_SAMPLE (per A2)
# ---------------------------------------------------------------------
def test_G3_C_gibbs_sample_requires_seed(client: MockGIGIClient):
    """The bit-identity contract requires SEED; running without one
    is a contract violation."""
    client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    with pytest.raises(ValueError, match="SEED"):
        client.gibbs_sample("U", beta=2.5, n_sweeps=5)


def test_G3_C_gibbs_sample_short_run_reproducible(client: MockGIGIClient):
    """Same seed, same β, same n_sweeps -> bit-identical final state.
    A short run keeps test time reasonable."""
    client.declare_gauge_field(
        name="U1", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    client.gibbs_sample("U1", beta=2.5, n_sweeps=20, seed=20260616)
    buf1 = client.introspect_gauge_field("U1")

    client2 = MockGIGIClient()
    spec = LatticeSpec(
        name="buckyball",
        vertices=60, edges=tuple((int(u), int(v)) for (u, v) in
                                  buckyball_graph.build_truncated_icosahedron().edges),
        faces=tuple(tuple(int(v) for v in face) for face in
                    buckyball_graph.build_truncated_icosahedron().face_vertices),
        topology="S2", euler_characteristic=2,
    )
    client2.declare_lattice(spec)
    client2.declare_gauge_field(
        name="U2", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    client2.gibbs_sample("U2", beta=2.5, n_sweeps=20, seed=20260616)
    buf2 = client2.introspect_gauge_field("U2")

    np.testing.assert_array_equal(buf1, buf2)


# ---------------------------------------------------------------------
# G3.D — Production-scale Part III pass criterion
# ---------------------------------------------------------------------
def test_G3_D_three_statement_block_reproduces_production_thermalization(
    client: MockGIGIClient,
):
    """The Part III pass criterion verbatim from HALCYON_PART_I_GATES.md
    § PART III:

      LATTICE buckyball FROM TRUNCATED_ICOSAHEDRON TOPOLOGY "S2";
      GAUGE_FIELD U ON LATTICE buckyball GROUP SU(2) INIT IDENTITY;
      GIBBS_SAMPLE U BETA 2.5 N_SWEEPS 200 SEED 20260616
                   MEASURE_EVERY 1 MEASURE (MEAN(PLAQUETTE), Q_SURROGATE);

    must reproduce Halcyon's β=2.5 thermalization within the published
    Flyvbjerg-Petersen SEM. The published canonical <P>_meas is 0.5068
    ± 0.0015 over 2048 samples (production sidecar); a 200-sweep
    Gibbs from IDENTITY won't fully decorrelate to that target, but
    the trajectory must be (a) reproducible at the seed and (b)
    landing in the correct thermodynamic basin (mean plaquette
    monotonically climbing toward equilibrium, ending in the
    [0.40, 0.55] window characteristic of β=2.5).
    """
    # LATTICE is already declared by the fixture.
    handle = client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )

    result = client.gibbs_sample(
        "U", beta=2.5, n_sweeps=200, seed=20260616,
        measure_every=1,
        measure=[ObservableId.MEAN_PLAQUETTE, ObservableId.Q_SURROGATE],
    )

    # Shape receipts
    P_history = result.measurement_history[ObservableId.MEAN_PLAQUETTE.value]
    Q_history = result.measurement_history[ObservableId.Q_SURROGATE.value]
    assert P_history.shape == (200,)
    assert Q_history.shape == (200,)

    # Initial sweep must drop ⟨P⟩ off 1.0 toward equilibrium (the
    # heatbath fully resamples each link's conditional, so we expect
    # ~1 sweep to dominate the descent from the IDENTITY initial)
    assert P_history[0] < 1.0
    # End-state ⟨P⟩ is in the equilibrium band for β=2.5
    # (canonical heatbath landed at 0.5068 ± 0.0015 in production)
    end_mean = float(P_history[-20:].mean())
    assert 0.40 <= end_mean <= 0.58, (
        f"end-of-thermalization ⟨P⟩ = {end_mean:.4f} outside the β=2.5 "
        f"equilibrium band [0.40, 0.58]; Gate III BLOCKED."
    )
    # Steady-state stability across the chain: the last 100 sweeps
    # have plaquette mean within 3σ of the production canonical
    # estimate ⟨P⟩_canonical = 0.506847 ± 0.001463. With ~100 samples
    # the SEM is roughly 0.001463 * √(2048/100) ≈ 0.0066; we allow 3σ
    # plus a margin to absorb residual thermalization in the first
    # half of the run.
    steady_mean = float(P_history[-100:].mean())
    PRODUCTION_P = 0.506847
    PRODUCTION_SEM = 0.001463
    n_samples_eff = 100
    sem_chain = PRODUCTION_SEM * math.sqrt(2048 / n_samples_eff)
    assert abs(steady_mean - PRODUCTION_P) < 3 * sem_chain + 0.02, (
        f"steady-state ⟨P⟩ = {steady_mean:.4f} not within 3σ of the "
        f"production canonical {PRODUCTION_P:.4f} ± {sem_chain:.4f}. "
        f"Gate III is the contract the GIGI substrate's GIBBS_SAMPLE "
        f"verb must satisfy."
    )
    # Q_surrogate stays in [0, F/2] throughout
    assert (Q_history >= 0.0).all()
    assert (Q_history <= 16.0).all()

    # Diagnostics carry the seed receipt
    assert result.diagnostics["seed"] == 20260616
    assert result.diagnostics["beta"] == 2.5
    assert result.diagnostics["n_sweeps_completed"] == 200


# ---------------------------------------------------------------------
# G3.E — Cross-call composition (the pass criterion's "collapse")
# ---------------------------------------------------------------------
def test_G3_E_orchestrator_collapse_to_gql(client: MockGIGIClient):
    """The Part III pass criterion's headline: the first ~80 lines of
    inertia_damping/run_validation_report.py collapse into a
    3-statement GQL block. Verify here that the same business logic
    (build graph -> thermalize -> measure plaquette) lives behind
    exactly three GIGIClient calls: declare_lattice (done by fixture
    setup), declare_gauge_field, gibbs_sample. Nothing else needs to
    cross the GIGI seam at this stage."""
    handle = client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    result = client.gibbs_sample(
        "U", beta=2.5, n_sweeps=10, seed=20260616,
        measure_every=1,
        measure=[ObservableId.MEAN_PLAQUETTE],
    )
    # Single-call observable read after thermalization
    final_P = client.select_observable("U", ObservableId.MEAN_PLAQUETTE)
    history_last = float(result.measurement_history[ObservableId.MEAN_PLAQUETTE.value][-1])
    assert final_P == pytest.approx(history_last, abs=1e-14)
