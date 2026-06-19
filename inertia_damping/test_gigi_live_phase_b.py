"""Phase B live tests — fire GIBBS_SAMPLE on real gigi-stream.

THIS IS the architectural unlock: Halcyon's β=2.5 thermalization
runs on GIGI's Rust engine, not the Python kernel. The pass criterion
from ``HALCYON_PART_I_GATES.md`` § PART III is enforced against the
LIVE engine.

GIBBS_SAMPLE reaches the engine via ``POST /v1/gql`` (per the III.7
"no dedicated gibbs_sample route" locked decision D5). The GQL
statement ``GIBBS_SAMPLE U BETA β N_SWEEPS n MEASURE_EVERY k MEASURE
(...) SEED s;`` returns a Rows envelope with the measurement chains
under per-observable column names.

Reference values pinned at the release-profile production canonical
(Gigi's III.8c fix on 2026-06-18, deploy
``deployment-01KVEMVFC226E8QAC4JBYYWQVD``):

    <P>_canonical (release) = 0.5074863
    FP_SEM (release)        = 0.0015235
    over 1948 post-thermal samples at β=2.5 SEED=20260616

These differ from the debug-profile Python-kernel values used by
``test_G3_D`` against the mock (0.506847 ± 0.001463) — same physics,
different FP profile, both correct within their own engine. Bee's
CSPRNG decision (option c) explicitly drops mock↔live byte-equality;
this test pins the *live* contract.

To run:

    # Terminal 1 (GIGI repo):
    cargo run --release --bin gigi-stream

    # Terminal 2 (davis-wilson-lattice):
    GIGI_URL=http://localhost:3142 python -m pytest \\
        inertia_damping/test_gigi_live_phase_b.py -v

The suite skips cleanly when gigi-stream isn't reachable.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import pytest

from inertia_damping import buckyball_graph
from inertia_damping.gigi_client import (
    GaugeFieldInit,
    Group,
    LatticeSpec,
    LiveGIGIClient,
    ObservableId,
)


# Release-profile production canonical (Gigi's III.8c receipt, 2026-06-18)
PRODUCTION_P_RELEASE = 0.5074863
PRODUCTION_SEM_RELEASE = 0.0015235
PRODUCTION_N_SAMPLES = 1948


@pytest.fixture(scope="module")
def live_client() -> LiveGIGIClient:
    url = os.environ.get("GIGI_URL", "http://localhost:3142")
    try:
        return LiveGIGIClient(base_url=url, ping=True)
    except (ConnectionError, RuntimeError) as ex:
        pytest.skip(
            f"gigi-stream not reachable: {ex}. "
            f"Start it with `cargo run --release --bin gigi-stream` "
            f"or set GIGI_URL to a running instance."
        )


@pytest.fixture(scope="module")
def buckyball_spec() -> LatticeSpec:
    graph = buckyball_graph.build_truncated_icosahedron()
    return LatticeSpec(
        name="halcyon_phase_b_buckyball",
        vertices=graph.n_vertices,
        edges=tuple((int(u), int(v)) for (u, v) in graph.edges),
        faces=tuple(tuple(int(v) for v in face) for face in graph.face_vertices),
        topology="S2",
        euler_characteristic=graph.n_vertices - graph.n_edges + graph.n_faces,
    )


# ---------------------------------------------------------------------
# G_LIVE_B0 — GIBBS_SAMPLE smoke (short run, structural receipts)
# ---------------------------------------------------------------------
def test_G_LIVE_B0_gibbs_smoke(
    live_client: LiveGIGIClient,
    buckyball_spec: LatticeSpec,
):
    """Quick structural check: short GIBBS_SAMPLE returns measurement
    history of the right shape. Validates the wire path (GQL parse +
    executor + Rows envelope) before the load-bearing band check."""
    live_client.declare_lattice(buckyball_spec)
    live_client.declare_gauge_field(
        name="U_smoke_live",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    result = live_client.gibbs_sample(
        field_name="U_smoke_live",
        beta=2.5,
        n_sweeps=10,
        measure_every=1,
        measure=[ObservableId.MEAN_PLAQUETTE, ObservableId.Q_SURROGATE],
        seed=20260616,
    )
    P = result.measurement_history[ObservableId.MEAN_PLAQUETTE.value]
    Q = result.measurement_history[ObservableId.Q_SURROGATE.value]
    assert P.shape == (10,)
    assert Q.shape == (10,)
    # Identity-init drops off 1.0 after one heatbath sweep
    assert P[0] < 1.0, f"first sweep should drop <P> from 1.0; got {P[0]}"
    # Q_SURROGATE stays in [0, F/2] = [0, 16] throughout
    assert (Q >= 0.0).all() and (Q <= 16.0).all()
    # Diagnostics carry the seed receipt
    assert result.diagnostics["seed"] == 20260616
    assert result.diagnostics["beta"] == pytest.approx(2.5, abs=1e-12)
    assert result.diagnostics["n_sweeps_completed"] == 10


# ---------------------------------------------------------------------
# G_LIVE_B1 — Bit-identity contract (per HALCYON_TO_GIGI_REPLY § A2)
# ---------------------------------------------------------------------
def test_G_LIVE_B1_gibbs_sample_reproducible_at_same_seed(
    live_client: LiveGIGIClient,
    buckyball_spec: LatticeSpec,
):
    """A2 same-process bit-identity: same seed + same β + same n_sweeps
    on the live engine must reproduce bit-identical measurement
    chains. This is the LIVE side of test_G3_C_gibbs_sample_short_run
    _reproducible."""
    live_client.declare_lattice(buckyball_spec)
    live_client.declare_gauge_field(
        name="U_repro_1",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    r1 = live_client.gibbs_sample(
        "U_repro_1", beta=2.5, n_sweeps=20, seed=20260616,
        measure=[ObservableId.MEAN_PLAQUETTE],
    )

    # Re-declare a fresh field at IDENTITY, run the same chain
    live_client.declare_gauge_field(
        name="U_repro_2",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    r2 = live_client.gibbs_sample(
        "U_repro_2", beta=2.5, n_sweeps=20, seed=20260616,
        measure=[ObservableId.MEAN_PLAQUETTE],
    )

    P1 = r1.measurement_history[ObservableId.MEAN_PLAQUETTE.value]
    P2 = r2.measurement_history[ObservableId.MEAN_PLAQUETTE.value]
    np.testing.assert_array_equal(P1, P2, err_msg=(
        "Same-seed GIBBS_SAMPLE on the live engine produced different "
        "measurement chains. A2 bit-identity contract is broken on the "
        "engine side. Either the registry's mutable handle is leaking "
        "state across calls, or the CSPRNG isn't fully reseeded per call."
    ))


# ---------------------------------------------------------------------
# G_LIVE_B2 — Production thermalization pass criterion (THE algorithm test)
# ---------------------------------------------------------------------
def test_G_LIVE_B2_production_thermalization_pass_criterion(
    live_client: LiveGIGIClient,
    buckyball_spec: LatticeSpec,
):
    """The Part III pass criterion from HALCYON_PART_I_GATES.md §
    PART III, fired against the REAL engine:

        LATTICE buckyball ...
        GAUGE_FIELD U INIT IDENTITY
        GIBBS_SAMPLE U BETA 2.5 N_SWEEPS 200 SEED 20260616
                     MEASURE_EVERY 1 MEASURE (MEAN(PLAQUETTE), Q_SURROGATE)

    must reproduce the release-profile production canonical
    <P> = 0.5074863 ± 0.0015235 within 3σ on the 100-sweep tail.

    This IS "GIGI query instead of Python for the algo." When it
    passes, the architectural promise — same physics, same verdict,
    same caveats, but the gauge-field math is a query on the
    substrate — has live receipt.
    """
    live_client.declare_lattice(buckyball_spec)
    live_client.declare_gauge_field(
        name="U_prod_live",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    result = live_client.gibbs_sample(
        field_name="U_prod_live",
        beta=2.5,
        n_sweeps=200,
        measure_every=1,
        measure=[ObservableId.MEAN_PLAQUETTE, ObservableId.Q_SURROGATE],
        seed=20260616,
    )
    P_history = result.measurement_history[ObservableId.MEAN_PLAQUETTE.value]
    Q_history = result.measurement_history[ObservableId.Q_SURROGATE.value]
    assert P_history.shape == (200,)
    assert Q_history.shape == (200,)

    # Identity-init must drop ⟨P⟩ off 1.0
    assert P_history[0] < 1.0
    # End-state ⟨P⟩ in equilibrium band for β=2.5 (loose sanity)
    end_mean = float(P_history[-20:].mean())
    assert 0.40 <= end_mean <= 0.58, (
        f"end-of-thermalization ⟨P⟩ = {end_mean:.4f} outside the β=2.5 "
        f"equilibrium band [0.40, 0.58]; live engine band BLOCKED."
    )
    # **THE receipt**: 100-sweep tail within 3σ of release-profile canonical
    steady_mean = float(P_history[-100:].mean())
    n_eff = 100
    sem_chain = PRODUCTION_SEM_RELEASE * math.sqrt(PRODUCTION_N_SAMPLES / n_eff)
    # 3σ tolerance + 0.02 absorption margin (matches the mock-side test)
    tolerance = 3 * sem_chain + 0.02
    delta = abs(steady_mean - PRODUCTION_P_RELEASE)
    assert delta < tolerance, (
        f"\n"
        f"  LIVE engine steady-state ⟨P⟩ = {steady_mean:.6f}\n"
        f"  Production canonical (release) = {PRODUCTION_P_RELEASE:.6f}\n"
        f"  Delta = {delta:.6f}, tolerance = {tolerance:.6f}\n"
        f"\n"
        f"  The live engine's β=2.5 thermalization disagrees with the\n"
        f"  III.8a release-profile production canonical. Either the\n"
        f"  release-profile FMA drift is larger than the III.8c fix\n"
        f"  accounted for, or the engine has regressed against its own\n"
        f"  pinned receipt. Cross-reference with halcyon_part_iii_gold::\n"
        f"  tdd_hal_iii_8b_d_production_thermalization_pass_criterion.\n"
    )
    # Q_surrogate stays in [0, 16]
    assert (Q_history >= 0.0).all() and (Q_history <= 16.0).all()
    # Diagnostics
    assert result.diagnostics["seed"] == 20260616
    assert result.diagnostics["beta"] == pytest.approx(2.5, abs=1e-12)
    assert result.diagnostics["n_sweeps_completed"] == 200
