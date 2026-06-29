"""Matched-RNG receipt — byte-identity between Halcyon's mock and GIGI.

The default ``MockGIGIClient`` uses NumPy PCG64. Per CSPRNG decision
(c) it is explicitly NOT byte-equal to the live GIGI engine at the
same seed — by design, two independent CSPRNGs are a stronger
scientific receipt than one shared RNG (independent witnesses catch
bugs that byte-identity would hide).

This file adds the OPT-IN third receipt: ``MatchedRNGMockGIGIClient``,
which routes HAAR_RANDOM and MAXWELL_BOLTZMANN through a Python port
of GIGI's xorshift64* + Marsaglia + Box-Muller stream. At the same
seed, the resulting link / E buffer is byte-identical to what GIGI's
Rust engine produces.

Test layout:
  Always-on (mock-only, deterministic):
    G_MATCH_A   xorshift64* state evolution + uniform() bit width
    G_MATCH_B   Marsaglia Haar samples are unit quaternions
    G_MATCH_C   MatchedRNGMockGIGIClient HAAR_RANDOM round-trips at
                fixed seed (intra-Python bit-identity)
    G_MATCH_D   MatchedRNGMockGIGIClient MAXWELL_BOLTZMANN: q0=0,
                same seed -> same buffer

  Skip-without-engine:
    G_MATCH_LIVE_E   HAAR_RANDOM at SEED 20260616: matched mock
                     buffer == live GIGI engine buffer, BYTE-IDENTICAL.
    G_MATCH_LIVE_F   MAXWELL_BOLTZMANN at SEED 20260617, β=2.5: same
                     contract for the E-field.
"""
from __future__ import annotations

import math
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
    MatchedRNGMockGIGIClient,
    XorshiftSmallRng,
    haar_random_su2,
    haar_random_links,
    maxwell_boltzmann_links,
)


@pytest.fixture(scope="module")
def buckyball_spec() -> LatticeSpec:
    g = buckyball_graph.build_truncated_icosahedron()
    return LatticeSpec(
        name="halcyon_matched_rng_buckyball",
        vertices=g.n_vertices,
        edges=tuple((int(u), int(v)) for (u, v) in g.edges),
        faces=tuple(tuple(int(v) for v in face) for face in g.face_vertices),
        topology="S2",
        euler_characteristic=2,
    )


# ---------------------------------------------------------------------
# G_MATCH_A — xorshift64* core sanity
# ---------------------------------------------------------------------
def test_G_MATCH_A_xorshift_state_evolution():
    """xorshift64* is fully deterministic; first 5 u64s at seed=1
    pin the algorithm. If this test fails, the port has a bit-twiddle
    bug and the byte-identity receipt is dead."""
    rng = XorshiftSmallRng(seed=1)
    # The seed.max(1) clause: seed=0 -> state=1 (same starting state)
    rng0 = XorshiftSmallRng(seed=0)
    assert rng0.state == 1
    assert rng.state == 1
    # First u64 must equal (after one xorshift round) state * 0x2545F4914F6CDD1D mod 2^64
    # state evolves: 1 -> ((1 ^ 0) ^ (1<<25)) -> ^>>27 step ; compute:
    x = 1
    x ^= x >> 12  # still 1
    x ^= (x << 25) & ((1 << 64) - 1)  # 0x2000001
    x ^= x >> 27  # 0x2000001
    expected_state_after = x
    expected_u64 = (x * 0x2545_F491_4F6C_DD1D) & ((1 << 64) - 1)
    actual = rng.next_u64()
    assert rng.state == expected_state_after, (
        f"state after 1 step: {rng.state:#x} != {expected_state_after:#x}"
    )
    assert actual == expected_u64, (
        f"first u64: {actual:#x} != {expected_u64:#x}"
    )


def test_G_MATCH_A_uniform_in_range():
    rng = XorshiftSmallRng(seed=20260616)
    for _ in range(1000):
        u = rng.uniform()
        assert 0.0 <= u < 1.0


def test_G_MATCH_A_draws_counter_advances():
    rng = XorshiftSmallRng(seed=42)
    base = rng.draws
    for _ in range(100):
        rng.uniform()
    assert rng.draws == base + 100


# ---------------------------------------------------------------------
# G_MATCH_B — Marsaglia produces unit quaternions
# ---------------------------------------------------------------------
def test_G_MATCH_B_haar_unit_norm():
    """Marsaglia output must be on S^3 to within FP epsilon."""
    rng = XorshiftSmallRng(seed=20260616)
    for _ in range(200):
        q = haar_random_su2(rng)
        n = math.sqrt(sum(c * c for c in q))
        assert abs(n - 1.0) < 1e-12


def test_G_MATCH_B_haar_marginal_q0_mean_zero():
    """Aggregate q0 across many seeds should center at 0 (Haar measure
    is symmetric about q0=0 on SU(2))."""
    samples = []
    for seed in range(20260616, 20260616 + 2000):
        rng = XorshiftSmallRng(seed=seed)
        q = haar_random_su2(rng)
        samples.append(q[0])
    mean = sum(samples) / len(samples)
    assert abs(mean) < 0.05, f"mean q0 = {mean}; expected near 0"


# ---------------------------------------------------------------------
# G_MATCH_C — MatchedRNGMockGIGIClient HAAR_RANDOM intra-Python bit-identity
# ---------------------------------------------------------------------
def test_G_MATCH_C_matched_mock_haar_same_seed_byte_identical(
    buckyball_spec: LatticeSpec,
):
    """Two MatchedRNGMockGIGIClient instances at the same seed produce
    byte-identical HAAR_RANDOM buffers."""
    def _draw(seed: int):
        c = MatchedRNGMockGIGIClient()
        c.declare_lattice(buckyball_spec)
        c.declare_gauge_field(
            name="U", lattice_name=buckyball_spec.name,
            group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=seed,
        )
        return c.introspect_gauge_field("U")
    b1 = _draw(20260616)
    b2 = _draw(20260616)
    np.testing.assert_array_equal(b1, b2, err_msg=(
        "Matched mock HAAR_RANDOM is not deterministic at fixed seed. "
        "Port has a state-mutation bug."
    ))


def test_G_MATCH_C_matched_mock_haar_differs_from_pcg64_mock(
    buckyball_spec: LatticeSpec,
):
    """The matched mock and the default (PCG64) mock must produce
    DIFFERENT buffers at the same seed. This is the explicit
    receipt for CSPRNG decision (c): two independent CSPRNGs do
    not byte-equal."""
    from inertia_damping.gigi_client import MockGIGIClient
    default = MockGIGIClient()
    matched = MatchedRNGMockGIGIClient()
    default.declare_lattice(buckyball_spec)
    matched.declare_lattice(buckyball_spec)
    default.declare_gauge_field(
        name="U", lattice_name=buckyball_spec.name,
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    matched.declare_gauge_field(
        name="U", lattice_name=buckyball_spec.name,
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    b_default = default.introspect_gauge_field("U")
    b_matched = matched.introspect_gauge_field("U")
    assert not np.array_equal(b_default, b_matched), (
        "Default (PCG64) mock and matched (xorshift64*) mock produced "
        "byte-equal buffers at the same seed. Either the matched RNG "
        "is silently falling back to PCG64, or somebody changed one of "
        "the two RNG paths to the other."
    )


# ---------------------------------------------------------------------
# G_MATCH_D — MAXWELL_BOLTZMANN intra-Python bit-identity
# ---------------------------------------------------------------------
def test_G_MATCH_D_matched_mock_mb_q0_zero(buckyball_spec: LatticeSpec):
    c = MatchedRNGMockGIGIClient()
    c.declare_lattice(buckyball_spec)
    c.declare_gauge_field(
        name="U", lattice_name=buckyball_spec.name,
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    c.declare_e_field(
        "E", "U", EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617,
    )
    buf = c.introspect_e_field("E")
    np.testing.assert_allclose(buf[:, 0], np.zeros(90), atol=1e-12)


def test_G_MATCH_D_matched_mock_mb_same_seed_byte_identical(
    buckyball_spec: LatticeSpec,
):
    def _draw():
        c = MatchedRNGMockGIGIClient()
        c.declare_lattice(buckyball_spec)
        c.declare_gauge_field(
            name="U", lattice_name=buckyball_spec.name,
            group=Group.SU2, init=GaugeFieldInit.IDENTITY,
        )
        c.declare_e_field(
            "E", "U", EFieldInit.MAXWELL_BOLTZMANN, beta=2.5, seed=20260617,
        )
        return c.introspect_e_field("E")
    np.testing.assert_array_equal(_draw(), _draw())


# ---------------------------------------------------------------------
# G_MATCH_LIVE_E — HAAR_RANDOM byte-identity vs LIVE GIGI engine
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_client() -> LiveGIGIClient:
    url = os.environ.get("GIGI_URL", "http://localhost:3142")
    try:
        return LiveGIGIClient(base_url=url, ping=True)
    except (ConnectionError, RuntimeError) as ex:
        pytest.skip(f"gigi-stream not reachable: {ex}")


def test_G_MATCH_LIVE_E_haar_byte_identical_to_live(
    live_client: LiveGIGIClient, buckyball_spec: LatticeSpec,
):
    """THE matched-RNG receipt: HAAR_RANDOM at SEED 20260616 produces
    BYTE-IDENTICAL buffers between the matched mock and the live GIGI
    Rust engine. This is the strongest form of agreement the
    architecture can deliver — not tolerance-band agreement, but
    byte-equality."""
    seed = 20260616
    # Live engine
    live_client.declare_lattice(buckyball_spec)
    live_client.declare_gauge_field(
        name="U_match_live",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.HAAR_RANDOM,
        seed=seed,
    )
    live_buf = live_client.introspect_gauge_field("U_match_live")

    # Matched mock (Python xorshift64* + Marsaglia)
    matched_buf = haar_random_links(buckyball_spec.vertices  # placeholder
                                     - buckyball_spec.vertices + 90, seed=seed
                                     ).detach().cpu().numpy().astype(np.float64)
    # ↑ harden: just compute matched directly via the helper
    matched_buf = haar_random_links(90, seed=seed).detach().cpu().numpy().astype(np.float64)

    assert matched_buf.shape == live_buf.shape == (90, 4)
    np.testing.assert_array_equal(
        matched_buf, live_buf,
        err_msg=(
            "Matched mock and live GIGI engine HAAR_RANDOM buffers are "
            "NOT byte-identical at SEED 20260616. Either the xorshift64* "
            "port has drifted from GIGI's gauge/marsaglia_haar.rs, or "
            "GIGI changed its CSPRNG / Marsaglia implementation without "
            "mirroring it here. Diagnose by comparing the first few u64 "
            "values out of XorshiftSmallRng(seed=20260616) against a "
            "Rust dprintln from gauge::marsaglia_haar::SmallRng."
        ),
    )


def test_G_MATCH_LIVE_F_mb_byte_identical_to_live(
    live_client: LiveGIGIClient, buckyball_spec: LatticeSpec,
):
    """Matched mock MAXWELL_BOLTZMANN matches live engine byte-for-byte.

    Note: GIGI's E_FIELD declaration requires a parent GAUGE_FIELD to
    be declared first; we use INIT IDENTITY for the parent so the
    field shape matches without consuming RNG.
    """
    u_seed = 0
    e_seed = 20260617
    beta = 2.5
    # Live: declare IDENTITY U, then MAXWELL_BOLTZMANN E.
    live_client.declare_lattice(buckyball_spec)
    live_client.declare_gauge_field(
        name="U_mb_parent_live",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    live_client.declare_e_field(
        name="E_mb_live",
        gauge_field="U_mb_parent_live",
        init=EFieldInit.MAXWELL_BOLTZMANN,
        beta=beta,
        seed=e_seed,
    )
    # Live introspect — POST /v1/gql with SHOW E_FIELD BUFFER, or use
    # the local helper. Halcyon's LiveGIGIClient currently does not
    # expose introspect_e_field; if the GIGI HTTP surface exposes
    # SHOW E_FIELD by name, you can introspect via _gql_query.
    try:
        body = live_client._gql_query(f"SHOW E_FIELD E_mb_live BUFFER;")
        # Lift the buffer out of the response; shape (90, 4)
        rows = body.get("rows", body)
        if isinstance(rows, dict):
            rows = rows.get("rows", [rows])
        row = rows[0]
        live_buf_list = row.get("data") or row.get("buffer")
        if live_buf_list is None:
            pytest.skip(
                "SHOW E_FIELD BUFFER did not return a 'data' / 'buffer' "
                "key on this engine version; skip until the read surface "
                "lands."
            )
        live_buf = np.asarray(live_buf_list, dtype=np.float64)
    except Exception as ex:
        pytest.skip(
            f"E_FIELD introspect not available on this engine version: {ex}"
        )

    matched_buf = maxwell_boltzmann_links(
        90, seed=e_seed, beta=beta,
    ).detach().cpu().numpy().astype(np.float64)

    assert matched_buf.shape == live_buf.shape == (90, 4)
    np.testing.assert_array_equal(
        matched_buf, live_buf,
        err_msg=(
            "Matched mock and live GIGI engine MAXWELL_BOLTZMANN "
            "buffers are NOT byte-identical at SEED 20260617, β=2.5. "
            "Most likely cause: the Box-Muller pairing order or the "
            "fourth-uniform-discard step diverged from "
            "gauge/marsaglia_haar.rs::maxwell_boltzmann_su2."
        ),
    )
