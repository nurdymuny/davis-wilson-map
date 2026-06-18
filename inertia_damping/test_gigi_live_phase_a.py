"""Phase A live tests — run the Part I + II gates against real gigi-stream.

Phase A scope (per the 2026-06-18 wyqh19me8 audit synthesis):
  - GIGI Parts I + II are live (HTTP surface shipped 2026-06-17).
  - GIGI Part III (GIBBS_SAMPLE) NOT yet shipped → algorithm tests
    stay on MockGIGIClient.
  - Halcyon's production path is embedded (PyO3/CFFI) NOT HTTP — this
    suite is a Phase A end-to-end VERIFICATION, not the swap path.

These tests SKIP cleanly when gigi-stream isn't reachable. To run:

    # Terminal 1 (in the GIGI repo):
    cargo run --release --bin gigi-stream

    # Terminal 2 (in davis-wilson-lattice):
    GIGI_URL=http://localhost:3142 python -m pytest \
        inertia_damping/test_gigi_live_phase_a.py -v

The contracts pinned here:
  G_LIVE_A   POST /v1/lattice + GET /v1/lattice/{name} round-trip
  G_LIVE_B   GAUGE_FIELD INIT IDENTITY: shape, every link = (1,0,0,0)
  G_LIVE_C   GAUGE_FIELD INIT HAAR_RANDOM SEED s: shape, unit-norm
             links, seed echoed back. NOT byte-equal to Mock at same
             seed (xorshift64* vs PCG64, Bee's CSPRNG decision c).
  G_LIVE_D   GAUGE_FIELD INIT FROM_FIELD: byte-equal to source buffer
             (cross-engine pin for Halcyon's test_G2_A receipt).
  G_LIVE_E   Round-trip the IDENTITY buffer through the kernel's face
             walker; all 32 face holonomies = identity quaternion.
             Pins the Part I gold-walker receipt across the engine
             boundary.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# Import torch lazily inside tests if needed (kernel-side walks)
from inertia_damping import buckyball_graph
from inertia_damping.gigi_client import (
    GaugeFieldInit,
    Group,
    LatticeSpec,
    LiveGIGIClient,
)


# ---------------------------------------------------------------------
# Reachability fixture — skip the whole module if gigi-stream is down
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_client() -> LiveGIGIClient:
    url = os.environ.get("GIGI_URL", "http://localhost:3142")
    try:
        client = LiveGIGIClient(base_url=url, ping=True)
    except ConnectionError as ex:
        pytest.skip(
            f"gigi-stream not reachable: {ex}. "
            f"Start it with `cargo run --release --bin gigi-stream` "
            f"or set GIGI_URL to a running instance."
        )
    except RuntimeError as ex:
        pytest.skip(f"LiveGIGIClient setup failed: {ex}")
    return client


@pytest.fixture(scope="module")
def buckyball_spec() -> LatticeSpec:
    graph = buckyball_graph.build_truncated_icosahedron()
    return LatticeSpec(
        name="halcyon_phase_a_buckyball",
        vertices=graph.n_vertices,
        edges=tuple((int(u), int(v)) for (u, v) in graph.edges),
        faces=tuple(tuple(int(v) for v in face) for face in graph.face_vertices),
        topology="S2",
        euler_characteristic=graph.n_vertices - graph.n_edges + graph.n_faces,
    )


# ---------------------------------------------------------------------
# G_LIVE_A — LATTICE round-trip via HTTP
# ---------------------------------------------------------------------
def test_G_LIVE_A_lattice_roundtrip(
    live_client: LiveGIGIClient,
    buckyball_spec: LatticeSpec,
):
    name = live_client.declare_lattice(buckyball_spec)
    assert name == buckyball_spec.name
    view = live_client.lattice_view(name)
    assert view["n_vertices"] == 60
    assert view["n_edges"] == 90
    assert view["n_faces"] == 32
    # The engine should echo the topology hint we sent
    if view.get("topology"):
        assert "S2" in view["topology"]


# ---------------------------------------------------------------------
# G_LIVE_B — GAUGE_FIELD INIT IDENTITY round-trip
# ---------------------------------------------------------------------
def test_G_LIVE_B_identity_field_roundtrip(
    live_client: LiveGIGIClient,
    buckyball_spec: LatticeSpec,
):
    live_client.declare_lattice(buckyball_spec)
    handle = live_client.declare_gauge_field(
        name="U_id_live",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    assert handle.group == Group.SU2
    assert handle.repr_dim == 4
    buf = live_client.introspect_gauge_field("U_id_live")
    assert buf.shape == (90, 4)
    # Identity quaternion = (1, 0, 0, 0) on every edge
    expected = np.zeros((90, 4))
    expected[:, 0] = 1.0
    np.testing.assert_allclose(buf, expected, atol=1e-14, err_msg=(
        "Live GIGI returned a non-identity buffer for INIT IDENTITY. "
        "Part II identity-init contract broken on the engine side."
    ))


# ---------------------------------------------------------------------
# G_LIVE_C — GAUGE_FIELD INIT HAAR_RANDOM (structural, NOT byte-equal to mock)
# ---------------------------------------------------------------------
def test_G_LIVE_C_haar_random_structural(
    live_client: LiveGIGIClient,
    buckyball_spec: LatticeSpec,
):
    live_client.declare_lattice(buckyball_spec)
    seed = 20260616
    handle = live_client.declare_gauge_field(
        name="U_haar_live",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.HAAR_RANDOM,
        seed=seed,
    )
    # The engine should echo the seed back in the response
    assert handle.init_seed == seed, (
        f"engine returned init_seed={handle.init_seed}; expected {seed}. "
        "Halcyon's seed-receipt contract requires the engine to echo "
        "the seed it actually used."
    )
    buf = live_client.introspect_gauge_field("U_haar_live")
    assert buf.shape == (90, 4)
    norms = np.sqrt((buf * buf).sum(axis=1))
    np.testing.assert_allclose(norms, np.ones(90), atol=1e-12, err_msg=(
        "Live GIGI HAAR_RANDOM produced non-unit-quaternion links. "
        "Either Marsaglia is misconfigured or the wire serialization "
        "lost precision."
    ))


def test_G_LIVE_C_haar_random_NOT_byte_equal_to_mock(
    live_client: LiveGIGIClient,
    buckyball_spec: LatticeSpec,
):
    """Bee's CSPRNG decision (option c): mock and live use DIFFERENT
    CSPRNGs (PCG64 vs xorshift64*); byte-equality between them is
    explicitly NOT contracted. This test PINS that decision — if a
    future change makes them byte-equal at the same seed, that's a
    silent contract drift to flag."""
    from inertia_damping.gigi_client import MockGIGIClient

    # Live engine draw
    live_client.declare_lattice(buckyball_spec)
    live_client.declare_gauge_field(
        name="U_diff_a",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.HAAR_RANDOM,
        seed=20260616,
    )
    live_buf = live_client.introspect_gauge_field("U_diff_a")

    # Mock engine draw at the same seed
    mock = MockGIGIClient()
    mock.declare_lattice(buckyball_spec)
    mock.declare_gauge_field(
        name="U_diff_b",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.HAAR_RANDOM,
        seed=20260616,
    )
    mock_buf = mock.introspect_gauge_field("U_diff_b")

    assert not np.array_equal(live_buf, mock_buf), (
        "Mock and Live produced byte-equal HAAR_RANDOM draws at the "
        "same seed. This is a contract violation: Bee's CSPRNG "
        "decision c explicitly DROPPED mock-vs-live byte-equality "
        "(xorshift64* in GIGI vs PCG64 in Mock). If this test fails, "
        "either GIGI changed to PCG64, or the mock changed to "
        "xorshift64* — either way, surface it before merge."
    )


# ---------------------------------------------------------------------
# G_LIVE_D — INIT FROM_FIELD byte-equal to source (cross-engine pin)
# ---------------------------------------------------------------------
def test_G_LIVE_D_from_field_byte_equal_to_source(
    live_client: LiveGIGIClient,
    buckyball_spec: LatticeSpec,
):
    """The cross-engine receipt for test_G2_A_identity_field_round_trip:
    INIT FROM_FIELD must produce a byte-equal copy of the source's
    buffer through GIGI's Rust clone path."""
    live_client.declare_lattice(buckyball_spec)
    live_client.declare_gauge_field(
        name="U_src_live",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.HAAR_RANDOM,
        seed=20260616,
    )
    live_client.declare_gauge_field(
        name="U_dst_live",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.FROM_FIELD,
        from_field="U_src_live",
    )
    src_buf = live_client.introspect_gauge_field("U_src_live")
    dst_buf = live_client.introspect_gauge_field("U_dst_live")
    np.testing.assert_array_equal(src_buf, dst_buf, err_msg=(
        "INIT FROM_FIELD on the live engine returned a buffer that is "
        "NOT byte-equal to the source. Cross-engine contract pin for "
        "test_G2_A_identity_field_round_trip is BROKEN on the engine "
        "side."
    ))


# ---------------------------------------------------------------------
# G_LIVE_E — Face walker on live IDENTITY buffer = all identity holonomies
# ---------------------------------------------------------------------
def test_G_LIVE_E_face_walker_on_live_identity_buffer(
    live_client: LiveGIGIClient,
    buckyball_spec: LatticeSpec,
):
    """End-to-end Phase A receipt: declare IDENTITY field via HTTP,
    introspect its buffer, walk all 32 face holonomies kernel-side.
    Every face holonomy must be the identity quaternion. This pins
    the Part I gold-walker receipt ACROSS the engine boundary."""
    import torch
    from inertia_damping import buckyball_action

    live_client.declare_lattice(buckyball_spec)
    live_client.declare_gauge_field(
        name="U_walker_live",
        lattice_name=buckyball_spec.name,
        group=Group.SU2,
        init=GaugeFieldInit.IDENTITY,
    )
    buf = live_client.introspect_gauge_field("U_walker_live")
    assert buf.shape == (90, 4)

    # Walk faces kernel-side (Part II HTTP doesn't expose HOLONOMY yet)
    graph = buckyball_graph.build_truncated_icosahedron()
    U_torch = torch.from_numpy(buf)
    face_holonomies = buckyball_action.all_face_holonomies(U_torch, graph)
    face_np = face_holonomies.detach().cpu().numpy().astype(np.float64)
    assert face_np.shape == (32, 4)
    expected = np.zeros((32, 4))
    expected[:, 0] = 1.0
    np.testing.assert_allclose(face_np, expected, atol=1e-14, err_msg=(
        "Face holonomies walked over the live engine's IDENTITY buffer "
        "are NOT all identity. Either the wire serialization corrupted "
        "the buffer (signs / row order / repr_dim packing) or the live "
        "engine's IDENTITY init disagrees with the kernel convention."
    ))
