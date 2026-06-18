"""Part II gate test — SU2GaugeField with group-erased storage.

Locks the receipts in HALCYON_PART_I_GATES.md § PART II + the Q2
pushback in HALCYON_TO_GIGI_REPLY_2026-06-17.md:
  - GAUGE_FIELD declare/introspect round-trip is bit-identical.
  - INIT IDENTITY produces the kernel's identity_links exactly.
  - INIT HAAR_RANDOM SEED s is reproducible at the same seed (same
    process / same OS / same BLAS) — the bit-identity contract from
    HALCYON_TO_GIGI_REPLY § A2.
  - HAAR_RANDOM produces an SU(2)-valid distribution (each link has
    quaternion norm 1; aggregate q0 distribution is consistent with
    Haar marginals).
  - Group taxonomy: SU(3) / U(1) / Z(N) compile-fail with a clear
    "not implemented" — the group-erased storage layer exists but
    the verb math is SU(2)-only at launch (Q2 pushback).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from inertia_damping import buckyball_action, buckyball_graph
from inertia_damping.gigi_client import (
    GaugeFieldInit,
    Group,
    LatticeSpec,
    MockGIGIClient,
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
# G2.A — Round-trip: declare -> introspect -> redeclare from buffer
# ---------------------------------------------------------------------
def test_G2_A_identity_field_round_trip(client: MockGIGIClient):
    """Declare INIT IDENTITY, introspect, declare a second field FROM
    the first, introspect — both buffers must be bit-identical."""
    h1 = client.declare_gauge_field(
        name="U1", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    assert h1.repr_dim == 4
    assert h1.group == Group.SU2

    buf1 = client.introspect_gauge_field("U1")
    assert buf1.shape == (90, 4)

    h2 = client.declare_gauge_field(
        name="U2", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.FROM_FIELD,
        from_field="U1",
    )
    buf2 = client.introspect_gauge_field("U2")
    np.testing.assert_array_equal(buf1, buf2)


def test_G2_A_identity_matches_kernel_exactly(client: MockGIGIClient):
    """INIT IDENTITY must produce the same buffer as the kernel's
    identity_links."""
    client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    buf = client.introspect_gauge_field("U")
    kernel_ref = buckyball_action.identity_links(90).detach().cpu().numpy().astype(np.float64)
    np.testing.assert_array_equal(buf, kernel_ref)


# ---------------------------------------------------------------------
# G2.B — HAAR_RANDOM bit-identity contract (A2)
# ---------------------------------------------------------------------
def test_G2_B_haar_random_requires_seed(client: MockGIGIClient):
    """INIT HAAR_RANDOM without a SEED is a contract violation
    (reproducibility requires the seed)."""
    with pytest.raises(ValueError, match="SEED"):
        client.declare_gauge_field(
            name="U", lattice_name="buckyball",
            group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM,
        )


def test_G2_B_haar_random_reproducible_at_same_seed(client: MockGIGIClient):
    """A2 bit-identity contract: same seed, same process -> exactly
    the same link buffer."""
    client.declare_gauge_field(
        name="U1", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    client.declare_gauge_field(
        name="U2", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    buf1 = client.introspect_gauge_field("U1")
    buf2 = client.introspect_gauge_field("U2")
    np.testing.assert_array_equal(buf1, buf2)


def test_G2_B_haar_random_differs_at_different_seed(client: MockGIGIClient):
    """Different seeds must produce different draws (no silent seed
    collapse)."""
    client.declare_gauge_field(
        name="U1", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    client.declare_gauge_field(
        name="U2", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260617,
    )
    buf1 = client.introspect_gauge_field("U1")
    buf2 = client.introspect_gauge_field("U2")
    assert not np.array_equal(buf1, buf2)


# ---------------------------------------------------------------------
# G2.C — HAAR_RANDOM produces SU(2)-valid draws
# ---------------------------------------------------------------------
def test_G2_C_haar_links_have_unit_quaternion_norm(client: MockGIGIClient):
    """Every link in a HAAR_RANDOM draw must be a unit quaternion."""
    client.declare_gauge_field(
        name="U", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=20260616,
    )
    buf = client.introspect_gauge_field("U")
    norms = np.sqrt((buf * buf).sum(axis=1))
    np.testing.assert_allclose(norms, np.ones(90), atol=1e-12)


def test_G2_C_haar_marginals_consistent(client: MockGIGIClient):
    """Aggregate q0 distribution from many seeds must be consistent
    with the Haar marginal on SU(2): p(q0) = (2/π) sqrt(1 - q0²),
    q0 ∈ [-1, 1]. Mean of q0 should be ≈ 0; mean of q0² should be
    ≈ 1/4 (the Haar second moment). Loose tolerances — this is a
    *distribution* check, not a bit-identity check."""
    q0_vals = []
    for seed in range(20260616, 20260616 + 20):
        client.declare_gauge_field(
            name=f"U_{seed}", lattice_name="buckyball",
            group=Group.SU2, init=GaugeFieldInit.HAAR_RANDOM, seed=seed,
        )
        buf = client.introspect_gauge_field(f"U_{seed}")
        q0_vals.append(buf[:, 0])
    q0 = np.concatenate(q0_vals)
    assert q0.shape == (90 * 20,)
    assert abs(q0.mean()) < 0.08            # 0 ± 1/sqrt(N), N=1800
    assert abs((q0 * q0).mean() - 0.25) < 0.05   # Haar 2nd moment


# ---------------------------------------------------------------------
# G2.D — Group taxonomy gate (Q2 pushback)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("group", [Group.SU3, Group.U1, Group.Z2, Group.Z3])
def test_G2_D_non_SU2_groups_raise_NotImplementedError(
    client: MockGIGIClient, group: Group,
):
    """Group-erased storage exists, but the verb math is SU(2)-only at
    launch. Asking for any other group must fail loudly, not silently
    pretend to support it (Q2 pushback in HALCYON_TO_GIGI_REPLY)."""
    with pytest.raises(NotImplementedError, match="SU\\(2\\)"):
        client.declare_gauge_field(
            name="U_other", lattice_name="buckyball",
            group=group, init=GaugeFieldInit.IDENTITY,
        )


def test_G2_D_gauge_field_requires_declared_lattice(client: MockGIGIClient):
    """Cannot declare a GAUGE_FIELD on an undeclared LATTICE."""
    with pytest.raises(ValueError, match="not declared"):
        client.declare_gauge_field(
            name="U_orphan", lattice_name="undeclared_lattice",
            group=Group.SU2, init=GaugeFieldInit.IDENTITY,
        )
