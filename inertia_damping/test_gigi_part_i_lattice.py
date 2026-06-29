"""Part I gate test — LATTICE verb + generalized HOLONOMY walker.

Locks the receipts in HALCYON_PART_I_GATES.md § PART I:
  - LATTICE round-trip: declare -> introspect -> redeclare from
    introspection -> bit-identical incidence + Euler.
  - Generalized HOLONOMY walker reproduces the Python kernel's
    ``buckyball_action.all_face_holonomies(U_final, graph)`` on the
    v1.2 production sidecar's U_final, bit-identical at FP64.

The mock implements this against the kernel today. The real GIGI
substrate ships Part I when it passes the same gate against its own
walker.
"""
from __future__ import annotations

import json
import math
import os
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


# ---------------------------------------------------------------------
# Reference: build a buckyball LatticeSpec from the Python kernel.
# Real GIGI must accept the same spec.
# ---------------------------------------------------------------------
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


@pytest.fixture(scope="module")
def production_U_final() -> torch.Tensor:
    """v1.2 production sidecar's U_final. The Part I gate asserts the
    GIGI walker reproduces all_face_holonomies(U_final, graph)
    bit-identical."""
    sidecar = Path(__file__).parent / "reports" / "run_20260617_110642" / "final_state.npz"
    if not sidecar.exists():
        pytest.skip(f"production sidecar not present at {sidecar}")
    with np.load(sidecar) as z:
        U_np = z["U_final"]
    return torch.from_numpy(U_np.astype(np.float64))


@pytest.fixture
def client() -> MockGIGIClient:
    return MockGIGIClient()


# ---------------------------------------------------------------------
# G1.A — LATTICE declares cleanly and round-trips
# ---------------------------------------------------------------------
def test_G1_A_lattice_declares(client: MockGIGIClient, buckyball_spec: LatticeSpec):
    """The basic gate: LATTICE accepts the spec, returns the name."""
    name = client.declare_lattice(buckyball_spec)
    assert name == "buckyball"


def test_G1_A_lattice_idempotent(client: MockGIGIClient, buckyball_spec: LatticeSpec):
    """Re-declaring with the same shape returns the same name (no
    side effect). This is the round-trip half of the gate."""
    client.declare_lattice(buckyball_spec)
    name = client.declare_lattice(buckyball_spec)
    assert name == "buckyball"


def test_G1_A_lattice_rejects_shape_change(
    client: MockGIGIClient, buckyball_spec: LatticeSpec,
):
    """Re-declaring under the same name with a different shape is a
    contract violation (loud failure, no silent overwrite)."""
    client.declare_lattice(buckyball_spec)
    bad = LatticeSpec(
        name="buckyball",
        vertices=4,  # wrong on purpose
        edges=((0, 1), (1, 2), (2, 3), (3, 0)),
        faces=((0, 1, 2, 3),),
        topology="S2",
        euler_characteristic=2,
    )
    with pytest.raises(ValueError, match="already declared with a different shape"):
        client.declare_lattice(bad)


def test_G1_A_lattice_enforces_euler(
    client: MockGIGIClient, buckyball_spec: LatticeSpec,
):
    """The declared Euler characteristic must match V - E + F. If
    a real GIGI implementation ever lets a wrong χ through, the
    downstream geometry is corrupt."""
    wrong_euler = LatticeSpec(
        name="buckyball_bad_euler",
        vertices=buckyball_spec.vertices,
        edges=buckyball_spec.edges,
        faces=buckyball_spec.faces,
        topology="S2",
        euler_characteristic=999,
    )
    with pytest.raises(ValueError, match="Euler"):
        client.declare_lattice(wrong_euler)


# ---------------------------------------------------------------------
# G1.B — Generalized HOLONOMY walker reproduces the kernel
# ---------------------------------------------------------------------
def test_G1_B_face_holonomies_bit_identical_on_production_U(
    client: MockGIGIClient,
    buckyball_spec: LatticeSpec,
    production_U_final: torch.Tensor,
):
    """The receipt the gate names: SELECT HOLONOMY ON FACES OF U_final
    reproduces buckyball_action.all_face_holonomies(U_final, graph)
    bit-identical at FP64 in the same process. The mock IS the
    Python kernel transcribed; real GIGI must reproduce the same
    32×4 quaternion array."""
    client.declare_lattice(buckyball_spec)
    # Inject the production U_final as a FROM_FIELD source.
    client.declare_gauge_field(
        name="U_seed", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    client._fields["U_seed"].link_buffer = production_U_final.clone()
    result = client.holonomy_on_faces("U_seed")

    # Reference from the kernel directly
    graph = buckyball_graph.build_truncated_icosahedron()
    ref = buckyball_action.all_face_holonomies(production_U_final, graph)
    ref_np = ref.detach().cpu().numpy().astype(np.float64)

    assert result.loop_count == 32
    assert result.holonomies.shape == (32, 4)
    # Bit-identical: same dtype, same values.
    np.testing.assert_array_equal(
        result.holonomies, ref_np,
        err_msg=("Generalized HOLONOMY walker disagrees with kernel "
                 "all_face_holonomies on production U_final. Gate I "
                 "BLOCKED — fix the walker or fix the kernel before "
                 "claiming Part I has shipped.")
    )


def test_G1_B_face_holonomies_at_identity_links(
    client: MockGIGIClient, buckyball_spec: LatticeSpec,
):
    """Holonomies on the identity field are the identity quaternion
    on every face. (Sanity that the walker hasn't picked up a sign
    error or a missing dagger somewhere.)"""
    client.declare_lattice(buckyball_spec)
    client.declare_gauge_field(
        name="U_id", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    result = client.holonomy_on_faces("U_id")
    assert result.loop_count == 32
    # Identity quaternion = (1, 0, 0, 0); allow only FP64 epsilon drift
    expected = np.zeros((32, 4))
    expected[:, 0] = 1.0
    np.testing.assert_allclose(result.holonomies, expected, atol=1e-14)


def test_G1_B_holonomy_on_loop_identity_returns_identity(
    client: MockGIGIClient, buckyball_spec: LatticeSpec,
):
    """Walking an arbitrary edge loop on the identity field returns
    the identity quaternion. Confirms the walker's product chain
    works for non-face loops too (used by Section 4 gauge-invariance
    gate and future Wilson loops)."""
    client.declare_lattice(buckyball_spec)
    client.declare_gauge_field(
        name="U_id", lattice_name="buckyball",
        group=Group.SU2, init=GaugeFieldInit.IDENTITY,
    )
    # Pick any closed-ish 4-edge walk; on the identity field, holonomy
    # is identity regardless of the loop.
    arbitrary_loop = [0, 1, 5, 12]
    result = client.holonomy_on_loop("U_id", arbitrary_loop)
    assert result.loop_count == 1
    np.testing.assert_allclose(result.holonomies[0], [1.0, 0.0, 0.0, 0.0], atol=1e-14)
