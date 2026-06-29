"""
buckyball_action.py — Wilson action and per-link staples for SU(2) 2D Yang–Mills
on the truncated icosahedron (buckyball / soccer-ball graph).

GEOMETRY
    V=60, E=90, F=32, chi=2 (S^2).  Edges carry SU(2) link variables U_e in the
    validated quaternion representation A = a0 I + i a.sigma, stored as a
    real tensor of shape (E, 4) = (90, 4).  Faces are oriented cycles supplied
    by buckyball_graph.build_truncated_icosahedron(): graph.faces[f] is a list
    of (edge_idx, sign in {+1, -1}) tuples; sign = +1 means U_e enters the
    face product as U_e, sign = -1 means U_e^dag enters.  Each edge appears
    in exactly two faces, once with each sign (verified in the graph module
    by edge_sign_sum == 0).

ACTION (Wilson, SU(2), N=2)
    S = (beta / N) * sum_f [N - Re Tr U_f]
      = beta * sum_f [1 - q0(U_f)]
    using Re Tr U / 2 = q0(U) for SU(2).

STAPLE
    For an edge e contained in face f with sign sigma_f(e) in {+1,-1}, define
    the staple A_f at e by the identity
        U_f = U_e^{sigma_f(e)} . A_f                  (sigma=+1 case)
        U_f = U_e^{-sigma_f(e)} . A_f, A_f = (U_e . U_f^dag . U_e)... no — see below.
    Cleanest derivation: walk the oriented face starting at e and accumulate
    the OTHER links in order.  Then U_f = sign_action(U_e, sigma) . A_f.
    The per-edge staple is staple_sum(e) = sum over the two faces of A_f.

    At U=I: every A_f = I, so staple_sum(e) = 2 I for every edge.

GAUGE TRANSFORMATION
    Under U_e -> g(v_from) . U_e . g(v_to)^dag (edges oriented from
    edges[e,0]=v_from to edges[e,1]=v_to), face holonomies conjugate
    by g(v_base): U_f -> g(v_base) U_f g(v_base)^dag, so Re Tr U_f and
    the Wilson action are invariant.

QUATERNION PRIMITIVES
    qmul / qconj / qnorm are lazily imported from the validated
    validation/matter_sector/su2_gauge_higgs.py via importlib (same pattern
    as symplectic_integrator.py).  No re-derivation in this file.

WF#1 / journal entry: see inertia_damping/JOURNAL.md.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, List, Tuple

import numpy as np
import torch

# Validated quaternion primitives are loaded lazily.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

RDTYPE = torch.float64
N_SU2 = 2  # gauge group rank (used in the (beta/N) prefactor)

_su2_mod: Any = None


def _load_su2():
    """Import qmul / qconj / qnorm from the validated matter-sector module."""
    global _su2_mod
    if _su2_mod is not None:
        return _su2_mod
    rel = os.path.join("validation", "matter_sector", "su2_gauge_higgs.py")
    abs_path = os.path.join(_ROOT, rel)
    spec = importlib.util.spec_from_file_location("_su2_gauge_higgs", abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream SU(2) module from {abs_path!r}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _su2_mod = mod
    return mod


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _load_su2().qmul(a, b)


def _qconj(a: torch.Tensor) -> torch.Tensor:
    return _load_su2().qconj(a)


def _qnorm(a: torch.Tensor) -> torch.Tensor:
    return _load_su2().qnorm(a)


def identity_links(n_edges: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return U with every edge set to the identity quaternion (1,0,0,0)."""
    U = torch.zeros((n_edges, 4), dtype=RDTYPE, device=device)
    U[:, 0] = 1.0
    return U


def _signed_link(U: torch.Tensor, edge_idx: int, sign: int) -> torch.Tensor:
    """Return U_e if sign=+1, U_e^dag if sign=-1.  Shape: (4,)."""
    if sign == +1:
        return U[edge_idx]
    elif sign == -1:
        return _qconj(U[edge_idx])
    else:
        raise ValueError(f"orientation sign must be +/-1, got {sign}")


def face_holonomy(
    U: torch.Tensor,
    face_idx: int,
    graph,
) -> torch.Tensor:
    """Ordered product of link variables around face `face_idx`.

    Returns the quaternion (shape (4,)) representing the SU(2) face holonomy
    U_f = U_{e1}^{s1} . U_{e2}^{s2} . ... . U_{ek}^{sk}, where (ei, si) are the
    entries of graph.faces[face_idx] in cyclic order.

    For the explicit 2x2 complex matrix, use face_holonomy_matrix.
    """
    face = graph.faces[face_idx]
    assert len(face) in (5, 6), f"face {face_idx}: length {len(face)} not in (5,6)"
    h = _signed_link(U, face[0][0], face[0][1])
    for eidx, sign in face[1:]:
        h = _qmul(h, _signed_link(U, eidx, sign))
    return h


def face_holonomy_matrix(
    U: torch.Tensor,
    face_idx: int,
    graph,
) -> torch.Tensor:
    """Return the 2x2 complex tensor for face holonomy (convenience)."""
    q = face_holonomy(U, face_idx, graph)
    return _quat_to_matrix_torch(q)


def _quat_to_matrix_torch(q: torch.Tensor) -> torch.Tensor:
    """A = a0 I + i a.sigma, returned as a 2x2 complex128 tensor."""
    q_cpu = q.detach().to("cpu", dtype=RDTYPE)
    a0, a1, a2, a3 = (float(q_cpu[i]) for i in range(4))
    M = torch.zeros((2, 2), dtype=torch.complex128)
    M[0, 0] = a0 + 1j * a3
    M[0, 1] = a2 + 1j * a1
    M[1, 0] = -a2 + 1j * a1
    M[1, 1] = a0 - 1j * a3
    return M


def all_face_holonomies(U: torch.Tensor, graph) -> torch.Tensor:
    """Stack of all 32 face holonomies, shape (n_faces, 4)."""
    out = torch.zeros((graph.n_faces, 4), dtype=RDTYPE, device=U.device)
    for f in range(graph.n_faces):
        out[f] = face_holonomy(U, f, graph)
    return out


def wilson_action(U: torch.Tensor, graph, beta: float) -> torch.Tensor:
    """Wilson action S = (beta/N) sum_f [N - Re Tr U_f] for SU(2).

    For SU(2), Re Tr U_f = 2 q0(U_f), so
        S = (beta/2) sum_f [2 - 2 q0(U_f)] = beta * sum_f [1 - q0(U_f)].
    Returned as a 0-d torch tensor (float64).
    """
    Uf = all_face_holonomies(U, graph)            # (F, 4)
    re_tr = 2.0 * Uf[:, 0]                        # = Re Tr U_f
    contrib = (N_SU2 - re_tr).sum()
    return (beta / N_SU2) * contrib


def _face_index_of_edge_in_face(
    face: List[Tuple[int, int]], edge_idx: int
) -> int:
    """Return the position k such that face[k][0] == edge_idx.  Raises if absent."""
    for k, (e, _s) in enumerate(face):
        if e == edge_idx:
            return k
    raise ValueError(f"edge {edge_idx} not found in face {face}")


def face_staple_at_edge(
    U: torch.Tensor,
    face_idx: int,
    edge_idx: int,
    graph,
) -> torch.Tensor:
    """Per-face staple A_f at edge `edge_idx`, defined by

        U_f  =  U_e^{sigma_f(e)} . A_f

    where U_f is the ordered face product starting at edge_idx.  Concretely
    A_f is the ordered product of the *other* (k-1) signed links of the face.
    Returned as a quaternion (shape (4,)).
    """
    face = graph.faces[face_idx]
    k = _face_index_of_edge_in_face(face, edge_idx)
    # walk the face starting one step past the named edge
    n = len(face)
    e_next, s_next = face[(k + 1) % n]
    A = _signed_link(U, e_next, s_next)
    for j in range(2, n):
        e_j, s_j = face[(k + j) % n]
        A = _qmul(A, _signed_link(U, e_j, s_j))
    return A


def _edge_face_membership(graph) -> List[List[Tuple[int, int]]]:
    """For each edge e, list of (face_idx, sign_of_e_in_face)."""
    cache = getattr(graph, "_edge_face_cache", None)
    if cache is not None:
        return cache
    membership: List[List[Tuple[int, int]]] = [[] for _ in range(graph.n_edges)]
    for fi, face in enumerate(graph.faces):
        for eidx, sign in face:
            membership[eidx].append((fi, sign))
    for e, ms in enumerate(membership):
        assert len(ms) == 2, f"edge {e} touches {len(ms)} faces, expected 2"
        signs = sorted(m[1] for m in ms)
        assert signs == [-1, +1], (
            f"edge {e} sign multiset {signs} != [-1, +1]"
        )
    # cache on the graph object
    try:
        graph._edge_face_cache = membership
    except Exception:
        pass
    return membership


def staple_sum(U: torch.Tensor, edge_idx: int, graph) -> torch.Tensor:
    """Effective staple at edge `edge_idx` for the Wilson action.

    For a face f with edge e entering at orientation sign sigma in {+1,-1}:
        sigma=+1: U_f = U_e . A_f  -> contributes A_f
        sigma=-1: U_f = U_e^dag . A_f, equivalent to U_f^dag = A_f^dag . U_e,
                  so the force-relevant term is A_f^dag.
    Re Tr U_f = Re Tr U_f^dag, so re-writing the sigma=-1 term as U_f^dag puts
    U_e on the LEFT in both cases. The effective staple is therefore
        V_e = sum_{f: sigma=+1} A_f + sum_{f: sigma=-1} A_f^dag.
    Force consumers expect U_e . V_e to give the gauge-covariant Wilson term.
    At U=I every A_f = I and qconj(I) = I, so staple_sum = 2 I exactly.

    Returned as a 2x2 complex tensor.
    """
    membership = _edge_face_membership(graph)
    total_q = torch.zeros(4, dtype=RDTYPE, device=U.device)
    for face_idx, sign in membership[edge_idx]:
        A = face_staple_at_edge(U, face_idx, edge_idx, graph)
        if sign == +1:
            total_q = total_q + A
        else:
            total_q = total_q + _qconj(A)
    return _quat_to_matrix_torch(total_q)


def staple_sum_q(U: torch.Tensor, edge_idx: int, graph) -> torch.Tensor:
    """Same as staple_sum but returned as a quaternion (shape (4,))."""
    membership = _edge_face_membership(graph)
    total_q = torch.zeros(4, dtype=RDTYPE, device=U.device)
    for face_idx, sign in membership[edge_idx]:
        A = face_staple_at_edge(U, face_idx, edge_idx, graph)
        if sign == +1:
            total_q = total_q + A
        else:
            total_q = total_q + _qconj(A)
    return total_q


__all__ = [
    "identity_links",
    "face_holonomy",
    "face_holonomy_matrix",
    "all_face_holonomies",
    "wilson_action",
    "face_staple_at_edge",
    "staple_sum",
    "staple_sum_q",
    "N_SU2",
]
