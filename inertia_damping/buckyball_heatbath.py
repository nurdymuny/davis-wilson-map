"""
buckyball_heatbath.py - Kennedy-Pendleton SU(2) heatbath for 2D Yang-Mills
on the truncated icosahedron (buckyball / soccer-ball graph).

CONVENTIONS (matched to the validated 4D heatbath in lattice/gauge_heatbath_gpu.py
and the action in inertia_damping/buckyball_action.py)
    - SU(2) link variables stored as real quaternions (E, 4) = (90, 4),
      A = a0 I + i a.sigma, so Re Tr A = 2 q0(A).
    - Wilson action: S = (beta/N) sum_f [N - Re Tr U_f], N = 2.
    - For an edge e contained in face f with orientation sign sigma_f(e) in {+1,-1},
      the face holonomy factors as U_f = U_e^{sigma_f(e)} . A_f, with A_f built by
      buckyball_action.face_staple_at_edge.

EFFECTIVE STAPLE
    The local Boltzmann weight at edge e (rest fixed) is
        P(U_e) ~ exp( (beta/N) * Re Tr( U_e . V_eff ) )
    where V_eff is the quaternion
        V_eff = sum_{f \ni e}  A_f^{(+)}   if sigma_f(e) = +1
                              (A_f^{(-)})^dagger   if sigma_f(e) = -1.
    The second case uses Re Tr(U_e^dag . A) = Re Tr(U_e . A^dag) for SU(2).
    On a 3-regular graph every edge sits in exactly 2 faces with opposite signs
    (verified upstream by edge_sign_sum == 0), so V_eff is the sum of one
    A and one A^dag.

KP SAMPLER (single SU(2) link, scalar form -- the buckyball has only 90 edges,
so we update sequentially per edge in CPU)
    1) Compute V_eff quaternion, k = |V_eff|.
    2) If k < EPS_K, draw U_new uniformly from Haar SU(2) (limit case).
    3) Otherwise sample y0 in [-1,1] from rho(y0) ~ sqrt(1-y0^2) exp(xi y0),
       xi = beta * k  (for SU(2): (beta/N) * 2 * k_std = beta * k_std).
       Sample (y1,y2,y3) uniformly on the sphere of radius sqrt(1-y0^2).
       Set U_new = qmul(Y, qconj(V_hat)), V_hat = V_eff / k.

Sequential single-edge updates preserve detailed balance (each step is the
standard Metropolis-rejection-free heatbath conditional on the rest), so no
checkerboard scheme is needed.

PLAQUETTE
    Mean plaquette <P> = mean_f q0(U_f) for SU(2), matching the 4D convention
    (1/N) Re Tr U_p = q0 for N=2.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from inertia_damping.buckyball_action import (
    N_SU2,
    _edge_face_membership,
    _qconj,
    _qmul,
    _qnorm,
    all_face_holonomies,
    face_staple_at_edge,
    identity_links,
)

RDTYPE = torch.float64
EPS_K = 1e-12
MAX_KP_ITERS = 400


# ----------------------------------------------------------------------
# Per-edge effective staple (quaternion form)
# ----------------------------------------------------------------------
def _effective_staple_q(
    U: torch.Tensor,
    edge_idx: int,
    graph,
) -> torch.Tensor:
    """Sum of per-face staples, with the sigma=-1 face contribution conjugated.

    Returns a quaternion of shape (4,) representing V_eff such that the local
    weight is exp((beta/N) Re Tr(U_e . V_eff)).
    """
    membership = _edge_face_membership(graph)
    total = torch.zeros(4, dtype=RDTYPE, device=U.device)
    for face_idx, sigma in membership[edge_idx]:
        A = face_staple_at_edge(U, face_idx, edge_idx, graph)
        if sigma == +1:
            total = total + A
        elif sigma == -1:
            total = total + _qconj(A)
        else:
            raise ValueError(f"bad sigma {sigma} for edge {edge_idx}")
    return total


# ----------------------------------------------------------------------
# Kennedy-Pendleton x0 sampler  (scalar, numpy generator)
# ----------------------------------------------------------------------
def _sample_kp_x0(xi: float, rng: np.random.Generator) -> float:
    """Sample x0 in [-1, 1] from rho(x0) ~ sqrt(1-x0^2) exp(xi x0), xi > 0.

    Kennedy-Pendleton 1985 rejection. Falls back to Haar (x0 in [-1,1] from
    sqrt(1-x0^2)) only if rejection exhausts MAX_KP_ITERS, which essentially
    never triggers for xi > ~1e-6.
    """
    xi_safe = max(xi, 1e-12)
    for _ in range(MAX_KP_ITERS):
        r1 = max(rng.random(), 1e-300)
        r2 = rng.random()
        r3 = max(rng.random(), 1e-300)
        r4 = rng.random()
        c = math.cos(2.0 * math.pi * r2) ** 2
        delta = (-math.log(r1) * c - math.log(r3)) / xi_safe
        if delta < 2.0 and (r4 * r4) <= (1.0 - 0.5 * delta):
            return 1.0 - delta
    # Fallback: Haar-like draw (rare; only when xi is essentially zero)
    while True:
        x = 2.0 * rng.random() - 1.0
        if rng.random() <= math.sqrt(max(1.0 - x * x, 0.0)):
            return x


def _sample_haar_su2(rng: np.random.Generator) -> np.ndarray:
    """Uniform sample from SU(2) Haar measure (returns a quaternion of unit norm)."""
    # Marsaglia / standard: x0 from rho ~ sqrt(1-x0^2) on [-1,1], then sphere.
    while True:
        x0 = 2.0 * rng.random() - 1.0
        if rng.random() <= math.sqrt(max(1.0 - x0 * x0, 0.0)):
            break
    r = math.sqrt(max(1.0 - x0 * x0, 0.0))
    # uniform on 2-sphere of radius r
    cth = 2.0 * rng.random() - 1.0
    sth = math.sqrt(max(1.0 - cth * cth, 0.0))
    phi = 2.0 * math.pi * rng.random()
    return np.array(
        [x0, r * sth * math.cos(phi), r * sth * math.sin(phi), r * cth],
        dtype=np.float64,
    )


def _sample_su2_link(
    V_eff_q: torch.Tensor,
    beta: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sample a new SU(2) link quaternion from the staple-conditioned distribution.

    P(U) ~ exp((beta/N) Re Tr(U . V_eff)) = exp(beta * q0(qmul(U, V_eff)))
    using Re Tr(W) = 2 q0(W) for SU(2) and N = 2.
    """
    v = V_eff_q.detach().cpu().numpy().astype(np.float64)
    k = float(np.linalg.norm(v))
    if k < EPS_K:
        # Pure Haar fallback
        Y = _sample_haar_su2(rng)
        return torch.tensor(Y, dtype=RDTYPE, device=V_eff_q.device)

    xi = beta * k  # SU(2) effective coupling for the KP sampler
    y0 = _sample_kp_x0(xi, rng)
    r = math.sqrt(max(1.0 - y0 * y0, 0.0))
    # uniform on 2-sphere of radius r
    cth = 2.0 * rng.random() - 1.0
    sth = math.sqrt(max(1.0 - cth * cth, 0.0))
    phi = 2.0 * math.pi * rng.random()
    y1 = r * sth * math.cos(phi)
    y2 = r * sth * math.sin(phi)
    y3 = r * cth

    Y_q = torch.tensor([y0, y1, y2, y3], dtype=RDTYPE, device=V_eff_q.device)
    # U_new = Y . V_hat^dag,  V_hat = V_eff / k  (a unit quaternion)
    V_hat = V_eff_q / k
    U_new = _qmul(Y_q, _qconj(V_hat))
    # Numerical projection back to the unit sphere (cheap insurance vs drift)
    nrm = _qnorm(U_new).clamp_min(1e-15)
    return U_new / nrm


# ----------------------------------------------------------------------
# Plaquette measurement
# ----------------------------------------------------------------------
def mean_plaquette(U: torch.Tensor, graph) -> float:
    """<P> = (1/F) sum_f (1/N) Re Tr U_f = (1/F) sum_f q0(U_f)  for SU(2)."""
    Uf = all_face_holonomies(U, graph)  # (F, 4)
    return float(Uf[:, 0].mean().item())


def plaquette_per_face(U: torch.Tensor, graph) -> torch.Tensor:
    """Per-face q0(U_f), shape (F,)."""
    Uf = all_face_holonomies(U, graph)
    return Uf[:, 0].clone()


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def heatbath_sweep(
    U: torch.Tensor,
    graph,
    beta: float,
    generator: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """One full sweep: visit all 90 edges once, in index order.

    Updates U in place AND returns it. Pass a fresh numpy Generator for
    reproducibility; if None, uses np.random.default_rng() (nondeterministic).
    """
    rng = generator if generator is not None else np.random.default_rng()
    n_edges = graph.n_edges
    for e in range(n_edges):
        V_eff = _effective_staple_q(U, e, graph)
        U[e] = _sample_su2_link(V_eff, beta, rng)
    return U


def heatbath_run(
    U_init: torch.Tensor,
    graph,
    beta: float,
    n_sweeps: int,
    measure_every: int = 1,
    generator: Optional[np.random.Generator] = None,
    keep_per_face: bool = False,
) -> Dict[str, object]:
    """Run n_sweeps heatbath sweeps starting from U_init.

    Returns a dict:
        U_final                  : (E, 4) final link tensor
        P_history                : list[float], <P> after each measured sweep
        sweep_indices            : list[int], 1-based sweep numbers measured
        plaquette_per_face_history (optional, if keep_per_face=True): list of (F,) tensors
    """
    rng = generator if generator is not None else np.random.default_rng()
    U = U_init.clone()
    P_history: List[float] = []
    sweep_indices: List[int] = []
    per_face: List[torch.Tensor] = []
    for s in range(1, n_sweeps + 1):
        heatbath_sweep(U, graph, beta, generator=rng)
        if measure_every > 0 and (s % measure_every == 0):
            P_history.append(mean_plaquette(U, graph))
            sweep_indices.append(s)
            if keep_per_face:
                per_face.append(plaquette_per_face(U, graph))
    out: Dict[str, object] = {
        "U_final": U,
        "P_history": P_history,
        "sweep_indices": sweep_indices,
    }
    if keep_per_face:
        out["plaquette_per_face_history"] = per_face
    return out


def thermalize(
    graph,
    beta: float,
    n_sweeps: int = 100,
    n_measure: int = 200,
    n_measure_every: int = 2,
    seed: int = 20260616,
) -> Dict[str, object]:
    """Cold-start, thermalize, then measure <P> over n_measure measurement sweeps.

    Args:
        graph           : BuckyballGraph
        beta            : Wilson coupling
        n_sweeps        : pure-thermalization sweeps (no measurement kept)
        n_measure       : measurement sweeps following thermalization
        n_measure_every : measure <P> every k-th sweep during the measurement
                          phase (sub-sample to thin autocorrelation)
        seed            : numpy Generator seed

    Returns dict with P_mean / P_std / P_sem / U_final / P_history.
    """
    rng = np.random.default_rng(seed)
    U = identity_links(graph.n_edges)
    # Thermalize
    for _ in range(n_sweeps):
        heatbath_sweep(U, graph, beta, generator=rng)
    # Measure
    run = heatbath_run(
        U, graph, beta, n_sweeps=n_measure,
        measure_every=n_measure_every, generator=rng,
    )
    Ps = np.asarray(run["P_history"], dtype=np.float64)
    if Ps.size == 0:
        raise RuntimeError("thermalize: no measurements collected; check n_measure / n_measure_every")
    P_mean = float(Ps.mean())
    P_std = float(Ps.std(ddof=1)) if Ps.size > 1 else 0.0
    P_sem = P_std / math.sqrt(Ps.size) if Ps.size > 1 else 0.0
    return {
        "P_mean": P_mean,
        "P_std": P_std,
        "P_sem": P_sem,
        "n_samples": int(Ps.size),
        "U_final": run["U_final"],
        "P_history": list(Ps),
    }


__all__ = [
    "heatbath_sweep",
    "heatbath_run",
    "thermalize",
    "mean_plaquette",
    "plaquette_per_face",
]


if __name__ == "__main__":
    # Quick smoke test
    from inertia_damping.buckyball_graph import build_truncated_icosahedron
    g = build_truncated_icosahedron()
    out = thermalize(g, beta=2.5, n_sweeps=100, n_measure=100, n_measure_every=1, seed=20260615)
    print(f"beta=2.5  <P>={out['P_mean']:.5f} +/- {out['P_sem']:.5f}  "
          f"(std {out['P_std']:.5f}, n={out['n_samples']})")
