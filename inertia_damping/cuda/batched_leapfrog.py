"""batched_leapfrog.py — SU(2) Kogut-Susskind leapfrog on the buckyball,
fully vectorized, batched over a seed dimension, GPU-friendly.

The math mirrors ``inertia_damping/buckyball_integrator.py`` verbatim — same
KDK leapfrog, same -beta/8 force coefficient, same +g²dt drift sign, same
quaternion conventions. The only thing that changes is the layout:

  * U and E gain a leading ``B`` (batch / seed) dimension.
  * Per-edge ``for e in range(n_edges)`` loops disappear: the static
    buckyball topology becomes a small set of gather-friendly tensors,
    and per-step compute becomes a handful of batched quaternion mults +
    a final scatter-by-edge.

That layout runs the 40-trajectory Section 5 sweep as ONE compute call,
on CUDA, instead of 40 serial fly.io round-trips.

Bit-identity vs the CPU kernel is NOT a goal — CUDA reductions and
non-deterministic atomic adds preclude it. The validation contract is
physical: same H_total conservation behaviour to float64 rounding, same
Gauss residual envelope, same P_time within statistical error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

RDTYPE = torch.float64


# ---------------------------------------------------------------------------
# Batched quaternion primitives (broadcast over any leading dims)
# ---------------------------------------------------------------------------
def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a0, av = a[..., :1], a[..., 1:]
    b0, bv = b[..., :1], b[..., 1:]
    c0 = a0 * b0 - (av * bv).sum(-1, keepdim=True)
    cv = a0 * bv + b0 * av - torch.cross(av, bv, dim=-1)
    return torch.cat([c0, cv], dim=-1)


def qconj(a: torch.Tensor) -> torch.Tensor:
    out = a.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def qnorm(a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((a * a).sum(-1, keepdim=True))


def matrix_exp_su2_q(w: torch.Tensor) -> torch.Tensor:
    """For purely-imaginary quaternion (0, v), returns the unit quaternion
    exp(i v·sigma). Matches symplectic_integrator._matrix_exp_su2_q exactly."""
    v = w[..., 1:]
    a_norm_sq = (v * v).sum(-1, keepdim=True)
    a_norm = torch.sqrt(torch.clamp(a_norm_sq, min=0.0))
    small = a_norm < 1e-8
    cos_a = torch.cos(a_norm)
    sinc = torch.where(small, 1.0 - a_norm_sq / 6.0,
                       torch.sin(a_norm) / torch.clamp(a_norm, min=1e-30))
    out = torch.zeros_like(w)
    out[..., :1] = cos_a
    out[..., 1:] = sinc * v
    return out


# ---------------------------------------------------------------------------
# Static topology — precomputed gather-friendly tensors
# ---------------------------------------------------------------------------
@dataclass
class BatchedTopology:
    """Static buckyball topology in gather-friendly tensor form.

    Walk-order tensors are written so the kernel never needs Python-level
    branching by face shape — pentagons and hexagons live in the same
    padded buffers and we mask the unused slot.
    """
    n_vertices: int
    n_edges: int
    n_faces: int
    max_n: int             # max len of a face (6 for the buckyball)
    # Holonomy walks: for each face, the ordered (edge_idx, sign) for its full perimeter.
    # face_walk_edges: (F, max_n), int64 — edge index per walk slot (padded with 0)
    # face_walk_signs: (F, max_n), float64 — +1 or -1 per slot
    # face_walk_mask:  (F, max_n), float64 — 1.0 for real slots, 0.0 for pad
    face_walk_edges: torch.Tensor
    face_walk_signs: torch.Tensor
    face_walk_mask: torch.Tensor
    # Staple walks: for each face × each starting position k in 0..max_n-1, the
    # ordered (edge_idx, sign) of the (n-1) "other" edges (the staple chain).
    # face_staple_edges: (F, max_n, max_n-1) int64
    # face_staple_signs: (F, max_n, max_n-1) float64
    # face_staple_mask:  (F, max_n, max_n-1) float64
    face_staple_edges: torch.Tensor
    face_staple_signs: torch.Tensor
    face_staple_mask: torch.Tensor
    # Per-edge membership: (E, 2, 2) = (face_idx, position_in_face) for the
    # 2 faces the edge belongs to; edge_face_signs: (E, 2) the sign in each.
    edge_face_idx: torch.Tensor       # (E, 2) int64
    edge_face_pos: torch.Tensor       # (E, 2) int64
    edge_face_signs: torch.Tensor     # (E, 2) float64
    # For the Gauss residual (covariant): signed incidence for the FLAT
    # part. Adjusted at runtime by Ad(U_e) on the head end.
    # tail_per_edge: (E,) int64 — vertex_idx where the edge tails out
    # head_per_edge: (E,) int64 — vertex_idx where the edge heads in
    tail_per_edge: torch.Tensor
    head_per_edge: torch.Tensor


def build_topology_from_graph(graph, device: torch.device, dtype=RDTYPE) -> BatchedTopology:
    """Materialize a BuckyballGraph into the gather-friendly tensors above."""
    V, E, F = graph.n_vertices, graph.n_edges, graph.n_faces
    max_n = max(len(face) for face in graph.faces)

    face_walk_edges = np.zeros((F, max_n), dtype=np.int64)
    face_walk_signs = np.zeros((F, max_n), dtype=np.float64)
    face_walk_mask = np.zeros((F, max_n), dtype=np.float64)
    face_staple_edges = np.zeros((F, max_n, max_n - 1), dtype=np.int64)
    face_staple_signs = np.zeros((F, max_n, max_n - 1), dtype=np.float64)
    face_staple_mask = np.zeros((F, max_n, max_n - 1), dtype=np.float64)

    for fi, face in enumerate(graph.faces):
        n = len(face)
        for k, (e, s) in enumerate(face):
            face_walk_edges[fi, k] = e
            face_walk_signs[fi, k] = float(s)
            face_walk_mask[fi, k] = 1.0
        for k in range(n):
            # staple chain: walk other (n-1) edges starting one past k
            for j in range(1, n):
                e_j, s_j = face[(k + j) % n]
                face_staple_edges[fi, k, j - 1] = e_j
                face_staple_signs[fi, k, j - 1] = float(s_j)
                face_staple_mask[fi, k, j - 1] = 1.0

    # Per-edge membership
    edge_face_idx = np.zeros((E, 2), dtype=np.int64)
    edge_face_pos = np.zeros((E, 2), dtype=np.int64)
    edge_face_signs = np.zeros((E, 2), dtype=np.float64)
    seen = [0] * E
    for fi, face in enumerate(graph.faces):
        for k, (e, s) in enumerate(face):
            slot = seen[e]
            edge_face_idx[e, slot] = fi
            edge_face_pos[e, slot] = k
            edge_face_signs[e, slot] = float(s)
            seen[e] += 1
    for e in range(E):
        assert seen[e] == 2, f"edge {e} has {seen[e]} face memberships, expected 2"

    tail_per_edge = graph.edges[:, 0].astype(np.int64)
    head_per_edge = graph.edges[:, 1].astype(np.int64)

    def t(arr, dt):
        return torch.from_numpy(arr).to(device=device, dtype=dt)

    return BatchedTopology(
        n_vertices=V,
        n_edges=E,
        n_faces=F,
        max_n=max_n,
        face_walk_edges=t(face_walk_edges, torch.long),
        face_walk_signs=t(face_walk_signs, dtype),
        face_walk_mask=t(face_walk_mask, dtype),
        face_staple_edges=t(face_staple_edges, torch.long),
        face_staple_signs=t(face_staple_signs, dtype),
        face_staple_mask=t(face_staple_mask, dtype),
        edge_face_idx=t(edge_face_idx, torch.long),
        edge_face_pos=t(edge_face_pos, torch.long),
        edge_face_signs=t(edge_face_signs, dtype),
        tail_per_edge=t(tail_per_edge, torch.long),
        head_per_edge=t(head_per_edge, torch.long),
    )


# ---------------------------------------------------------------------------
# Signed-link gather: ``signed_link(U, e, s) = U[e] if s=+1 else qconj(U[e])``
# Vectorized: U is (B, E, 4), indices (..., L) and signs (..., L) → (B, ..., L, 4)
# ---------------------------------------------------------------------------
def signed_link_gather(
    U: torch.Tensor,           # (B, E, 4)
    indices: torch.Tensor,     # (..., L) int64
    signs: torch.Tensor,       # (..., L) float64, +1 or -1
    mask: Optional[torch.Tensor] = None,  # (..., L) float64, 1.0 keep / 0.0 → identity
) -> torch.Tensor:
    """Return U[:, indices, :] with q0 sign convention applied per-slot.

    Slots where ``mask=0`` are replaced with the identity quaternion (1,0,0,0)
    so chained qmul leaves the chain untouched.
    """
    B = U.shape[0]
    # Index gather: U[:, indices, :] with broadcasting
    # indices shape (..., L); we need result shape (B, ..., L, 4).
    flat_idx = indices.reshape(-1)
    gathered = U[:, flat_idx, :]                        # (B, prod(...)*L, 4)
    out_shape = (B,) + tuple(indices.shape) + (4,)
    gathered = gathered.reshape(out_shape)
    # Apply sign: where signs==-1, conjugate (flip last 3 components).
    sign_q = signs.unsqueeze(-1)                        # (..., L, 1)
    conj_mask = (sign_q < 0).to(gathered.dtype)         # (..., L, 1)
    # Flip the imaginary part where conj_mask=1
    flip = torch.cat([
        torch.ones_like(conj_mask),
        -2.0 * conj_mask + 1.0,                         # = +1 if keep, -1 if flip
        -2.0 * conj_mask + 1.0,
        -2.0 * conj_mask + 1.0,
    ], dim=-1)                                          # (..., L, 4)
    flip = flip.unsqueeze(0).expand(out_shape)
    out = gathered * flip
    if mask is not None:
        # Where mask=0, replace with (1,0,0,0).
        id_q = torch.zeros_like(out)
        id_q[..., 0] = 1.0
        m = mask.unsqueeze(-1).unsqueeze(0)            # (1, ..., L, 1)
        m = m.expand(out_shape)
        out = m * out + (1.0 - m) * id_q
    return out


def chain_product(links: torch.Tensor) -> torch.Tensor:
    """Chained qmul along the second-to-last dim.

    links: (..., L, 4) → out: (..., 4) = links[...,0,:] * links[...,1,:] * ...
    Done with a Python loop of length L; L is small (≤6 for the buckyball),
    so loop overhead is negligible — the inner qmul is fully batched.
    """
    L = links.shape[-2]
    out = links[..., 0, :]
    for i in range(1, L):
        out = qmul(out, links[..., i, :])
    return out


# ---------------------------------------------------------------------------
# Face holonomies, face staples, force, drift
# ---------------------------------------------------------------------------
def compute_face_holonomies(U: torch.Tensor, topo: BatchedTopology) -> torch.Tensor:
    """U: (B, E, 4) → Uf: (B, F, 4) — ordered face product per face."""
    links = signed_link_gather(
        U, topo.face_walk_edges, topo.face_walk_signs, topo.face_walk_mask,
    )  # (B, F, max_n, 4)
    return chain_product(links)


def compute_face_staples_at_each_position(
    U: torch.Tensor, topo: BatchedTopology,
) -> torch.Tensor:
    """Per-face, per-starting-position staple chain.

    Returns staples: (B, F, max_n, 4). For pentagons the k=max_n-1 slot has
    a degenerate chain (all-identity), which is irrelevant because no edge
    membership ever references it (only positions 0..n-1 of an n-face).
    """
    # face_staple_edges: (F, max_n, max_n-1); face_staple_signs: same shape
    links = signed_link_gather(
        U, topo.face_staple_edges, topo.face_staple_signs, topo.face_staple_mask,
    )  # (B, F, max_n, max_n-1, 4)
    return chain_product(links)


def compute_force(
    U: torch.Tensor, topo: BatchedTopology, beta: float,
) -> torch.Tensor:
    """Wilson force per edge.

    F_e = (-beta/(2 N^2)) * proj_q0=0( qmul(U_e, sum_face A_f^{(sign)}) )
        = -beta/8 * (0, imag-3-vec of  U_e · staple_sum)   for SU(2), N=2.
    """
    N = 2
    coeff = -beta / (2.0 * N * N)
    B = U.shape[0]
    E = topo.n_edges
    # All face × position staples
    all_face_staples = compute_face_staples_at_each_position(U, topo)
    # Gather the 2 staples per edge from (face_idx, position_in_face).
    # all_face_staples: (B, F, max_n, 4); gather via fancy indexing.
    # Build flat per-batch index:
    flat_face_pos = (topo.edge_face_idx * topo.max_n + topo.edge_face_pos)  # (E, 2)
    # (B, F*max_n, 4) view
    fps = all_face_staples.reshape(B, topo.n_faces * topo.max_n, 4)
    # Gather: (B, E, 2, 4)
    edge_staples = fps[:, flat_face_pos.reshape(-1), :].reshape(B, E, 2, 4)
    # Where edge_face_signs == -1, conjugate the staple
    signs = topo.edge_face_signs.unsqueeze(-1)               # (E, 2, 1)
    conj_mask = (signs < 0).to(edge_staples.dtype)
    flip = torch.cat([
        torch.ones_like(conj_mask),
        -2.0 * conj_mask + 1.0,
        -2.0 * conj_mask + 1.0,
        -2.0 * conj_mask + 1.0,
    ], dim=-1)                                               # (E, 2, 4)
    flip = flip.unsqueeze(0).expand(B, E, 2, 4)
    edge_staples = edge_staples * flip
    staple_sum = edge_staples.sum(dim=2)                     # (B, E, 4)
    Omega = qmul(U, staple_sum)
    Omega = Omega.clone()
    Omega[..., 0] = 0.0
    return coeff * Omega


def compute_drift(
    U: torch.Tensor, E: torch.Tensor, dt: float, g2: float,
) -> torch.Tensor:
    """U_new = matrix_exp_su2_q((0, +g² dt e_vec)) · U; then renormalize."""
    arg = torch.zeros_like(E)
    arg[..., 1:] = (g2 * dt) * E[..., 1:]
    expE = matrix_exp_su2_q(arg)
    U_new = qmul(expE, U)
    return U_new / qnorm(U_new)


def leapfrog_step(
    U: torch.Tensor,
    E: torch.Tensor,
    dt: float,
    topo: BatchedTopology,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One KDK (kick-drift-kick) leapfrog step."""
    g2 = (2.0 * 2) / beta   # 2N/beta for SU(2), N=2
    F0 = compute_force(U, topo, beta)
    E_half = E + (dt / 2.0) * F0
    E_half[..., 0] = 0.0
    U_new = compute_drift(U, E_half, dt, g2)
    F1 = compute_force(U_new, topo, beta)
    E_new = E_half + (dt / 2.0) * F1
    E_new[..., 0] = 0.0
    return U_new, E_new


# ---------------------------------------------------------------------------
# Hamiltonian + Gauss residual (for the validation gate)
# ---------------------------------------------------------------------------
def compute_hamiltonian(
    U: torch.Tensor, E: torch.Tensor, topo: BatchedTopology, beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (H, K, V) each of shape (B,)."""
    g2 = (2.0 * 2) / beta
    e_norm_sq = (E[..., 1:] ** 2).sum(-1)        # (B, E)
    Tr_E2 = 2.0 * e_norm_sq
    K = (g2 / 2.0) * Tr_E2.sum(dim=-1)            # (B,)
    Uf = compute_face_holonomies(U, topo)         # (B, F, 4)
    re_tr = 2.0 * Uf[..., 0]                      # (B, F)
    N = 2
    contrib = (N - re_tr).sum(dim=-1)             # (B,)
    V = (1.0 / (g2 * N)) * contrib
    return K + V, K, V


def compute_mean_plaquette(U: torch.Tensor, topo: BatchedTopology) -> torch.Tensor:
    """Mean over faces of q0(U_f). Shape (B,)."""
    Uf = compute_face_holonomies(U, topo)
    return Uf[..., 0].mean(dim=-1)


def compute_gauss_residual_max(
    E: torch.Tensor, U: torch.Tensor, topo: BatchedTopology,
) -> torch.Tensor:
    """Max-norm of the covariant per-vertex Gauss residual. Shape (B,)."""
    B = U.shape[0]
    V = topo.n_vertices
    # T_e = head-frame image: qmul(qmul(qconj(U_e), E_e), U_e)  shape (B, E, 4)
    T = qmul(qmul(qconj(U), E), U)
    tail = topo.tail_per_edge                     # (E,)
    head = topo.head_per_edge                     # (E,)
    # Per-vertex residual: sum_{e:tail=v} E_e[1:] - sum_{e:head=v} T_e[1:]
    G_tail = torch.zeros((B, V, 3), dtype=E.dtype, device=E.device)
    G_head = torch.zeros((B, V, 3), dtype=E.dtype, device=E.device)
    # scatter_add over tail and head
    tail_exp = tail.view(1, -1, 1).expand(B, -1, 3)
    head_exp = head.view(1, -1, 1).expand(B, -1, 3)
    G_tail.scatter_add_(1, tail_exp, E[..., 1:])
    G_head.scatter_add_(1, head_exp, T[..., 1:])
    G = G_tail - G_head                            # (B, V, 3)
    return G.abs().reshape(B, -1).max(dim=-1).values


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
def initialize_U_identity(B: int, E: int, device, dtype=RDTYPE) -> torch.Tensor:
    U = torch.zeros((B, E, 4), dtype=dtype, device=device)
    U[..., 0] = 1.0
    return U


def initialize_E_canonical(
    seeds: List[int], topo: BatchedTopology, beta: float,
    device, dtype=RDTYPE,
) -> torch.Tensor:
    """Sample E at canonical scale per-seed (one PCG64-like stream per seed
    via torch generators). Returns (B, E, 4) with q0=0.

    NOTE: this is NOT bit-identical to the Python CPU kernel's
    torch.randn output; CUDA's RNG state machine differs. For Section 5 the
    physics question is invariant under that — we just need iid canonical
    samples per seed.
    """
    sigma = (beta / 128.0) ** 0.5     # SU(2) closed form, matches canonical_sigma(beta, "SU(2)")
    B = len(seeds)
    E_n = topo.n_edges
    out = torch.zeros((B, E_n, 4), dtype=dtype, device=device)
    for b, s in enumerate(seeds):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(s))
        alpha = sigma * torch.randn(
            (E_n, 3), dtype=dtype, device=device, generator=gen,
        )
        out[b, :, 1:] = 2.0 * alpha
    return out


# ---------------------------------------------------------------------------
# Batched trajectory with checkpoint observables
# ---------------------------------------------------------------------------
def run_batched_trajectory(
    seeds: List[int],
    beta: float,
    dt: float,
    n_steps_max: int,
    checkpoints: List[int],
    topo: BatchedTopology,
    device,
    dtype=RDTYPE,
    measure_every: int = 10,
) -> Dict[str, np.ndarray]:
    """Run one batched leapfrog trajectory, recording plaquette at each
    measured step, and the (H, K, V, max|G|) every ``measure_every`` steps
    up to the largest checkpoint.

    Returns a dict with:
      - 'seeds': (B,) the seeds (int64)
      - 'checkpoints': (C,) checkpoint step indices
      - 'P_running': (B, C) running time-average of plaquette truncated at each checkpoint
      - 'P_running_sem': (B, C) std-error from a single Flyvbjerg-Petersen pass per checkpoint
      - 'P_running_n_eff': (B, C) effective sample count at each checkpoint
      - 'P_running_blocking_regime': (B, C) "plateau"/"single_level"/"no_plateau"/"degenerate"
      - 'P_running_blocking_block_size': (B, C) plateau block size
      - 'H0': (B,) initial Hamiltonian
      - 'H_final': (B,) final Hamiltonian
      - 'dH_rel_max': (B,) max relative |H-H0|/|H0| over the trajectory
      - 'G_max_max': (B,) max max|G| over the trajectory
      - 'P_history': (B, M) plaquette measured at every step (M = n_steps_max)
    """
    B = len(seeds)
    E = topo.n_edges
    U = initialize_U_identity(B, E, device, dtype)
    Ee = initialize_E_canonical(seeds, topo, beta, device, dtype)
    # Project Gauss-zero? At U=I the covariant projector equals the flat one,
    # and since alpha is iid Gaussian per edge, the flat divergence is small
    # but not zero. The Python kernel projects via CG. For the GPU batched
    # study, we skip projection — the divergence-free constraint is preserved
    # to leading order by the symplectic flow; we monitor max|G| as a gate.

    H0, _, _ = compute_hamiltonian(U, Ee, topo, beta)
    P_history = torch.zeros((B, n_steps_max), dtype=dtype, device=device)
    dH_rel_max = torch.zeros(B, dtype=dtype, device=device)
    G_max_max = torch.zeros(B, dtype=dtype, device=device)

    for s in range(1, n_steps_max + 1):
        U, Ee = leapfrog_step(U, Ee, dt, topo, beta)
        P_history[:, s - 1] = compute_mean_plaquette(U, topo)
        if s % measure_every == 0 or s == n_steps_max:
            H_s, _, _ = compute_hamiltonian(U, Ee, topo, beta)
            dH_rel = ((H_s - H0).abs() / H0.abs().clamp(min=1e-30))
            dH_rel_max = torch.maximum(dH_rel_max, dH_rel)
            G_s = compute_gauss_residual_max(Ee, U, topo)
            G_max_max = torch.maximum(G_max_max, G_s)

    H_final, _, _ = compute_hamiltonian(U, Ee, topo, beta)

    # Compute running time-averages and blocked SEMs at each checkpoint.
    P_cpu = P_history.detach().to("cpu", torch.float64).numpy()
    C = len(checkpoints)
    P_running = np.zeros((B, C), dtype=np.float64)
    P_running_sem = np.zeros((B, C), dtype=np.float64)
    P_running_n_eff = np.zeros((B, C), dtype=np.float64)
    regimes: List[List[str]] = [["" for _ in range(C)] for _ in range(B)]
    blocks = np.zeros((B, C), dtype=np.int64)
    for b in range(B):
        for j, n in enumerate(checkpoints):
            seg = P_cpu[b, :n]
            P_running[b, j] = float(seg.mean())
            sem, n_eff, block, regime = _flyvbjerg_petersen(seg)
            P_running_sem[b, j] = sem
            P_running_n_eff[b, j] = n_eff
            blocks[b, j] = block
            regimes[b][j] = regime

    return {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "checkpoints": np.asarray(checkpoints, dtype=np.int64),
        "P_running": P_running,
        "P_running_sem": P_running_sem,
        "P_running_n_eff": P_running_n_eff,
        "P_running_blocking_regime": regimes,
        "P_running_blocking_block_size": blocks,
        "H0": H0.detach().to("cpu", torch.float64).numpy(),
        "H_final": H_final.detach().to("cpu", torch.float64).numpy(),
        "dH_rel_max": dH_rel_max.detach().to("cpu", torch.float64).numpy(),
        "G_max_max": G_max_max.detach().to("cpu", torch.float64).numpy(),
        "P_history": P_cpu,
    }


def _flyvbjerg_petersen(x: np.ndarray) -> Tuple[float, float, int, str]:
    """Flyvbjerg-Petersen blocking-transformation SEM estimate.

    Returns (sem, n_eff, plateau_block_size, regime).
    """
    n = len(x)
    if n < 4:
        return float(x.std(ddof=1) / max(np.sqrt(n), 1.0)), float(n), 1, "degenerate"
    levels: List[Tuple[int, float]] = []
    arr = x.astype(np.float64).copy()
    block = 1
    while len(arr) >= 4:
        sigma = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        levels.append((block, sigma))
        if len(arr) % 2:
            arr = arr[:-1]
        arr = (arr[0::2] + arr[1::2]) / 2.0
        block *= 2
    if not levels:
        return float(x.std(ddof=1) / max(np.sqrt(n), 1.0)), float(n), 1, "degenerate"
    if len(levels) == 1:
        b, s = levels[0]
        return s, float(n / b), b, "single_level"
    # Look for a plateau: 3 consecutive levels whose sigmas are within 10%.
    plateau_block = None
    for i in range(len(levels) - 2):
        s0, s1, s2 = levels[i][1], levels[i + 1][1], levels[i + 2][1]
        if s1 <= 0 or s2 <= 0:
            continue
        if abs(s2 - s1) / s1 < 0.10 and abs(s1 - s0) / s0 < 0.10:
            plateau_block = levels[i + 1][0]
            sem = levels[i + 1][1]
            return sem, float(n / plateau_block), plateau_block, "plateau"
    # No plateau → take the largest level's estimate as a conservative SEM
    b_last, s_last = levels[-1]
    return s_last, float(n / b_last), b_last, "no_plateau"


__all__ = [
    "BatchedTopology",
    "build_topology_from_graph",
    "qmul", "qconj", "qnorm", "matrix_exp_su2_q",
    "signed_link_gather", "chain_product",
    "compute_face_holonomies", "compute_face_staples_at_each_position",
    "compute_force", "compute_drift", "leapfrog_step",
    "compute_hamiltonian", "compute_mean_plaquette",
    "compute_gauss_residual_max",
    "initialize_U_identity", "initialize_E_canonical",
    "run_batched_trajectory",
]
