"""su3_4d_glueball_M2.py - M2 channel for 4D SU(3): 0++ scalar glueball via
APE smearing + variational basis + GEVP.

Same structure as su2_4d_glueball_M2.py but adapted to SU(3) conventions
(matrix-valued links, complex128, normalization Re Tr U / 3, SVD polar
projection back into SU(3)).

References
----------
- APE smearing: Albanese et al., Phys. Lett. B 192 (1987) 163.
- GEVP: Luscher & Wolff, Nucl. Phys. B 339 (1990) 222.
- SU(3) glueball spectrum on the lattice: M. Teper,
  "Glueball masses and other physical properties of SU(N) gauge theories
  in D=3+1: a review of lattice results for theorists", hep-th/9812187.

Conventions match inertia_damping/su3_4d_heatbath_gpu.py and the underlying
lattice/gauge_heatbath_gpu.py kernel:
    - U shape (4, L, L, L, L, 3, 3), complex128.
    - Direction 0 = time. Spatial dirs = 1, 2, 3.
    - Wilson normalization: O ~ Re Tr U_p / 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import linalg as scipy_linalg


CDTYPE = torch.complex128
RDTYPE = torch.float64
SPATIAL_DIRS = (1, 2, 3)
N_COLORS = 3


# ----------------------------------------------------------------------
# SU(3) helpers
# ----------------------------------------------------------------------
def dag(M: torch.Tensor) -> torch.Tensor:
    """Conjugate transpose on the last two axes."""
    return M.conj().transpose(-1, -2)


def retr(M: torch.Tensor) -> torch.Tensor:
    """Real part of the trace over the last two axes -> shape (...)."""
    return M.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real


def project_su3(M: torch.Tensor) -> torch.Tensor:
    """SVD polar projection of a batched complex matrix back into SU(3).

    Given M (..., 3, 3) complex, compute the polar factor U = V W^H where
    M = V S W^H is the SVD, then rescale by phase to enforce det = 1.

    This is the standard projection for APE smearing in SU(N) (see Teper
    hep-th/9812187 sec. 5; the same recipe is used in CHROMA and MILC).
    """
    # torch.linalg.svd handles complex; returns U, S, Vh.
    U_svd, S, Vh = torch.linalg.svd(M)
    U_unitary = U_svd @ Vh
    # Enforce det = 1 by dividing by det(U_unitary)^(1/N).
    det = torch.linalg.det(U_unitary)
    # Use principal Nth root via complex log/exp to avoid branch issues.
    phase = torch.exp(-torch.log(det) / N_COLORS)
    return U_unitary * phase.unsqueeze(-1).unsqueeze(-1)


# ----------------------------------------------------------------------
# APE smearing for SU(3) (spatial-only)
# ----------------------------------------------------------------------
def _spatial_staple_sum(U_spatial: torch.Tensor, mu_idx: int) -> torch.Tensor:
    """Sum of the four spatial staples around each link in direction
    SPATIAL_DIRS[mu_idx]. Mirrors the SU(2) version's geometry.

    U_spatial: shape (3, L, L, L, L, 3, 3) complex.
    """
    mu = SPATIAL_DIRS[mu_idx]
    Umu = U_spatial[mu_idx]
    staple_total = torch.zeros_like(Umu)
    for nu_idx, nu in enumerate(SPATIAL_DIRS):
        if nu == mu:
            continue
        Unu = U_spatial[nu_idx]
        # Forward staple: U_nu(x) U_mu(x+nu) U_nu(x+mu)^dagger
        Umu_at_nu = torch.roll(Umu, -1, dims=nu)
        Unu_at_mu = torch.roll(Unu, -1, dims=mu)
        fwd = Unu @ Umu_at_nu @ dag(Unu_at_mu)
        # Backward staple: U_nu(x-nu)^dagger U_mu(x-nu) U_nu(x-nu+mu)
        Unu_back = torch.roll(Unu, +1, dims=nu)
        Umu_back = torch.roll(Umu, +1, dims=nu)
        Unu_back_at_mu = torch.roll(Unu_back, -1, dims=mu)
        bwd = dag(Unu_back) @ Umu_back @ Unu_back_at_mu
        staple_total = staple_total + fwd + bwd
    return staple_total


def ape_smear_links(U_spatial: torch.Tensor,
                    alpha: float = 0.5,
                    n_smears: int = 0) -> torch.Tensor:
    """APE-smear spatial SU(3) links n_smears times with weight alpha.

    U_spatial: shape (3, L, L, L, L, 3, 3) complex. Returns same shape,
    still in SU(3) after SVD polar projection at each smearing step.
    """
    if n_smears <= 0:
        return U_spatial.clone()
    U = U_spatial.clone()
    for _ in range(n_smears):
        U_new = torch.zeros_like(U)
        for mu_idx in range(3):
            staple = _spatial_staple_sum(U, mu_idx)
            blend = (1.0 - alpha) * U[mu_idx] + (alpha / 6.0) * staple
            U_new[mu_idx] = project_su3(blend)
        U = U_new
    return U


# ----------------------------------------------------------------------
# Plaquette operator (0++)
# ----------------------------------------------------------------------
def build_zero_momentum_plaquette_op(U_spatial_smeared: torch.Tensor) -> torch.Tensor:
    """O(t) = (1/V_spatial) sum_{x,y,z, spatial planes} Re Tr U_p(x,y,z,t) / 3.

    Returns shape (L,) real.
    """
    L = U_spatial_smeared.shape[1]
    spatial_planes = [(0, 1), (0, 2), (1, 2)]
    out = torch.zeros(L, dtype=RDTYPE, device=U_spatial_smeared.device)
    for mu_idx, nu_idx in spatial_planes:
        Um = U_spatial_smeared[mu_idx]
        Un = U_spatial_smeared[nu_idx]
        mu = SPATIAL_DIRS[mu_idx]
        nu = SPATIAL_DIRS[nu_idx]
        Un_shift = torch.roll(Un, -1, dims=mu)
        Um_shift = torch.roll(Um, -1, dims=nu)
        Uplaq = Um @ Un_shift @ dag(Um_shift) @ dag(Un)
        out = out + (retr(Uplaq) / N_COLORS).mean(dim=(1, 2, 3))
    return out / 3.0


# ----------------------------------------------------------------------
# Variational basis assembly
# ----------------------------------------------------------------------
def build_operator_basis(
    U_per_cfg: Sequence[torch.Tensor],
    smear_levels: Sequence[int] = (0, 5, 10, 15),
    alpha: float = 0.5,
) -> np.ndarray:
    """For each config and smear level, compute O(t) per time slice.

    U_per_cfg: list of full SU(3) link tensors with shape (4, L, L, L, L, 3, 3),
    complex. Returns shape (n_cfg, N_smear, L) float64.
    """
    n_cfg = len(U_per_cfg)
    if n_cfg == 0:
        raise ValueError("U_per_cfg is empty")
    L = U_per_cfg[0].shape[1]
    N_smear = len(smear_levels)
    out = np.zeros((n_cfg, N_smear, L), dtype=np.float64)
    for c, U_full in enumerate(U_per_cfg):
        U_spatial = U_full[1:4]
        for i, n_smr in enumerate(smear_levels):
            U_smr = ape_smear_links(U_spatial, alpha=alpha, n_smears=int(n_smr))
            op_t = build_zero_momentum_plaquette_op(U_smr)
            out[c, i, :] = op_t.detach().cpu().numpy().astype(np.float64)
    return out


# ----------------------------------------------------------------------
# Correlator matrix + GEVP (numerics identical to SU(2); reuse pattern)
# ----------------------------------------------------------------------
def build_correlator_matrix(O_per_smear_per_cfg: np.ndarray, t: int) -> np.ndarray:
    n_cfg, N_smear, L = O_per_smear_per_cfg.shape
    means = O_per_smear_per_cfg.mean(axis=(0, 2))
    C = np.zeros((N_smear, N_smear), dtype=np.float64)
    O = O_per_smear_per_cfg
    O_shift = np.roll(O, -t, axis=2)
    for i in range(N_smear):
        for j in range(N_smear):
            prod = O[:, i, :] * O_shift[:, j, :]
            C[i, j] = prod.mean() - means[i] * means[j]
    C = 0.5 * (C + C.T)
    return C


def gevp_solve(C_t: np.ndarray, C_t0: np.ndarray) -> np.ndarray:
    C_t = 0.5 * (C_t + C_t.T)
    C_t0 = 0.5 * (C_t0 + C_t0.T)
    eigvals = scipy_linalg.eigvals(C_t, C_t0)
    real_parts = np.real(eigvals)
    order = np.argsort(-real_parts)
    return real_parts[order]


# ----------------------------------------------------------------------
# Mass extraction
# ----------------------------------------------------------------------
@dataclass
class GlueballM2Report:
    L: int
    beta: float
    n_configs: int
    smear_levels: Tuple[int, ...]
    alpha: float
    t0: int
    t_window: Tuple[int, int]
    lambda_max_t: np.ndarray
    m_0pp_t: np.ndarray
    m_0pp: Optional[float]
    m_0pp_error: Optional[float]


def _m_per_t(O_per_smear_per_cfg: np.ndarray, t0: int) -> np.ndarray:
    L = O_per_smear_per_cfg.shape[2]
    out = np.full(L, np.nan, dtype=np.float64)
    C_t0 = build_correlator_matrix(O_per_smear_per_cfg, t0)
    eps = 1e-12 * (np.trace(np.abs(C_t0)) / max(C_t0.shape[0], 1) + 1e-15)
    C_t0_reg = C_t0 + eps * np.eye(C_t0.shape[0])
    for t in range(L):
        if t == t0:
            continue
        C_t = build_correlator_matrix(O_per_smear_per_cfg, t)
        try:
            eigs = gevp_solve(C_t, C_t0_reg)
        except Exception:
            continue
        lam = float(eigs[0])
        if lam <= 0:
            continue
        dt = (t - t0)
        if dt == 0:
            continue
        out[t] = -np.log(lam) / dt
    return out


def extract_m_0pp(
    O_per_smear_per_cfg: np.ndarray,
    t_window: Tuple[int, int],
    t0: int = 1,
    L: int = 0,
    beta: float = 0.0,
    smear_levels: Sequence[int] = (0, 5, 10, 15),
    alpha: float = 0.5,
) -> GlueballM2Report:
    n_cfg, N_smear, L_op = O_per_smear_per_cfg.shape
    if L == 0:
        L = L_op
    m_full = _m_per_t(O_per_smear_per_cfg, t0)
    lam_max = np.full(L_op, np.nan, dtype=np.float64)
    for t in range(L_op):
        if np.isfinite(m_full[t]) and t != t0:
            lam_max[t] = np.exp(-m_full[t] * (t - t0))
    t_lo, t_hi = t_window
    sel = slice(t_lo, t_hi + 1)
    window_vals = m_full[sel]
    finite = np.isfinite(window_vals) & (window_vals > 0)
    if finite.sum() < 1:
        return GlueballM2Report(
            L=L_op, beta=beta, n_configs=n_cfg,
            smear_levels=tuple(int(s) for s in smear_levels), alpha=alpha,
            t0=t0, t_window=t_window,
            lambda_max_t=lam_max, m_0pp_t=m_full,
            m_0pp=None, m_0pp_error=None,
        )
    m_central = float(window_vals[finite].mean())
    m_jack = np.zeros(n_cfg, dtype=np.float64)
    for j in range(n_cfg):
        mask = np.ones(n_cfg, dtype=bool)
        mask[j] = False
        m_t_j = _m_per_t(O_per_smear_per_cfg[mask], t0)
        win_j = m_t_j[sel]
        finite_j = np.isfinite(win_j) & (win_j > 0)
        if finite_j.sum() < 1:
            m_jack[j] = np.nan
        else:
            m_jack[j] = float(win_j[finite_j].mean())
    finite_mask = np.isfinite(m_jack)
    if finite_mask.sum() >= 2:
        m_mean = m_jack[finite_mask].mean()
        diff = m_jack[finite_mask] - m_mean
        err = float(np.sqrt(((n_cfg - 1) / n_cfg) * (diff ** 2).sum()))
    else:
        err = None
    return GlueballM2Report(
        L=L_op, beta=beta, n_configs=n_cfg,
        smear_levels=tuple(int(s) for s in smear_levels), alpha=alpha,
        t0=t0, t_window=t_window,
        lambda_max_t=lam_max, m_0pp_t=m_full,
        m_0pp=m_central, m_0pp_error=err,
    )


__all__ = [
    "ape_smear_links",
    "build_zero_momentum_plaquette_op",
    "build_operator_basis",
    "build_correlator_matrix",
    "gevp_solve",
    "extract_m_0pp",
    "GlueballM2Report",
    "project_su3",
]
