"""su2_4d_glueball_M2.py - M2 channel of YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md.

Variational extraction of the 0++ scalar glueball mass on a 4D SU(2) lattice
via APE-smeared plaquette operators and the generalized eigenvalue problem
(GEVP).

Math
----
1. APE smearing (Albanese et al., Phys. Lett. B 192 (1987) 163):

       U_mu(x)' = Proj_{SU(2)} [ (1 - alpha) U_mu(x)
                                 + (alpha / 6) sum_{nu != mu, spatial} Staples ]

   where the spatial staples at link (x, mu) are the four U-shaped paths
   built only from spatial directions. The projection step keeps the smeared
   link in SU(2); for SU(2) we use quaternion normalization (the SU(2) group
   manifold is S^3 and projection is just division by the quaternion norm).
   No-op on the time direction's links - we smear only spatial links.

2. Variational basis: at each of N_smear levels we form the spatial-volume-
   averaged real-trace plaquette density on each time slice, giving N_smear
   different 0++ interpolating operators O_i(t).

3. GEVP (Luscher & Wolff, Nucl. Phys. B 339 (1990) 222):

       C(t) v_n = lambda_n(t, t0) C(t0) v_n,   lambda_n(t, t0) = exp(-E_n (t - t0))

   for the matrix C_ij(t) = <O_i(0) O_j(t)>_conn. Solve with scipy.linalg.eig
   in generalized form; the largest eigenvalue gives the ground state, so

       m_0++ = -log(lambda_max(t, t0)) / (t - t0).

   Average masses over a fit window in t (with t > t0) to suppress excited-
   state contamination, jackknife over configs for the error.

Status
------
M2 stub. Reads ensembles of spatial-link tensors (shape (3, L, L, L, L, 4),
i.e. 3 spatial dirs x L^4 sites x 4 quaternion components, matching
inertia_damping/su2_4d_heatbath_gpu.py conventions), runs APE smearing,
builds correlator matrix, returns m_0++ + jackknife error.

Production callers (push_ym4_glueball_bundle.py and friends) are NOT touched
in this stub; orchestrator wiring happens in a later round once the smoke
test confirms the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import linalg as scipy_linalg


RDTYPE = torch.float64
SPATIAL_DIRS = (1, 2, 3)  # mu = 0 is the time direction (Davis-Wilson convention)


# ----------------------------------------------------------------------
# Quaternion utilities (consistent with su2_4d_heatbath_gpu.py)
# ----------------------------------------------------------------------
def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ], dim=-1)


def qconj(a: torch.Tensor) -> torch.Tensor:
    out = a.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def qnorm(a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((a * a).sum(dim=-1)).clamp(min=1e-15)


def qproject_su2(a: torch.Tensor) -> torch.Tensor:
    """SU(2) projection: normalize the quaternion. SU(2) = unit quaternions."""
    return a / qnorm(a).unsqueeze(-1)


# ----------------------------------------------------------------------
# APE smearing for SU(2) (spatial-only)
# ----------------------------------------------------------------------
def _spatial_staple_sum(U_spatial: torch.Tensor, mu_idx: int) -> torch.Tensor:
    """Sum of the four spatial staples around each link in direction mu_idx.

    U_spatial has shape (3, L, L, L, L, 4) with axis 0 indexing SPATIAL_DIRS;
    note its data lives at lattice positions (t, x, y, z). The "mu" here is
    one of (1, 2, 3) = (x, y, z); the staple sum runs over the other two
    spatial directions only.
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
        fwd = qmul(qmul(Unu, Umu_at_nu), qconj(Unu_at_mu))
        # Backward staple: U_nu(x-nu)^dagger U_mu(x-nu) U_nu(x-nu+mu)
        Unu_back = torch.roll(Unu, +1, dims=nu)
        Umu_back = torch.roll(Umu, +1, dims=nu)
        Unu_back_at_mu = torch.roll(Unu_back, -1, dims=mu)
        bwd = qmul(qmul(qconj(Unu_back), Umu_back), Unu_back_at_mu)
        staple_total = staple_total + fwd + bwd
    return staple_total


def ape_smear_links(U_spatial: torch.Tensor,
                    alpha: float = 0.5,
                    n_smears: int = 0) -> torch.Tensor:
    """APE-smear the spatial links n_smears times with weight alpha.

    U_spatial: shape (3, L, L, L, L, 4) - the spatial subset of the full link
    field (only directions x, y, z; time links are not smeared in the M2
    operator construction).

    Returns a tensor of the same shape, still in SU(2) (each link is a unit
    quaternion).

    n_smears = 0 returns U_spatial unchanged.
    """
    if n_smears <= 0:
        return U_spatial.clone()
    U = U_spatial.clone()
    for _ in range(n_smears):
        U_new = torch.zeros_like(U)
        for mu_idx in range(3):
            staple = _spatial_staple_sum(U, mu_idx)
            blend = (1.0 - alpha) * U[mu_idx] + (alpha / 6.0) * staple
            U_new[mu_idx] = qproject_su2(blend)
        U = U_new
    return U


# ----------------------------------------------------------------------
# Operator construction: 0++ plaquette on each time slice
# ----------------------------------------------------------------------
def build_zero_momentum_plaquette_op(U_spatial_smeared: torch.Tensor) -> torch.Tensor:
    """Operator O(t) = (1/V_spatial) sum_{x,y,z, (i,j)} Re Tr U_p(x,y,z,t) / N

    N = 2 for SU(2) (Re Tr = q0 for unit quaternions, then divide by 2 to
    match the standard normalization Tr 1/2 = 1; equivalently, q0 IS already
    the normalized real trace for SU(2)).

    Returns shape (L,) - one value per time slice.
    """
    L = U_spatial_smeared.shape[1]
    spatial_planes = [(0, 1), (0, 2), (1, 2)]  # indices into U_spatial axis 0
    out = torch.zeros(L, dtype=U_spatial_smeared.dtype, device=U_spatial_smeared.device)
    for mu_idx, nu_idx in spatial_planes:
        Um = U_spatial_smeared[mu_idx]
        Un = U_spatial_smeared[nu_idx]
        mu = SPATIAL_DIRS[mu_idx]
        nu = SPATIAL_DIRS[nu_idx]
        Un_shift = torch.roll(Un, -1, dims=mu)
        Um_shift = torch.roll(Um, -1, dims=nu)
        Uplaq = qmul(qmul(qmul(Um, Un_shift), qconj(Um_shift)), qconj(Un))
        out = out + Uplaq[..., 0].mean(dim=(1, 2, 3))  # average over spatial axes
    return out / 3.0


# ----------------------------------------------------------------------
# Variational basis assembly
# ----------------------------------------------------------------------
def build_operator_basis(
    U_per_cfg: Sequence[torch.Tensor],
    smear_levels: Sequence[int] = (0, 5, 10, 15),
    alpha: float = 0.5,
) -> np.ndarray:
    """For each config and each smear level, compute O(t) per time slice.

    U_per_cfg: list of full-link tensors with shape (4, L, L, L, L, 4); axis
    0 = direction, index 0 = time. We extract spatial links U[1:4] and smear
    them.

    Returns array shape (n_cfg, N_smear, L), float64.
    """
    n_cfg = len(U_per_cfg)
    if n_cfg == 0:
        raise ValueError("U_per_cfg is empty")
    L = U_per_cfg[0].shape[1]
    N_smear = len(smear_levels)
    out = np.zeros((n_cfg, N_smear, L), dtype=np.float64)
    for c, U_full in enumerate(U_per_cfg):
        U_spatial = U_full[1:4]  # shape (3, L, L, L, L, 4)
        for i, n_smr in enumerate(smear_levels):
            U_smr = ape_smear_links(U_spatial, alpha=alpha, n_smears=int(n_smr))
            op_t = build_zero_momentum_plaquette_op(U_smr)
            out[c, i, :] = op_t.detach().cpu().numpy().astype(np.float64)
    return out


# ----------------------------------------------------------------------
# Correlator matrix
# ----------------------------------------------------------------------
def build_correlator_matrix(O_per_smear_per_cfg: np.ndarray, t: int) -> np.ndarray:
    """C_ij(t) = <O_i(t') O_j(t' + t)>_conn, origin-averaged over t'.

    O_per_smear_per_cfg has shape (n_cfg, N_smear, L). Returns (N_smear, N_smear).

    Connected: subtract <O_i> <O_j> using the full ensemble means.
    """
    n_cfg, N_smear, L = O_per_smear_per_cfg.shape
    means = O_per_smear_per_cfg.mean(axis=(0, 2))  # (N_smear,)
    C = np.zeros((N_smear, N_smear), dtype=np.float64)
    O = O_per_smear_per_cfg  # alias
    O_shift = np.roll(O, -t, axis=2)
    # <O_i(t') O_j(t' + t)> averaged over configs and over t'
    # shape (n_cfg, N_smear, N_smear, L) -> mean over cfg and L
    # build via einsum to avoid huge intermediate
    for i in range(N_smear):
        for j in range(N_smear):
            prod = O[:, i, :] * O_shift[:, j, :]  # (n_cfg, L)
            C[i, j] = prod.mean() - means[i] * means[j]
    # symmetrize (true correlator matrix is symmetric for real operators)
    C = 0.5 * (C + C.T)
    return C


def gevp_solve(C_t: np.ndarray, C_t0: np.ndarray) -> np.ndarray:
    """Generalized eigenvalues lambda_n satisfying C(t) v = lambda C(t0) v.

    Returns the eigenvalues sorted in descending real part. The largest
    corresponds to the lightest state.

    Uses scipy.linalg.eig in generalized form. Both matrices are symmetrized
    to suppress noise-induced asymmetry.
    """
    C_t = 0.5 * (C_t + C_t.T)
    C_t0 = 0.5 * (C_t0 + C_t0.T)
    eigvals = scipy_linalg.eigvals(C_t, C_t0)
    # Discard imaginary noise and sort by real part descending
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
    lambda_max_t: np.ndarray            # shape (L,)
    m_0pp_t: np.ndarray                 # shape (L,) - per-t effective mass
    m_0pp: Optional[float]              # weighted mean over fit window
    m_0pp_error: Optional[float]        # jackknife error


def _m_per_t(O_per_smear_per_cfg: np.ndarray, t0: int) -> np.ndarray:
    """Per-t effective mass from GEVP: m(t) = -log(lambda_max(t, t0)) / (t - t0).

    Returns shape (L,) with NaN where lambda_max <= 0 or t == t0.
    """
    L = O_per_smear_per_cfg.shape[2]
    out = np.full(L, np.nan, dtype=np.float64)
    C_t0 = build_correlator_matrix(O_per_smear_per_cfg, t0)
    # Regularize C_t0 to keep the gen-eig problem well posed in noisy regimes:
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
        m = -np.log(lam) / dt
        out[t] = m
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
    """End-to-end: per-config-per-smear plaquette operators -> m_0++ + jackknife.

    O_per_smear_per_cfg shape (n_cfg, N_smear, L).
    t_window: inclusive (t_lo, t_hi) for the constant-fit window in t.
    t0: GEVP reference time slice.
    """
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
    # Jackknife over configs
    m_jack = np.zeros(n_cfg, dtype=np.float64)
    n_finite_jack = 0
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
            n_finite_jack += 1
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
    "qproject_su2",
]
