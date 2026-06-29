"""
buckyball_yangmills_exact.py - Exact 2D Yang-Mills partition function and mean
plaquette for SU(2) on the truncated icosahedron (a 2-complex cell decomposition
of S^2 with 32 plaquettes, Euler characteristic chi=2).

DERIVATION (Migdal '75 / Witten '91 conventions, Wilson character expansion)
--------------------------------------------------------------------------
Action (matching inertia_damping/buckyball_action.wilson_action and the
heatbath in inertia_damping/buckyball_heatbath):
    S_W = (beta/N) * sum_p [N - Re Tr U_p]
        = beta * sum_p [1 - q0(U_p)]              for SU(2), N=2.

The per-plaquette Boltzmann weight, dropping the constant e^{-beta} per
plaquette (which factors out of Z and so out of <P>), is
    w(U_p) = exp(beta * q0(U_p)) = exp((beta/2) Re Tr U_p).

For SU(2), parametrise U_p by its half-angle: eigenvalues exp(+/- i theta),
Re Tr U / 2 = q0 = cos theta.  The standard Wilson character expansion
(Migdal, see e.g. Drouffe-Zuber 1983 Phys.Rep. eq. 6.21) is

    exp(beta cos theta)
      = sum_{n >= 1}  (2 n / beta) * I_n(beta) * sin(n theta) / sin theta
      = sum_{j = 0, 1/2, 1, ...}  (2j+1) * c_j(beta) * chi_j(U)

with n = 2j+1, chi_j(U) = sin((2j+1) theta) / sin theta, and the plaquette
character coefficient
    c_j(beta) = (2 / beta) * I_{2j+1}(beta).

Migdal's formula for a closed orientable surface with Euler characteristic
chi and P plaquettes, after integrating out all link variables (each link
appears in exactly two plaquettes with opposite orientation, which is the
only place the global topology enters):
    Z_unnorm  =  sum_j  (2j+1)^chi  *  c_j(beta)^P
                                    ~~~~~~~~~~~~~~
                                    Wilson character power

For the buckyball: chi = 2, P = 32 (12 pentagons + 20 hexagons), N_p = 32.

MEAN PLAQUETTE
--------------
Differentiating log Z with respect to beta and using that each plaquette
contributes additively beta * q0(U_p) to the (negated) log weight gives
    (d/dbeta) log Z_unnorm  =  sum_p < q0(U_p) >  =  P * <P>,
so
    <P>  =  (1/P) * d(log Z_unnorm) / d beta
         =  < c_j'(beta) / c_j(beta) >_{p_j}
with p_j proportional to (2j+1)^chi * c_j(beta)^P.

c_j'(beta) / c_j(beta) for c_j = (2/beta) I_n(beta), n = 2j+1, simplifies to
    c_j'/c_j = -1/beta + I_n'(beta)/I_n(beta)
             = -1/beta + (n/beta + I_{n+1}/I_n)         [using I_n' = (n/beta)I_n + I_{n+1}]
             = (n-1)/beta + I_{n+1}/I_n
             = 2j/beta + I_{2j+2}(beta) / I_{2j+1}(beta).

TRUNCATION
----------
The series in j converges rapidly: at fixed beta and large j, log c_j(beta) ~
-(2j+1) log(2j+1) (Bessel large-order asymptotics), so log(p_j) falls off
super-exponentially in j with the buckyball's P = 32 amplifying the decay.
Truncating at j_max = 80 (n_max = 161) is far beyond machine epsilon for
beta in [0, 10]; we sanity-check the truncation in `verify_truncation`.

REFERENCE NUMERICS
------------------
beta = 2.5, SU(2), buckyball (chi=2, P=32):
    <P>_exact = 0.507195100404...      (this is the workflow-1 anchor)

Use exact_mean_plaquette_su2_buckyball() to recompute and
exact_mean_plaquette_su2_2d(chi, n_plaquettes, beta, ...) for any closed
orientable surface decomposition with arbitrary plaquette count.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import iv

# Truncated-icosahedron constants (mirror buckyball_graph.py)
BUCKYBALL_CHI = 2
BUCKYBALL_N_PLAQUETTES = 32  # 12 pentagons + 20 hexagons

# Default series truncation; large enough that the suppressed tail is below
# float64 epsilon for any beta <= 10 with P >= 1 (verified in tests).
DEFAULT_J_MAX = 80


def _bessel_table(beta: float, j_max: int = DEFAULT_J_MAX) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ns, I_n, I_{n+1}) for n = 1..(j_max+1) inclusive.

    ns:    integer array, shape (j_max+1,), values 1, 2, ..., j_max+1.
    I_n:   I_n(beta), shape (j_max+1,).
    I_np1: I_{n+1}(beta), shape (j_max+1,).
    """
    ns = np.arange(1, j_max + 2)
    I_n = iv(ns, float(beta))
    I_np1 = iv(ns + 1, float(beta))
    return ns, I_n, I_np1


def _log_weights_and_ratio(
    beta: float,
    chi: int,
    n_plaquettes: int,
    j_max: int = DEFAULT_J_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalised p_j weights and the c_j'/c_j ratios.

    p_j is the discrete distribution over irreps j = 0, 1/2, 1, ... with
    p_j ~ (2j+1)^chi * [(2/beta) I_{2j+1}(beta)]^P, truncated to j <= j_max.

    ratio_j = c_j'(beta)/c_j(beta) = (n-1)/beta + I_{n+1}/I_n,  n = 2j+1.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive; got {beta!r}")
    if n_plaquettes <= 0:
        raise ValueError(f"n_plaquettes must be positive; got {n_plaquettes!r}")

    ns, I_n, I_np1 = _bessel_table(beta, j_max)
    # dim(j) = n = 2j+1; c_j = (2/beta) I_n(beta)
    log_c = np.log(2.0 / beta) + np.log(I_n)
    # log p_j (unnormalised) = chi*log(n) + P*log(c_j)
    log_p = chi * np.log(ns) + n_plaquettes * log_c
    log_p -= log_p.max()
    p = np.exp(log_p)
    p /= p.sum()

    ratio = (ns - 1) / beta + I_np1 / I_n
    return p, ratio


def exact_mean_plaquette_su2_2d(
    beta: float,
    chi: int = BUCKYBALL_CHI,
    n_plaquettes: int = BUCKYBALL_N_PLAQUETTES,
    j_max: int = DEFAULT_J_MAX,
) -> float:
    """Exact <P> for 2D-YM SU(2) on a closed orientable surface with the given
    Euler characteristic and plaquette count.  Wilson action, our convention
    S = (beta/N) sum_p [N - Re Tr U_p].

    <P> := (1/N_p) sum_p < (1/N) Re Tr U_p > = < q0(U_p) > for SU(2).
    """
    p, ratio = _log_weights_and_ratio(beta, chi, n_plaquettes, j_max)
    return float((p * ratio).sum())


def exact_mean_plaquette_su2_buckyball(beta: float, j_max: int = DEFAULT_J_MAX) -> float:
    """Convenience wrapper for chi = 2, n_plaquettes = 32."""
    return exact_mean_plaquette_su2_2d(
        beta=beta,
        chi=BUCKYBALL_CHI,
        n_plaquettes=BUCKYBALL_N_PLAQUETTES,
        j_max=j_max,
    )


def verify_truncation(
    beta: float,
    chi: int = BUCKYBALL_CHI,
    n_plaquettes: int = BUCKYBALL_N_PLAQUETTES,
) -> dict:
    """Sanity check: compare <P> for j_max = 40, 80, 160 and report deltas."""
    vals = {}
    for jm in (40, 80, 160):
        vals[jm] = exact_mean_plaquette_su2_2d(beta, chi, n_plaquettes, j_max=jm)
    return {
        "values": vals,
        "delta_80_vs_160": abs(vals[80] - vals[160]),
        "delta_40_vs_80": abs(vals[40] - vals[80]),
    }


__all__ = [
    "BUCKYBALL_CHI",
    "BUCKYBALL_N_PLAQUETTES",
    "DEFAULT_J_MAX",
    "exact_mean_plaquette_su2_2d",
    "exact_mean_plaquette_su2_buckyball",
    "verify_truncation",
]


if __name__ == "__main__":
    beta = 2.5
    P = exact_mean_plaquette_su2_buckyball(beta)
    chk = verify_truncation(beta)
    print(f"SU(2) 2D-YM, buckyball (chi=2, N_p=32), beta={beta}")
    print(f"  <P>_exact = {P:.10f}")
    print(f"  truncation check: j_max=40 -> {chk['values'][40]:.12f}")
    print(f"                    j_max=80 -> {chk['values'][80]:.12f}")
    print(f"                    j_max=160-> {chk['values'][160]:.12f}")
    print(f"  |80 - 160| = {chk['delta_80_vs_160']:.2e}")
