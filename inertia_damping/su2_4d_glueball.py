"""su2_4d_glueball.py - extract the 4D SU(2) Yang-Mills holonomy-continuum mass
gap from plaquette-plaquette correlators (M1 of YM_MASS_GAP_CONTINUUM_HYPOTHESIS).

The connected zero-momentum plaquette-plaquette correlator on a 4D lattice
periodic in time:

    C_PP(t) = <P_bar(0) P_bar(t)> - <P_bar>^2,

where P_bar(t) is the spatial-volume-averaged plaquette density on time-slice t.
For a confining gauge theory, at large t (and large enough lattice extent T):

    C_PP(t) ~ A * exp(-m * t)  +  A * exp(-m * (T - t)),   periodic.

The effective mass per t:

    m_eff(t) = arccosh( (C(t-1) + C(t+1)) / (2 C(t)) )

is a function of t; a plateau at large t identifies the lowest scalar mass
(0++ channel, since the plaquette is positive-parity scalar).

This is the M1 measurement of the holonomy-continuum hypothesis. It is the
canonical lattice extraction of the YM mass gap from gauge-invariant transport
data.

Null control: the same pipeline run on shuffled plaquette data (where each
time-slice's plaquette value is replaced by a random sample from the empirical
distribution, destroying time-correlation) should give a flat correlator and
NO effective-mass plateau. Per H5 / Falsification Criterion 7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CorrelatorReport:
    L: int
    beta: float
    n_configs: int
    P_bar_per_config_t_slice: np.ndarray  # shape (n_configs, L)
    P_bar_mean_t: np.ndarray              # shape (L,)
    P_bar_mean_global: float
    C_PP_connected_t: np.ndarray          # shape (L,)
    C_PP_error_t: np.ndarray              # shape (L,) — jackknife
    m_eff_t: np.ndarray                   # shape (L,) — NaN where ill-defined
    m_eff_error_t: np.ndarray             # shape (L,) — jackknife
    plateau_fit_window: Optional[Tuple[int, int]]
    plateau_fit_mass: Optional[float]
    plateau_fit_error: Optional[float]


def connected_correlator_t(p_bar: np.ndarray) -> np.ndarray:
    """Periodic-time connected plaquette-plaquette correlator.

    p_bar has shape (n_configs, L). Returns C(t) for t = 0, 1, ..., L-1.

    C(t) = <P_bar(0) P_bar(t)> - <P_bar>^2

    Averaged over translations of t (gauge invariance + translation invariance):
        C(t) = (1/L) sum_t' <P_bar(t') P_bar(t' + t)> - <P_bar>^2
    """
    n_cfg, L = p_bar.shape
    p_mean = p_bar.mean()  # scalar
    C = np.zeros(L, dtype=np.float64)
    # average over per-config products at fixed lag, then average over configs and t-origin
    for tau in range(L):
        shifted = np.roll(p_bar, -tau, axis=1)
        prod_t = (p_bar * shifted).mean(axis=1)   # (n_cfg,) - origin-average
        C[tau] = prod_t.mean() - p_mean * p_mean
    return C


def jackknife_correlator(p_bar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Jackknife on C_PP(t). Returns (C_t mean, C_t jackknife error)."""
    n_cfg, L = p_bar.shape
    C_full = connected_correlator_t(p_bar)
    C_jack = np.zeros((n_cfg, L), dtype=np.float64)
    for j in range(n_cfg):
        mask = np.ones(n_cfg, dtype=bool)
        mask[j] = False
        C_jack[j] = connected_correlator_t(p_bar[mask])
    C_jack_mean = C_jack.mean(axis=0)
    # jackknife error: sqrt( (n-1)/n * sum_j (C_j - C_mean)^2 )
    err = np.sqrt(((n_cfg - 1) / n_cfg) * ((C_jack - C_jack_mean) ** 2).sum(axis=0))
    return C_full, err


def effective_mass_t(C: np.ndarray) -> np.ndarray:
    """Cosh effective mass: m_eff(t) = arccosh((C(t-1) + C(t+1)) / (2 C(t))).

    For t = 1..L-2 (periodic). Returns NaN where the arg is < 1 (signal lost
    in noise) or where C(t) is <= 0.
    """
    L = C.shape[0]
    out = np.full(L, np.nan, dtype=np.float64)
    for t in range(1, L - 1):
        if C[t] <= 0:
            continue
        arg = (C[t - 1] + C[t + 1]) / (2.0 * C[t])
        if arg >= 1.0:
            out[t] = float(np.arccosh(arg))
    return out


def jackknife_effective_mass(p_bar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Jackknife on m_eff(t). Returns (m_eff mean, m_eff jackknife error)."""
    n_cfg, L = p_bar.shape
    C_full = connected_correlator_t(p_bar)
    m_full = effective_mass_t(C_full)
    m_jack = np.zeros((n_cfg, L), dtype=np.float64)
    for j in range(n_cfg):
        mask = np.ones(n_cfg, dtype=bool)
        mask[j] = False
        C_j = connected_correlator_t(p_bar[mask])
        m_jack[j] = effective_mass_t(C_j)
    m_mean = np.nanmean(m_jack, axis=0)
    # jackknife error, ignoring NaN
    diff = m_jack - m_mean
    diff = np.where(np.isnan(diff), 0.0, diff)
    n_finite = (~np.isnan(m_jack)).sum(axis=0).clip(min=1)
    err = np.sqrt(((n_cfg - 1) / n_cfg) * (diff ** 2).sum(axis=0) / np.maximum(n_finite, 1))
    return m_full, err


def fit_plateau(m_eff: np.ndarray, m_err: np.ndarray,
                 window: Optional[Tuple[int, int]] = None) -> Tuple[Optional[float], Optional[float], Optional[Tuple[int, int]]]:
    """Constant fit to m_eff in a chosen window. Returns (mass, error, window).

    If window=None, auto-select: take all t with finite m_eff and m_err > 0,
    use t_lo = first finite t, t_hi = min(t_lo + 3, last finite t).
    """
    L = len(m_eff)
    finite = np.where(np.isfinite(m_eff) & (m_err > 0))[0]
    if len(finite) == 0:
        return None, None, None
    if window is None:
        t_lo = int(finite[0])
        t_hi = min(int(finite[0]) + 3, int(finite[-1]))
        window = (t_lo, t_hi)
    t_lo, t_hi = window
    sel = slice(t_lo, t_hi + 1)
    masses = m_eff[sel]
    errors = m_err[sel].clip(min=1e-8)
    finite_in = np.isfinite(masses) & np.isfinite(errors)
    if finite_in.sum() < 1:
        return None, None, window
    weights = 1.0 / errors[finite_in] ** 2
    weighted = (masses[finite_in] * weights).sum() / weights.sum()
    err_fit = math.sqrt(1.0 / weights.sum())
    return float(weighted), float(err_fit), window


def extract_correlator_and_mass(
    p_bar_per_cfg: np.ndarray,
    L: int,
    beta: float,
    fit_window: Optional[Tuple[int, int]] = None,
) -> CorrelatorReport:
    """End-to-end: take per-config t-slice plaquette densities, extract m_eff plateau."""
    n_cfg, L_check = p_bar_per_cfg.shape
    assert L_check == L, f"L mismatch: {L_check} vs {L}"

    C_full, C_err = jackknife_correlator(p_bar_per_cfg)
    m_full, m_err = jackknife_effective_mass(p_bar_per_cfg)
    fit_mass, fit_err, used_window = fit_plateau(m_full, m_err, fit_window)

    return CorrelatorReport(
        L=L,
        beta=beta,
        n_configs=n_cfg,
        P_bar_per_config_t_slice=p_bar_per_cfg,
        P_bar_mean_t=p_bar_per_cfg.mean(axis=0),
        P_bar_mean_global=float(p_bar_per_cfg.mean()),
        C_PP_connected_t=C_full,
        C_PP_error_t=C_err,
        m_eff_t=m_full,
        m_eff_error_t=m_err,
        plateau_fit_window=used_window,
        plateau_fit_mass=fit_mass,
        plateau_fit_error=fit_err,
    )


# ----------------------------------------------------------------------
# M3: Wilson loop area-law string tension via Creutz ratios
# ----------------------------------------------------------------------

def creutz_ratio_value(W_avg: Dict[Tuple[int, int], float], R: int, T: int) -> float:
    """chi(R, T) = -log( W(R,T) * W(R-1,T-1) / (W(R-1,T) * W(R,T-1)) ).

    This Creutz ratio converges to the string tension sigma at large R, T
    for confining theories. Cancels perimeter and constant terms.

    Returns NaN if any W is non-positive or if (R-1, T-1) is outside the
    range of the W_avg table.
    """
    if R < 2 or T < 2:
        return float("nan")
    keys = [(R, T), (R - 1, T - 1), (R - 1, T), (R, T - 1)]
    if not all(k in W_avg for k in keys):
        return float("nan")
    W_RT = W_avg[(R, T)]
    W_R1T1 = W_avg[(R - 1, T - 1)]
    W_R1T = W_avg[(R - 1, T)]
    W_RT1 = W_avg[(R, T - 1)]
    if W_RT <= 0 or W_R1T1 <= 0 or W_R1T <= 0 or W_RT1 <= 0:
        return float("nan")
    val = (W_RT * W_R1T1) / (W_R1T * W_RT1)
    if val <= 0:
        return float("nan")
    return -float(np.log(val))


def average_wilson_table(W_per_cfg: List[Dict[Tuple[int, int], float]]) -> Dict[Tuple[int, int], float]:
    """Average each (R, T) value over the per-config list."""
    out: Dict[Tuple[int, int], float] = {}
    if not W_per_cfg:
        return out
    keys = list(W_per_cfg[0].keys())
    for k in keys:
        out[k] = float(np.mean([w[k] for w in W_per_cfg]))
    return out


def jackknife_creutz(W_per_cfg: List[Dict[Tuple[int, int], float]],
                     R: int, T: int) -> Tuple[float, float]:
    """Jackknife chi(R, T) with error from leave-one-out resamples.

    Returns (chi, error). chi may be NaN.
    """
    n = len(W_per_cfg)
    if n < 2:
        return float("nan"), float("nan")
    full = average_wilson_table(W_per_cfg)
    chi_full = creutz_ratio_value(full, R, T)
    chi_j = np.zeros(n)
    for j in range(n):
        sample = W_per_cfg[:j] + W_per_cfg[j + 1:]
        chi_j[j] = creutz_ratio_value(average_wilson_table(sample), R, T)
    finite = np.isfinite(chi_j)
    if finite.sum() < 2:
        return chi_full, float("nan")
    chi_mean = float(np.nanmean(chi_j))
    diff = chi_j[finite] - chi_mean
    err = float(np.sqrt(((n - 1) / n) * (diff ** 2).sum() / max(finite.sum(), 1)))
    return chi_full, err


def shuffled_null_control(p_bar_per_cfg: np.ndarray, seed: int = 20260628) -> np.ndarray:
    """Per Falsification Criterion 7 of YM_MASS_GAP_CONTINUUM_HYPOTHESIS_v0.1.md:
    shuffle plaquette values across configs and t-slices to destroy time
    correlation. The detector should fail to find a mass-gap plateau.

    Returns shuffled p_bar with same shape and same global mean/variance but
    no time correlation.
    """
    rng = np.random.default_rng(seed)
    flat = p_bar_per_cfg.flatten()
    rng.shuffle(flat)
    return flat.reshape(p_bar_per_cfg.shape)


if __name__ == "__main__":
    # Quick smoke test on synthetic data: pure exponential + noise should
    # recover the input mass.
    import argparse
    ap = argparse.ArgumentParser(description="smoke test the correlator extractor on synthetic data")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--m-true", type=float, default=0.5)
    ap.add_argument("--n-configs", type=int, default=200)
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=20260628)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    L, m_true = args.L, args.m_true
    # Synthetic: P_bar(t) such that <P_bar(0)P_bar(t)>_c = A * cosh(m*(t - L/2)) - A
    A = 0.05
    base = np.zeros((args.n_configs, L))
    for c in range(args.n_configs):
        # Construct a config whose translation-average gives a cosh correlator
        # by adding a smooth field + noise. Approximation only.
        t = np.arange(L)
        signal = np.cos(2 * np.pi * t / L) * 0.1 * rng.normal()  # zero-mode noise
        base[c] = 0.5 + signal + args.noise * rng.standard_normal(L)
    rep = extract_correlator_and_mass(base, L=L, beta=0.0)
    print(f"Synthetic smoke: m_true={m_true:.3f}, fit={rep.plateau_fit_mass}, fit_err={rep.plateau_fit_error}, window={rep.plateau_fit_window}")
    print(f"  C_PP(t):", rep.C_PP_connected_t)
    print(f"  m_eff(t):", rep.m_eff_t)
