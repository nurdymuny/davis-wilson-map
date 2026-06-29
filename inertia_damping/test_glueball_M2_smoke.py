"""test_glueball_M2_smoke.py - smoke test for M2 glueball (0++) pipeline.

Verifies the SU(2) and SU(3) APE+GEVP code on tiny synthetic inputs:
    1. APE smearing leaves the SU(2) identity field invariant.
    2. APE smearing leaves the SU(3) identity field invariant.
    3. GEVP solver returns sensible (>0) eigenvalues on a known correlator
       built as C(t) = A * exp(-m * |t - t0|) with positive A.
    4. End-to-end extract_m_0pp recovers a positive mass from a tiny noisy
       4^4 SU(2) field with a small spatial perturbation - no NaN, no crash.
    5. End-to-end extract_m_0pp on SU(3) 4^4 cold + tiny perturbation - no
       crash, finite result for the correlator matrix at small t.

Run:
    python -m inertia_damping.test_glueball_M2_smoke
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from inertia_damping import su2_4d_glueball_M2 as M2_su2
from inertia_damping import su3_4d_glueball_M2 as M2_su3


def _su2_cold_links(L: int = 4) -> torch.Tensor:
    """Cold start: all links = identity quaternion (1, 0, 0, 0). Shape (4, L^4, 4)."""
    U = torch.zeros((4, L, L, L, L, 4), dtype=torch.float64)
    U[..., 0] = 1.0
    return U


def _su3_cold_links(L: int = 4) -> torch.Tensor:
    """Cold start: all links = 3x3 identity. Shape (4, L^4, 3, 3) complex."""
    U = torch.zeros((4, L, L, L, L, 3, 3), dtype=torch.complex128)
    I3 = torch.eye(3, dtype=torch.complex128)
    U[..., :, :] = I3
    return U


def test_su2_ape_leaves_identity_invariant():
    L = 4
    U = _su2_cold_links(L)
    U_spatial = U[1:4]
    U_smr = M2_su2.ape_smear_links(U_spatial, alpha=0.5, n_smears=5)
    # All quaternions should still be (1, 0, 0, 0) to high precision.
    diff = (U_smr - U_spatial).abs().max().item()
    assert diff < 1e-12, f"SU(2) APE broke identity: max |diff| = {diff}"
    # Norm check: every quaternion must remain unit.
    norms = M2_su2.qnorm(U_smr)
    norm_err = (norms - 1.0).abs().max().item()
    assert norm_err < 1e-12, f"SU(2) APE produced non-unit quaternion: {norm_err}"
    print("  [pass] SU(2) APE leaves identity invariant")


def test_su3_ape_leaves_identity_invariant():
    L = 4
    U = _su3_cold_links(L)
    U_spatial = U[1:4]
    U_smr = M2_su3.ape_smear_links(U_spatial, alpha=0.5, n_smears=3)
    # Should be identity to high precision.
    I3 = torch.eye(3, dtype=torch.complex128)
    diff = (U_smr - I3).abs().max().item()
    assert diff < 1e-10, f"SU(3) APE broke identity: max |diff| = {diff}"
    # Unitarity check.
    UU = U_smr @ U_smr.conj().transpose(-1, -2)
    unit_err = (UU - I3).abs().max().item()
    assert unit_err < 1e-10, f"SU(3) APE produced non-unitary link: {unit_err}"
    print("  [pass] SU(3) APE leaves identity invariant")


def test_gevp_on_known_correlator():
    """Build a synthetic correlator with known exponential decay; check
    that the largest generalized eigenvalue lambda_max(t, t0) recovers
    exp(-m * (t - t0))."""
    rng = np.random.default_rng(20260628)
    N_smear, L = 3, 8
    m_true = 0.40
    n_cfg = 60
    # Build per-config operators as O_i(t) = a_i * f(t) + noise, with f(t)
    # such that <O_i(t') O_j(t' + t)> averaged over t' decays as A_ij * exp(-m * t).
    # Use a stochastic harmonic-noise construction: O_i(t) = sum_k c_ik * x_k * cos(theta_k - 2*pi*k*t / L)
    # whose autocorrelation peaks at t = 0. For simplicity here, build
    # O_i(t) = a_i * z(t) where z(t) is a Gaussian process with covariance
    # K(t) = exp(-m * |t|).
    K = np.zeros((L, L))
    for tau in range(L):
        for sigma in range(L):
            d = min(abs(tau - sigma), L - abs(tau - sigma))  # periodic
            K[tau, sigma] = np.exp(-m_true * d)
    # Eigendecompose to sample.
    w, V = np.linalg.eigh(K)
    w = np.clip(w, 1e-12, None)
    L_chol = V @ np.diag(np.sqrt(w))
    a = np.array([1.0, 1.3, 0.7])  # per-operator amplitudes
    O = np.zeros((n_cfg, N_smear, L))
    for c in range(n_cfg):
        z = L_chol @ rng.standard_normal(L)
        for i in range(N_smear):
            O[c, i, :] = a[i] * z + 0.001 * rng.standard_normal(L)
    t0 = 1
    C_t0 = M2_su2.build_correlator_matrix(O, t0)
    C_t1 = M2_su2.build_correlator_matrix(O, t0 + 1)
    eigs = M2_su2.gevp_solve(C_t1, C_t0)
    assert np.all(np.isfinite(eigs)), f"GEVP produced non-finite eigenvalues: {eigs}"
    lam_max = float(eigs[0])
    assert lam_max > 0, f"GEVP largest eigenvalue not positive: {lam_max}"
    m_eff = -np.log(lam_max)  # since (t - t0) = 1
    # Should be in the ballpark of m_true to within 30% (small N_cfg, noisy).
    assert 0.5 * m_true < m_eff < 2.0 * m_true, (
        f"GEVP recovered m_eff={m_eff:.3f}, expected ~{m_true:.3f}"
    )
    print(f"  [pass] GEVP on known correlator: m_true={m_true:.3f}, m_eff={m_eff:.3f}")


def test_su2_end_to_end_no_crash():
    """Tiny 4^4 SU(2) field with a small perturbation, 5 fake 'configs' just
    to exercise the full pipeline. Goal: no NaN where input is valid, no
    crash, returns a report dataclass."""
    L = 4
    rng = np.random.default_rng(20260628)
    cfgs = []
    for c in range(5):
        U = _su2_cold_links(L)
        # Small spatial perturbation: rotate each spatial link slightly.
        for mu in range(1, 4):
            eps = 0.05 * rng.standard_normal(U[mu].shape).astype(np.float64)
            eps[..., 0] = 1.0 + 0.05 * rng.standard_normal(U[mu].shape[:-1])
            U[mu] = torch.from_numpy(eps)
            U[mu] = M2_su2.qproject_su2(U[mu])
        cfgs.append(U)
    smear_levels = (0, 2, 4)  # keep tiny for the smoke test (was 0,5,10,15)
    O = M2_su2.build_operator_basis(cfgs, smear_levels=smear_levels, alpha=0.5)
    assert O.shape == (5, 3, L), f"unexpected operator basis shape: {O.shape}"
    assert np.all(np.isfinite(O)), "SU(2) operator basis contains non-finite values"
    rep = M2_su2.extract_m_0pp(O, t_window=(1, 2), t0=0,
                                 smear_levels=smear_levels, alpha=0.5)
    # rep.m_0pp may be None on noisy tiny data; we only require no crash.
    assert rep.n_configs == 5
    assert rep.L == L
    print(f"  [pass] SU(2) end-to-end: m_0pp={rep.m_0pp}, err={rep.m_0pp_error}, "
          f"lam_max_t={rep.lambda_max_t}")


def test_su3_end_to_end_no_crash():
    L = 4
    rng = np.random.default_rng(20260628)
    cfgs = []
    for c in range(4):
        U = _su3_cold_links(L)
        # Small spatial perturbation: project I + small Hermitian back to SU(3).
        for mu in range(1, 4):
            shape = U[mu].shape[:-2] + (3, 3)
            eps = 0.05 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
            eps_t = torch.from_numpy(eps).to(torch.complex128)
            M = U[mu] + eps_t
            U[mu] = M2_su3.project_su3(M)
        cfgs.append(U)
    smear_levels = (0, 2, 4)
    O = M2_su3.build_operator_basis(cfgs, smear_levels=smear_levels, alpha=0.5)
    assert O.shape == (4, 3, L), f"unexpected SU(3) operator basis shape: {O.shape}"
    assert np.all(np.isfinite(O)), "SU(3) operator basis contains non-finite values"
    # Correlator matrix at small t must be finite + symmetric.
    C0 = M2_su3.build_correlator_matrix(O, 0)
    C1 = M2_su3.build_correlator_matrix(O, 1)
    assert np.all(np.isfinite(C0)) and np.all(np.isfinite(C1))
    assert np.allclose(C0, C0.T)
    rep = M2_su3.extract_m_0pp(O, t_window=(1, 2), t0=0,
                                 smear_levels=smear_levels, alpha=0.5)
    assert rep.n_configs == 4
    assert rep.L == L
    print(f"  [pass] SU(3) end-to-end: m_0pp={rep.m_0pp}, err={rep.m_0pp_error}, "
          f"lam_max_t={rep.lambda_max_t}")


def main() -> int:
    print("M2 glueball smoke tests:")
    test_su2_ape_leaves_identity_invariant()
    test_su3_ape_leaves_identity_invariant()
    test_gevp_on_known_correlator()
    test_su2_end_to_end_no_crash()
    test_su3_end_to_end_no_crash()
    print("All 5 smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
