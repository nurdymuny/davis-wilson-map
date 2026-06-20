"""test_mass_dynamics.py — coupled (U, E, x, v) dynamics for the Halcyon
Falsification Battery (SPEC v2).

This is NOT an extension of inertia_damping/cuda/batched_leapfrog.py (the
Section 5 CUDA kernel). Per the SPEC v2 §9 implementation order, the existing
kernel is a pure (U, E) symplectic integrator with no hook for an external
test-mass DOF; this module is a new orchestrator that wraps the existing
buckyball_integrator gauge leapfrog and adds the coupled test-mass step.

The math, per SPEC v2 §3:
  - Gauge: standard buckyball SU(2) leapfrog (reused from buckyball_integrator).
  - Test mass: μ_Q(U) ẍ + c ẋ + K x = F(t)
    where μ_Q(U) = α_Halcyon · (1/E) Σ_e κ_Q(e) τ_Q²(e) |φ_n(e)|².
  - Coupling: μ_Q(U) is recomputed at every test-mass step from the current U.
  - τ_Q(e) = τ_0 / (1 + β_τ s_Q(e))  (v2 form; v1 √(1/staple) placeholder is rejected
    because it diverges at trivial vacuum).
  - Q_surrogate(t) = (P̄(t) - P_canonical) / P_canonical.

The lock-in protocol drives F(t) = F_0 · w_tukey(t) · cos(ω t) after a
pre-drive equilibration phase, and demodulates x(t) over the flat window.

Symplecticity note: the gauge half is symplectic (leapfrog on H_gauge). The
test-mass half is NOT (time-dependent driving and damping are dissipative).
This is acceptable: we are measuring a STEADY-STATE response, not conserving
H. Energy drift in the test mass is the expected dissipation, not a defect.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Import existing kernel helpers
from inertia_damping import buckyball_action as ba
from inertia_damping import buckyball_graph as bg
from inertia_damping import buckyball_integrator as bi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
P_CANONICAL_BETA_2_5 = 0.5072  # Migdal-Witten target at β=2.5

# Default SPEC v2 parameters
DEFAULT_TAU_0 = 1.0
DEFAULT_BETA_TAU = 2.0
DEFAULT_ALPHA_HALCYON = 1.0
DEFAULT_K = 1.0      # spring constant for test mass
DEFAULT_C = 0.1      # damping coefficient
DEFAULT_MU_PROXY = 1.0


# ---------------------------------------------------------------------------
# Eigenmode computation (cached at __init__)
# ---------------------------------------------------------------------------
def compute_edge_eigenmode_sq(graph, n_idx: int = 1) -> np.ndarray:
    """Compute |φ_n(e)|² for the n-th non-trivial graph-Laplacian eigenmode,
    projected onto edges via gradient.

    The graph Laplacian L_G = D Dᵀ lives on vertices (V × V). Its
    eigenvectors are vertex-valued φ_n(v). The natural projection onto edges
    is the gradient: φ_n(e) = φ_n(tail(e)) - φ_n(head(e)). We square and
    normalize so Σ_e |φ_n(e)|² = 1.

    Args:
        graph: BuckyballGraph instance (V=60, E=90)
        n_idx: index of the eigenmode (1 = lowest non-trivial; 0 would be
               the constant null vector)

    Returns:
        np.ndarray of shape (E,) with sum 1.
    """
    D = bi.signed_incidence(graph)  # (V, E)
    L_G = D @ D.T  # (V, V)
    L_sparse = csr_matrix(L_G)
    # Compute lowest n_idx+1 eigenvalues; sigma=0 with shift-invert for smallest
    evals, evecs = eigsh(L_sparse, k=n_idx + 1, sigma=0, which='LM')
    # Sort ascending
    sort_idx = np.argsort(evals)
    vertex_mode = evecs[:, sort_idx[n_idx]]  # (V,)
    # Project to edges via gradient: φ(e) = φ(tail) - φ(head)
    edges = graph.edges  # (E, 2)
    edge_gradient = vertex_mode[edges[:, 0]] - vertex_mode[edges[:, 1]]
    edge_sq = edge_gradient ** 2
    edge_sq = edge_sq / edge_sq.sum()
    return edge_sq


# ---------------------------------------------------------------------------
# Field observables (vectorized over edges)
# ---------------------------------------------------------------------------
def _build_edge_face_index_tensors(graph) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cache helper: return two (E,) int64 tensors of face indices per edge.

    Each edge belongs to exactly 2 faces (verified upstream by Euler check).
    """
    membership = ba._edge_face_membership(graph)
    n_edges = graph.n_edges
    f1 = torch.zeros(n_edges, dtype=torch.long)
    f2 = torch.zeros(n_edges, dtype=torch.long)
    for e in range(n_edges):
        f1[e] = membership[e][0][0]
        f2[e] = membership[e][1][0]
    return f1, f2


def compute_s_Q_per_edge(U: torch.Tensor, graph,
                         face_index_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                         ) -> torch.Tensor:
    """Normalized local Wilson-action density at each edge (vectorized).

    s_Q(e) = (1/4) * [(1 - q0(U_f1)) + (1 - q0(U_f2))]
    Range: s_Q(e) ∈ [0, 1].
    """
    Uf = ba.all_face_holonomies(U, graph)  # (F, 4)
    q0_per_face = Uf[:, 0]  # (F,)
    if face_index_cache is None:
        f1, f2 = _build_edge_face_index_tensors(graph)
    else:
        f1, f2 = face_index_cache
    q0_f1 = q0_per_face[f1]  # (E,)
    q0_f2 = q0_per_face[f2]  # (E,)
    return 0.25 * ((1.0 - q0_f1) + (1.0 - q0_f2))


def compute_kappa_Q_per_edge(U: torch.Tensor, graph,
                             face_index_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                             ) -> torch.Tensor:
    """Local curvature κ_Q(e) = (1 - q0(U_f1)) * (1 - q0(U_f2)), vectorized.

    Product (not sum) so κ_Q is distinguishable from s_Q in the μ_eff
    integral (prevents the two factors from collapsing).
    """
    Uf = ba.all_face_holonomies(U, graph)
    q0_per_face = Uf[:, 0]
    if face_index_cache is None:
        f1, f2 = _build_edge_face_index_tensors(graph)
    else:
        f1, f2 = face_index_cache
    q0_f1 = q0_per_face[f1]
    q0_f2 = q0_per_face[f2]
    return (1.0 - q0_f1) * (1.0 - q0_f2)


def compute_tau_Q_per_edge(s_Q: torch.Tensor, tau_0: float, beta_tau: float,
                           use_alternative: bool = False) -> torch.Tensor:
    """SPEC v2 τ_Q model.

    Default: τ_Q(e) = τ_0 / (1 + β_τ s_Q(e))
    Alternative (H9): τ_Q^alt(e) = τ_0 · exp(-β_τ s_Q(e))

    Limits verified per SPEC v2 §3:
    - s_Q=0 (trivial vacuum): τ_Q = τ_0 (uniform Newtonian limit)
    - s_Q→1: τ_Q → τ_0/(1+β_τ) (bounded)
    """
    if use_alternative:
        return tau_0 * torch.exp(-beta_tau * s_Q)
    else:
        return tau_0 / (1.0 + beta_tau * s_Q)


def compute_mu_eff(U: torch.Tensor, graph, phi_n_sq: torch.Tensor,
                   alpha_halcyon: float, tau_0: float, beta_tau: float,
                   mu_proxy: float = 1.0,
                   use_alternative_tau: bool = False,
                   face_index_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                   ) -> float:
    """μ_eff(Q) per SPEC v2 §3 Eq. (boxed)."""
    s_Q = compute_s_Q_per_edge(U, graph, face_index_cache=face_index_cache)
    kappa = compute_kappa_Q_per_edge(U, graph, face_index_cache=face_index_cache)
    tau = compute_tau_Q_per_edge(s_Q, tau_0, beta_tau, use_alternative=use_alternative_tau)
    integrand = kappa * tau ** 2 * phi_n_sq
    mu_eff = alpha_halcyon * float(integrand.sum()) / graph.n_edges
    return mu_proxy * mu_eff


def compute_Q_surrogate(U: torch.Tensor, graph,
                        P_canonical: float = P_CANONICAL_BETA_2_5) -> float:
    """SPEC v2 §3 Q_surrogate definition."""
    Uf = ba.all_face_holonomies(U, graph)
    P_bar = float(Uf[:, 0].mean())
    return (P_bar - P_canonical) / P_canonical


# ---------------------------------------------------------------------------
# Drive and demod
# ---------------------------------------------------------------------------
def tukey_window(t: float, t_total: float, alpha: float = 0.1) -> float:
    """Tukey window: cosine taper at each end, flat middle.

    alpha = fraction of total duration that is tapered (split between ends).
    """
    if t_total <= 0:
        return 0.0
    edge = alpha * t_total / 2.0
    if t < edge:
        return 0.5 * (1.0 + np.cos(np.pi * (t / edge - 1.0)))
    elif t > t_total - edge:
        return 0.5 * (1.0 + np.cos(np.pi * ((t - (t_total - edge)) / edge)))
    else:
        return 1.0


def drive_force(t: float, t_total: float, F_0: float, omega: float,
                tukey_alpha: float = 0.1) -> float:
    """F(t) = F_0 · w_tukey(t) · cos(ω t)."""
    return F_0 * tukey_window(t, t_total, tukey_alpha) * float(np.cos(omega * t))


def lockin_demodulate(x_history: np.ndarray, t_history: np.ndarray,
                      omega: float, F_0: float,
                      flat_start_frac: float = 0.05,
                      flat_end_frac: float = 0.95) -> Tuple[float, float, float, float]:
    """Extract χ_mag, χ_phase, X_I, X_Q from x(t) via cos/sin modulation."""
    N = len(x_history)
    i_start = int(N * flat_start_frac)
    i_end = int(N * flat_end_frac)
    x_flat = x_history[i_start:i_end]
    t_flat = t_history[i_start:i_end]
    X_I = 2.0 * float(np.mean(x_flat * np.cos(omega * t_flat)))
    X_Q = 2.0 * float(np.mean(x_flat * np.sin(omega * t_flat)))
    chi_mag = float(np.sqrt(X_I ** 2 + X_Q ** 2)) / F_0 if F_0 != 0 else 0.0
    # Per the demod derivation: for x(t) = |χ| F_0 cos(ωt + φ_χ),
    # <x cos(ωt)>_T = (|χ| F_0 / 2) cos(φ_χ)  →  X_I = |χ| F_0 cos(φ_χ)
    # <x sin(ωt)>_T = -(|χ| F_0 / 2) sin(φ_χ) →  X_Q = -|χ| F_0 sin(φ_χ)
    # So φ_χ = -atan2(X_Q, X_I)  (note the leading minus)
    chi_phase = float(-np.arctan2(X_Q, X_I))
    return chi_mag, chi_phase, X_I, X_Q


# ---------------------------------------------------------------------------
# Coupled dynamics
# ---------------------------------------------------------------------------
@dataclass
class TestMassConfig:
    """Configuration for a single TestMassDynamics run.

    Total test-mass inertia is mu_total = mu_baseline + mu_eff(U). The baseline
    is the Q-independent inertia (corresponds to the bare material mass on the
    bench); mu_eff is the Halcyon-coupling-induced shift. At Q=0 trivial vacuum
    mu_eff = 0 (by κ_Q construction) and the dynamics reduce exactly to a
    Newtonian driven damped oscillator with mass = mu_baseline.

    The H1 mu_proxy scales mu_eff only (the Q-dependent half), since the
    material-dependence test asks whether the SHIFT scales with material.
    """
    beta: float = 2.5
    tau_0: float = DEFAULT_TAU_0
    beta_tau: float = DEFAULT_BETA_TAU
    alpha_halcyon: float = DEFAULT_ALPHA_HALCYON
    mu_baseline: float = 1.0           # Q-independent inertia (Newtonian baseline)
    K_spring: float = DEFAULT_K
    c_damp: float = DEFAULT_C
    mu_proxy: float = DEFAULT_MU_PROXY  # scales mu_eff only, not mu_baseline
    mode_idx: int = 1
    use_alternative_tau: bool = False  # H9 test
    drive_omega: float = 1.0
    drive_F0: float = 0.01
    tukey_alpha: float = 0.1
    n_equil: int = 1000
    n_steps: int = 4000
    dt: float = 0.01
    freeze_gauge: bool = False    # if True, skip gauge leapfrog (debug only)


class TestMassDynamics:
    """Coupled (U, E, x, v) integrator per SPEC v2 §4."""

    def __init__(self, graph, config: TestMassConfig):
        self.graph = graph
        self.config = config
        # Precompute the eigenmode (one-time cost ~ms)
        phi_sq_np = compute_edge_eigenmode_sq(graph, n_idx=config.mode_idx)
        self.phi_n_sq = torch.from_numpy(phi_sq_np).to(dtype=torch.float64)
        # Precompute and cache the edge-face index tensors (vectorized lookup)
        self._face_index_cache = _build_edge_face_index_tensors(graph)
        # When freeze_gauge=True, cache the mu_eff value to avoid recomputing
        self._mu_eff_cached: Optional[float] = None

    def mu_eff(self, U: torch.Tensor) -> float:
        return compute_mu_eff(
            U, self.graph, self.phi_n_sq,
            alpha_halcyon=self.config.alpha_halcyon,
            tau_0=self.config.tau_0,
            beta_tau=self.config.beta_tau,
            mu_proxy=self.config.mu_proxy,
            use_alternative_tau=self.config.use_alternative_tau,
            face_index_cache=self._face_index_cache,
        )

    def Q_surrogate(self, U: torch.Tensor) -> float:
        return compute_Q_surrogate(U, self.graph)

    def coupled_leapfrog_step(self, U, E, x, v, dt, t, t_total, with_drive=True):
        """One coupled KDK step. Total inertia = mu_baseline + mu_eff(U)."""
        cfg = self.config
        if cfg.freeze_gauge:
            if self._mu_eff_cached is None:
                self._mu_eff_cached = self.mu_eff(U)
            mu = cfg.mu_baseline + self._mu_eff_cached
        else:
            mu = cfg.mu_baseline + self.mu_eff(U)

        # Test-mass kick 1
        F_t = drive_force(t, t_total, cfg.drive_F0, cfg.drive_omega,
                          cfg.tukey_alpha) if with_drive else 0.0
        a_t = (F_t - cfg.K_spring * x - cfg.c_damp * v) / mu
        v_half = v + 0.5 * dt * a_t

        # Gauge KDK (uses existing buckyball_integrator leapfrog)
        if cfg.freeze_gauge:
            U_new, E_new = U, E
        else:
            U_new, E_new = bi.leapfrog_step(U, E, dt, self.graph, cfg.beta)

        # Test-mass drift
        x_new = x + dt * v_half

        # μ at new U (skip recomputation if gauge frozen)
        if cfg.freeze_gauge:
            mu_new = mu
        else:
            mu_new = cfg.mu_baseline + self.mu_eff(U_new)

        # Test-mass kick 2
        F_t_dt = drive_force(t + dt, t_total, cfg.drive_F0, cfg.drive_omega,
                             cfg.tukey_alpha) if with_drive else 0.0
        a_t_dt = (F_t_dt - cfg.K_spring * x_new - cfg.c_damp * v_half) / mu_new
        v_new = v_half + 0.5 * dt * a_t_dt

        return U_new, E_new, x_new, v_new

    def evolve(self, U_init, E_init, x_init: float = 0.0, v_init: float = 0.0,
               with_drive: bool = True,
               record_every: int = 1) -> Dict[str, Any]:
        """Run full equilibration + driven trajectory."""
        cfg = self.config
        U, E = U_init.clone(), E_init.clone()
        x, v = float(x_init), float(v_init)
        t_total = cfg.n_steps * cfg.dt

        # Equilibrate (no drive)
        for _ in range(cfg.n_equil):
            U, E, x, v = self.coupled_leapfrog_step(
                U, E, x, v, cfg.dt, 0.0, t_total, with_drive=False,
            )

        # Cache once for frozen gauge (these observables don't change when U is frozen)
        if cfg.freeze_gauge:
            Q_cached = self.Q_surrogate(U)
            mu_cached = self.mu_eff(U)
        # Driven phase
        x_hist = []
        v_hist = []
        Q_hist = []
        mu_hist = []
        t_hist = []
        for s in range(cfg.n_steps):
            t = s * cfg.dt
            U, E, x, v = self.coupled_leapfrog_step(
                U, E, x, v, cfg.dt, t, t_total, with_drive=with_drive,
            )
            if s % record_every == 0:
                x_hist.append(x)
                v_hist.append(v)
                if cfg.freeze_gauge:
                    Q_hist.append(Q_cached)
                    mu_hist.append(mu_cached)
                else:
                    Q_hist.append(self.Q_surrogate(U))
                    mu_hist.append(self.mu_eff(U))
                t_hist.append(t + cfg.dt)

        return {
            "x_history": np.asarray(x_hist),
            "v_history": np.asarray(v_hist),
            "t_history": np.asarray(t_hist),
            "Q_surrogate_history": np.asarray(Q_hist),
            "mu_eff_history": np.asarray(mu_hist),
            "U_final": U,
            "E_final": E,
            "x_final": x,
            "v_final": v,
            "config": cfg,
        }


# ---------------------------------------------------------------------------
# Analytic reference (for unit tests)
# ---------------------------------------------------------------------------
def analytic_chi(omega: float, K: float, mu: float, c: float) -> complex:
    """Analytic χ(ω) = 1 / (K + i c ω - μ ω²) for a driven damped oscillator."""
    return 1.0 / (K + 1j * c * omega - mu * omega ** 2)


def newtonian_limit_check(graph) -> Dict[str, float]:
    """At Q=0 trivial vacuum (all U=identity), μ_eff should equal α_Halcyon ·
    (1/E) Σ |φ_n(e)|² · τ_0² · 0 = 0 (because κ_Q vanishes at trivial vacuum).

    This is by construction: at the trivial vacuum κ_Q(e) = (1-1)(1-1) = 0.
    Halcyon predicts NO inertia coupling at Q=0 — only at nonzero Q does
    α appear. So at Q=0 the test mass should behave as a free oscillator
    with μ → some baseline value. In the SPEC v2 model μ_eff(Q=0) = 0,
    meaning the test mass has no inertia from the Halcyon coupling alone.

    For the simulation we ADD a baseline μ_baseline so total μ = μ_baseline +
    μ_eff(Q). At Q=0, μ_eff = 0 so μ = μ_baseline = constant. The dynamics
    reduce to a standard driven damped oscillator.

    This function verifies that at Q=0 the system recovers Newtonian.
    """
    # Use a cold (identity) U; κ_Q = 0 by construction
    U_cold = ba.identity_links(graph.n_edges)
    phi_sq_np = compute_edge_eigenmode_sq(graph, n_idx=1)
    phi_sq = torch.from_numpy(phi_sq_np).to(dtype=torch.float64)
    mu_eff_cold = compute_mu_eff(
        U_cold, graph, phi_sq,
        alpha_halcyon=1.0, tau_0=1.0, beta_tau=2.0, mu_proxy=1.0,
    )
    return {
        "mu_eff_at_trivial_vacuum": mu_eff_cold,
        "kappa_sum_at_trivial_vacuum": 0.0,
        "expected_mu_eff": 0.0,
        "newtonian_limit_recovered": abs(mu_eff_cold) < 1e-12,
    }


__all__ = [
    "P_CANONICAL_BETA_2_5",
    "TestMassConfig",
    "TestMassDynamics",
    "compute_edge_eigenmode_sq",
    "compute_s_Q_per_edge",
    "compute_kappa_Q_per_edge",
    "compute_tau_Q_per_edge",
    "compute_mu_eff",
    "compute_Q_surrogate",
    "drive_force",
    "tukey_window",
    "lockin_demodulate",
    "analytic_chi",
    "newtonian_limit_check",
]
