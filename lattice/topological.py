"""
Topological charge computation for lattice gauge theory.

The topological charge Q measures the "instanton number" of a configuration:

    Q = (1/32π²) ∫ d⁴x Tr[F_μν F̃_μν]  (continuum)
    
    Q = (1/32π²) Σ_x ε_μνρσ Tr[F_μν(x) F_ρσ(x)]  (lattice)

For smooth configurations, Q is quantized: Q ∈ ℤ.
The integer part is the discrete component r of the Davis-Wilson map.

Accurate measurement requires:
1. Improved discretization (clover or 5Li)
2. UV smoothing (APE/stout smearing)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import jit, prange

from .gauge_config import GaugeConfig
from .wilson_loops import compute_plaquette


@jit(nopython=True, cache=True)
def compute_clover_leaf(
    U: NDArray[np.complex128],
    t: int, x: int, y: int, z: int,
    mu: int, nu: int
) -> NDArray[np.complex128]:
    """
    Compute the "clover leaf" combination of four plaquettes.
    
    C_μν(x) = P_μν(x) + P_μν(x-μ̂) + P_μν(x-ν̂) + P_μν(x-μ̂-ν̂)
    
    This is used to construct the clover field strength tensor.
    
    Args:
        U: Gauge configuration
        t, x, y, z: Site coordinates
        mu, nu: Plane directions
    
    Returns:
        Sum of four plaquettes (3×3 complex matrix)
    """
    L = int(U.shape[1])
    
    # Four corners for the clover
    # Site x
    P1 = compute_plaquette(U, t, x, y, z, mu, nu)
    
    # Site x - μ̂: initialize then apply shift
    t_m = t
    x_m = x
    y_m = y
    z_m = z
    if mu == 0:
        t_m = (t - 1) % L
    elif mu == 1:
        x_m = (x - 1) % L
    elif mu == 2:
        y_m = (y - 1) % L
    else:  # mu == 3
        z_m = (z - 1) % L
    P2 = compute_plaquette(U, t_m, x_m, y_m, z_m, mu, nu)
    
    # Site x - ν̂: initialize then apply shift
    t_n = t
    x_n = x
    y_n = y
    z_n = z
    if nu == 0:
        t_n = (t - 1) % L
    elif nu == 1:
        x_n = (x - 1) % L
    elif nu == 2:
        y_n = (y - 1) % L
    else:  # nu == 3
        z_n = (z - 1) % L
    P3 = compute_plaquette(U, t_n, x_n, y_n, z_n, mu, nu)
    
    # Site x - μ̂ - ν̂: start with x-μ̂, then apply ν shift
    t_mn = t_m
    x_mn = x_m
    y_mn = y_m
    z_mn = z_m
    if nu == 0:
        t_mn = (t_mn - 1) % L
    elif nu == 1:
        x_mn = (x_mn - 1) % L
    elif nu == 2:
        y_mn = (y_mn - 1) % L
    else:  # nu == 3
        z_mn = (z_mn - 1) % L
    P4 = compute_plaquette(U, t_mn, x_mn, y_mn, z_mn, mu, nu)
    
    return P1 + P2 + P3 + P4


@jit(nopython=True, cache=True)
def compute_field_strength_clover(
    U: NDArray[np.complex128],
    t: int, x: int, y: int, z: int,
    mu: int, nu: int
) -> NDArray[np.complex128]:
    """
    Compute the clover discretization of the field strength F_μν(x).
    
    F_μν(x) = (1/8i) [C_μν(x) - C_μν(x)†]
    
    where C_μν is the clover leaf.
    
    This is traceless and anti-Hermitian (Lie algebra element).
    
    Args:
        U: Gauge configuration
        t, x, y, z: Site coordinates
        mu, nu: Plane directions (mu < nu)
    
    Returns:
        Field strength tensor (3×3 anti-Hermitian traceless matrix)
    """
    clover = compute_clover_leaf(U, t, x, y, z, mu, nu)
    
    # Anti-Hermitian part: (C - C†) / 2i = (C - C†) * (-i/2)
    F = (clover - clover.conj().T) / 8.0j
    
    # Make traceless (should already be approximately traceless)
    trace_F = (F[0, 0] + F[1, 1] + F[2, 2]) / 3.0
    F[0, 0] -= trace_F
    F[1, 1] -= trace_F
    F[2, 2] -= trace_F
    
    return F


@jit(nopython=True, cache=True)
def compute_topological_charge_density(U: NDArray[np.complex128]) -> NDArray[np.float64]:
    """
    Compute the topological charge density q(x) at each site.
    
    q(x) = (1/32π²) ε_μνρσ Tr[F_μν(x) F_ρσ(x)]
    
    The sum Σ_x q(x) gives the total topological charge Q.
    
    Args:
        U: Gauge configuration
    
    Returns:
        Array of shape (L, L, L, L) with charge density at each site
    """
    L = int(U.shape[1])
    q = np.zeros((L, L, L, L), dtype=np.float64)
    
    # Normalization factor
    norm = 1.0 / (32.0 * np.pi ** 2)
    
    for t in range(L):  # Changed from prange
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    # Compute all six field strength components
                    F01 = compute_field_strength_clover(U, t, x, y, z, 0, 1)
                    F02 = compute_field_strength_clover(U, t, x, y, z, 0, 2)
                    F03 = compute_field_strength_clover(U, t, x, y, z, 0, 3)
                    F12 = compute_field_strength_clover(U, t, x, y, z, 1, 2)
                    F13 = compute_field_strength_clover(U, t, x, y, z, 1, 3)
                    F23 = compute_field_strength_clover(U, t, x, y, z, 2, 3)
                    
                    # ε_μνρσ Tr[F_μν F_ρσ] = 8 Tr[F01 F23 + F02 F31 + F03 F12]
                    # Note: F31 = -F13
                    term1 = np.trace(np.dot(F01, F23))
                    term2 = np.trace(np.dot(F02, -F13))  # F31 = -F13
                    term3 = np.trace(np.dot(F03, F12))
                    
                    q[t, x, y, z] = norm * 8.0 * (term1 + term2 + term3).real
    
    return q


def compute_topological_charge(config: GaugeConfig, method: str = "clover") -> float:
    """
    Compute the total topological charge Q of a configuration.
    
    Args:
        config: Gauge configuration
        method: "clover" (default) or "plaquette"
    
    Returns:
        Topological charge (float, should be near-integer for smooth configs)
    
    Note: For best results, apply smearing before measuring.
    """
    if method == "clover":
        q = compute_topological_charge_density(config.U)
        return float(np.sum(q))
    elif method == "plaquette":
        # Simple plaquette-based definition (less accurate)
        # COPILOT: Implement plaquette-based Q if needed
        raise NotImplementedError("Plaquette method not implemented")
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Smearing (for cleaner topological charge signal)
# ============================================================================

@jit(nopython=True, cache=True)
def _compute_staple(
    U: NDArray[np.complex128],
    t: int, x: int, y: int, z: int,
    mu: int
) -> NDArray[np.complex128]:
    """
    Compute the staple sum for link U_μ(x).
    
    Staple = Σ_{ν≠μ} [ U_ν(x) U_μ(x+ν̂) U_ν†(x+μ̂) + U_ν†(x-ν̂) U_μ(x-ν̂) U_ν(x+μ̂-ν̂) ]
    
    This is the sum over the 6 "staples" that share link U_μ(x).
    """
    L = int(U.shape[1])
    staple = np.zeros((3, 3), dtype=np.complex128)
    
    coords = [t, x, y, z]
    
    for nu in range(4):
        if nu == mu:
            continue
        
        # Forward staple: U_ν(x) U_μ(x+ν̂) U_ν†(x+μ̂)
        coords_p_nu = coords.copy()
        coords_p_nu[nu] = (coords_p_nu[nu] + 1) % L
        
        coords_p_mu = coords.copy()
        coords_p_mu[mu] = (coords_p_mu[mu] + 1) % L
        
        U_nu_x = U[nu, coords[0], coords[1], coords[2], coords[3]]
        U_mu_xpnu = U[mu, coords_p_nu[0], coords_p_nu[1], coords_p_nu[2], coords_p_nu[3]]
        U_nu_xpmu = U[nu, coords_p_mu[0], coords_p_mu[1], coords_p_mu[2], coords_p_mu[3]]
        
        staple += np.dot(np.dot(U_nu_x, U_mu_xpnu), U_nu_xpmu.conj().T)
        
        # Backward staple: U_ν†(x-ν̂) U_μ(x-ν̂) U_ν(x+μ̂-ν̂)
        coords_m_nu = coords.copy()
        coords_m_nu[nu] = (coords_m_nu[nu] - 1) % L
        
        coords_pmu_mnu = coords.copy()
        coords_pmu_mnu[mu] = (coords_pmu_mnu[mu] + 1) % L
        coords_pmu_mnu[nu] = (coords_pmu_mnu[nu] - 1) % L
        
        U_nu_xmnu = U[nu, coords_m_nu[0], coords_m_nu[1], coords_m_nu[2], coords_m_nu[3]]
        U_mu_xmnu = U[mu, coords_m_nu[0], coords_m_nu[1], coords_m_nu[2], coords_m_nu[3]]
        U_nu_xpmumnu = U[nu, coords_pmu_mnu[0], coords_pmu_mnu[1], coords_pmu_mnu[2], coords_pmu_mnu[3]]
        
        staple += np.dot(np.dot(U_nu_xmnu.conj().T, U_mu_xmnu), U_nu_xpmumnu)
    
    return staple


def apply_smearing(
    config: GaugeConfig,
    n_steps: int = 10,
    rho: float = 0.1,
    method: str = "ape"
) -> GaugeConfig:
    """
    Apply UV smearing to smooth the configuration.
    
    Smearing suppresses UV fluctuations while preserving long-distance physics,
    making topological charge measurement more accurate.
    
    Args:
        config: Input configuration
        n_steps: Number of smearing iterations
        rho: Smearing parameter (step size)
        method: "ape" (APE smearing) or "stout" (stout smearing)
    
    Returns:
        Smeared configuration
    
    APE smearing:
        U'_μ(x) = Proj_SU3[ (1-α) U_μ(x) + (α/6) Σ staples ]
        where α = 6*rho
    """
    from .su3 import project_to_su3
    
    if n_steps == 0:
        return config
    
    U = config.U.copy()
    L = config.L
    alpha = 6 * rho
    
    for step in range(n_steps):
        U_new = np.zeros_like(U)
        
        for mu in range(4):
            for t in range(L):
                for x in range(L):
                    for y in range(L):
                        for z in range(L):
                            # Compute staple sum
                            staple = _compute_staple(U, t, x, y, z, mu)
                            
                            # APE update: (1-α) U + (α/6) staple
                            U_link = U[mu, t, x, y, z]
                            U_new[mu, t, x, y, z] = project_to_su3(
                                (1 - alpha) * U_link + (alpha / 6) * staple
                            )
        
        U = U_new
    
    return GaugeConfig(
        U=U,
        L=config.L,
        beta=config.beta,
        metadata={**config.metadata, "smearing_steps": n_steps, "smearing_rho": rho}
    )


# ============================================================================
# Wilson Flow (Gradient Flow) - Proper RG Scale Definition
# ============================================================================

def apply_wilson_flow(
    config: GaugeConfig,
    flow_time: float = 1.0,
    dt: float = 0.01,
) -> GaugeConfig:
    """
    Apply Wilson flow (gradient flow) to the configuration.
    
    Wilson flow is the PROPER way to define the Davis-Wilson map at a
    consistent RG scale. Unlike APE smearing, it has a well-defined
    continuum limit and the flow time t directly corresponds to a
    physical smoothing scale sqrt(8t).
    
    The flow equation is:
        dV_μ(x,t)/dt = -g₀² [∂S_W/∂V_μ(x,t)] V_μ(x,t)
    
    where S_W is the Wilson action and the derivative is taken in the
    Lie algebra direction.
    
    Args:
        config: Input configuration
        flow_time: Total flow time t (resolution ε ~ sqrt(8t))
        dt: Integration step size
    
    Returns:
        Flowed configuration V_t
    
    Reference:
        Lüscher, JHEP 08 (2010) 071 [arXiv:1006.4518]
    """
    from .su3 import project_to_su3, su3_exp
    
    U = config.U.copy()
    L = config.L
    
    n_steps = int(flow_time / dt)
    
    for step in range(n_steps):
        # Compute the flow force Z_μ(x) = -∂S/∂U · U†
        Z = _compute_flow_force(U, L)
        
        # Runge-Kutta integration (RK3 Lüscher scheme for better stability)
        # Step 1: W0 = exp(1/4 Z) U
        W0 = _flow_step(U, Z, 0.25, L)
        
        # Step 2: Compute Z at W0
        Z1 = _compute_flow_force(W0, L)
        
        # Step 3: W1 = exp(8/9 Z1 - 17/36 Z) W0
        W1 = np.zeros_like(U)
        for mu in range(4):
            for t in range(L):
                for x in range(L):
                    for y in range(L):
                        for z in range(L):
                            Z_combined = (8.0/9.0) * Z1[mu,t,x,y,z] - (17.0/36.0) * Z[mu,t,x,y,z]
                            W1[mu,t,x,y,z] = np.dot(
                                su3_exp(dt * Z_combined),
                                W0[mu,t,x,y,z]
                            )
        
        # Step 4: Compute Z at W1
        Z2 = _compute_flow_force(W1, L)
        
        # Step 5: U_new = exp(3/4 Z2 - 8/9 Z1 + 17/36 Z) W1
        for mu in range(4):
            for t in range(L):
                for x in range(L):
                    for y in range(L):
                        for z in range(L):
                            Z_final = (3.0/4.0) * Z2[mu,t,x,y,z] - (8.0/9.0) * Z1[mu,t,x,y,z] + (17.0/36.0) * Z[mu,t,x,y,z]
                            U[mu,t,x,y,z] = project_to_su3(np.dot(
                                su3_exp(dt * Z_final),
                                W1[mu,t,x,y,z]
                            ))
    
    return GaugeConfig(
        U=U,
        L=config.L,
        beta=config.beta,
        metadata={
            **config.metadata,
            "flow_time": flow_time,
            "flow_dt": dt,
            "flow_steps": n_steps,
            "resolution_scale": np.sqrt(8 * flow_time),
        }
    )


def _compute_flow_force(
    U: NDArray[np.complex128],
    L: int,
) -> NDArray[np.complex128]:
    """
    Compute the Wilson flow force Z_μ(x) = -∂S_W/∂U_μ · U_μ†
    
    For Wilson action:
        Z_μ(x) = Σ_{ν≠μ} Staple_μν(x) · U_μ(x)†
    
    projected to traceless anti-Hermitian (Lie algebra).
    """
    Z = np.zeros_like(U)
    
    for mu in range(4):
        for t in range(L):
            for x in range(L):
                for y in range(L):
                    for z in range(L):
                        staple = _compute_staple(U, t, x, y, z, mu)
                        U_link = U[mu, t, x, y, z]
                        
                        # Ω = staple · U†
                        Omega = np.dot(staple, U_link.conj().T)
                        
                        # Project to traceless anti-Hermitian: Z = (Ω - Ω†)/2 - Tr(...)/3
                        Z_raw = (Omega - Omega.conj().T) / 2.0
                        trace = (Z_raw[0,0] + Z_raw[1,1] + Z_raw[2,2]) / 3.0
                        Z[mu, t, x, y, z, 0, 0] = Z_raw[0, 0] - trace
                        Z[mu, t, x, y, z, 0, 1] = Z_raw[0, 1]
                        Z[mu, t, x, y, z, 0, 2] = Z_raw[0, 2]
                        Z[mu, t, x, y, z, 1, 0] = Z_raw[1, 0]
                        Z[mu, t, x, y, z, 1, 1] = Z_raw[1, 1] - trace
                        Z[mu, t, x, y, z, 1, 2] = Z_raw[1, 2]
                        Z[mu, t, x, y, z, 2, 0] = Z_raw[2, 0]
                        Z[mu, t, x, y, z, 2, 1] = Z_raw[2, 1]
                        Z[mu, t, x, y, z, 2, 2] = Z_raw[2, 2] - trace
    
    return Z


def _flow_step(
    U: NDArray[np.complex128],
    Z: NDArray[np.complex128],
    eps: float,
    L: int,
) -> NDArray[np.complex128]:
    """Single flow step: W = exp(eps * Z) · U"""
    from .su3 import su3_exp, project_to_su3
    
    W = np.zeros_like(U)
    
    for mu in range(4):
        for t in range(L):
            for x in range(L):
                for y in range(L):
                    for z in range(L):
                        expZ = su3_exp(eps * Z[mu, t, x, y, z])
                        W[mu, t, x, y, z] = project_to_su3(
                            np.dot(expZ, U[mu, t, x, y, z])
                        )
    
    return W


def compute_flow_observable(
    config: GaugeConfig,
    flow_time: float,
    observable: str = "E",
) -> float:
    """
    Compute an observable on the flowed configuration.
    
    Args:
        config: Input configuration
        flow_time: Flow time t
        observable: "E" (energy density), "Q" (topological charge)
    
    Returns:
        Observable value at flow time t
    
    Common use: t²<E(t)> should plateau at ≈ 0.3 for t in scaling window.
    """
    flowed = apply_wilson_flow(config, flow_time=flow_time)
    
    if observable == "E":
        # Energy density from clover field strength
        q = compute_topological_charge_density(flowed.U)
        E = np.sum(q**2)  # Simplified; proper E uses Tr[F²]
        return flow_time**2 * E / flowed.L**4
    elif observable == "Q":
        return compute_topological_charge(flowed)
    else:
        raise ValueError(f"Unknown observable: {observable}")
