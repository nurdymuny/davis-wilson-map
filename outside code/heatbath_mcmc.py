"""
SU(3) Heatbath MCMC for Lattice QCD
===================================

Cabibbo-Marinari algorithm: Update SU(3) by sequential SU(2) subgroup heatbath.

This is the missing piece - proper thermalization generates physically
correlated configurations instead of random noise.

Author: Bee Rosa Davis
Date: December 2025
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def su2_heatbath(staple: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Generate SU(2) matrix from heatbath distribution.
    
    Distribution: P(U) ∝ exp(β/2 * Re Tr(U * staple†))
    
    Uses Kennedy-Pendleton algorithm for efficient sampling.
    
    Args:
        staple: 2x2 complex matrix (the "environment")
        beta: Inverse coupling
        
    Returns:
        New SU(2) matrix from heatbath distribution
    """
    device = staple.device
    dtype = staple.dtype
    
    # Compute determinant and normalize
    # staple = k * V where V ∈ SU(2) and k = sqrt(det(staple))
    det = staple[0, 0] * staple[1, 1] - staple[0, 1] * staple[1, 0]
    k = torch.sqrt(torch.abs(det) + 1e-10)
    
    if k < 1e-10:
        # Degenerate case: return random SU(2)
        return random_su2(device, dtype)
    
    # Effective beta for this update
    a = beta * k.real
    
    # Kennedy-Pendleton algorithm for sampling
    # Sample x = cos(θ/2)² from P(x) ∝ sqrt(1-x) * exp(2*a*x)
    
    # Rejection sampling
    max_iter = 100
    for _ in range(max_iter):
        # Generate candidate
        r1 = torch.rand(1, device=device)
        r2 = torch.rand(1, device=device)
        r3 = torch.rand(1, device=device)
        r4 = torch.rand(1, device=device)
        
        # Compute x candidate using Creutz formula
        lambda_sq = -1.0 / (2.0 * a + 1e-10) * (torch.log(r1) + torch.log(r2) * torch.cos(2 * math.pi * r3) ** 2)
        
        if lambda_sq > 1.0:
            continue
            
        # Accept/reject
        if r4 ** 2 <= 1.0 - lambda_sq:
            x = 1.0 - 2.0 * lambda_sq
            break
    else:
        # Fallback: use mean-field approximation
        x = torch.tensor(0.5, device=device)
    
    # x = a₀² where U = a₀*I + i*σ·a
    # Generate random direction for a = (a₁, a₂, a₃) on sphere
    a0 = torch.sqrt(torch.clamp(x, 0, 1))
    
    # Random point on 2-sphere with radius sqrt(1 - a0²)
    r = torch.sqrt(torch.clamp(1.0 - a0 ** 2, 0, 1))
    phi = 2 * math.pi * torch.rand(1, device=device)
    costheta = 2 * torch.rand(1, device=device) - 1
    sintheta = torch.sqrt(1 - costheta ** 2)
    
    a1 = r * sintheta * torch.cos(phi)
    a2 = r * sintheta * torch.sin(phi)
    a3 = r * costheta
    
    # Construct SU(2) matrix: U = a₀*I + i*(a₁σ₁ + a₂σ₂ + a₃σ₃)
    # = [[a₀ + i*a₃, a₂ + i*a₁], [-a₂ + i*a₁, a₀ - i*a₃]]
    U_new = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    U_new[0, 0] = a0 + 1j * a3
    U_new[0, 1] = a2 + 1j * a1
    U_new[1, 0] = -a2 + 1j * a1
    U_new[1, 1] = a0 - 1j * a3
    
    # Multiply by V† to get final update
    # U_final = U_new * V†, where staple = k * V
    V = staple / (k + 1e-10)
    V_dag = V.conj().T
    
    return U_new @ V_dag


def random_su2(device, dtype=torch.complex64) -> torch.Tensor:
    """Generate random SU(2) matrix using Haar measure."""
    # Quaternion representation: uniform on 3-sphere
    x = torch.randn(4, device=device)
    x = x / torch.norm(x)
    
    a0, a1, a2, a3 = x[0], x[1], x[2], x[3]
    
    U = torch.zeros(2, 2, dtype=dtype, device=device)
    U[0, 0] = a0 + 1j * a3
    U[0, 1] = a2 + 1j * a1
    U[1, 0] = -a2 + 1j * a1
    U[1, 1] = a0 - 1j * a3
    
    return U


def embed_su2_in_su3(su2: torch.Tensor, subgroup: int) -> torch.Tensor:
    """
    Embed SU(2) matrix in SU(3).
    
    Three subgroups indexed by which row/column pair to use:
        0: rows/cols (0,1) - "12" subgroup
        1: rows/cols (0,2) - "13" subgroup  
        2: rows/cols (1,2) - "23" subgroup
    """
    device = su2.device
    dtype = torch.complex64
    
    su3 = torch.eye(3, dtype=dtype, device=device)
    
    if subgroup == 0:  # 12 subgroup
        su3[0, 0] = su2[0, 0]
        su3[0, 1] = su2[0, 1]
        su3[1, 0] = su2[1, 0]
        su3[1, 1] = su2[1, 1]
    elif subgroup == 1:  # 13 subgroup
        su3[0, 0] = su2[0, 0]
        su3[0, 2] = su2[0, 1]
        su3[2, 0] = su2[1, 0]
        su3[2, 2] = su2[1, 1]
    else:  # 23 subgroup
        su3[1, 1] = su2[0, 0]
        su3[1, 2] = su2[0, 1]
        su3[2, 1] = su2[1, 0]
        su3[2, 2] = su2[1, 1]
    
    return su3


def extract_su2_subgroup(su3: torch.Tensor, subgroup: int) -> torch.Tensor:
    """Extract SU(2) subgroup from SU(3) matrix."""
    if subgroup == 0:
        return su3[:2, :2].clone()
    elif subgroup == 1:
        su2 = torch.zeros(2, 2, dtype=su3.dtype, device=su3.device)
        su2[0, 0] = su3[0, 0]
        su2[0, 1] = su3[0, 2]
        su2[1, 0] = su3[2, 0]
        su2[1, 1] = su3[2, 2]
        return su2
    else:
        return su3[1:, 1:].clone()


def compute_staple(U: torch.Tensor, site: Tuple[int, ...], mu: int) -> torch.Tensor:
    """
    Compute the staple sum for link U_μ(x).
    
    Staple = sum over ν≠μ of:
        U_ν(x+μ) U_μ†(x+ν) U_ν†(x)           (forward)
      + U_ν†(x+μ-ν) U_μ†(x-ν) U_ν(x-ν)       (backward)
    
    Args:
        U: Gauge field [T, X, Y, Z, 4, 3, 3]
        site: Lattice site (t, x, y, z)
        mu: Direction of link being updated
        
    Returns:
        3x3 complex staple matrix
    """
    device = U.device
    dtype = U.dtype
    dims = U.shape[:4]
    
    staple = torch.zeros(3, 3, dtype=dtype, device=device)
    
    t, x, y, z = site
    
    for nu in range(4):
        if nu == mu:
            continue
        
        # Shift vectors
        shift_mu = [0, 0, 0, 0]
        shift_mu[mu] = 1
        
        shift_nu = [0, 0, 0, 0]
        shift_nu[nu] = 1
        
        # Forward staple: U_ν(x+μ) U_μ†(x+ν) U_ν†(x)
        site_plus_mu = tuple((s + shift_mu[i]) % dims[i] for i, s in enumerate(site))
        site_plus_nu = tuple((s + shift_nu[i]) % dims[i] for i, s in enumerate(site))
        
        U_nu_xpmu = U[site_plus_mu[0], site_plus_mu[1], site_plus_mu[2], site_plus_mu[3], nu]
        U_mu_xpnu = U[site_plus_nu[0], site_plus_nu[1], site_plus_nu[2], site_plus_nu[3], mu]
        U_nu_x = U[t, x, y, z, nu]
        
        staple += U_nu_xpmu @ U_mu_xpnu.conj().T @ U_nu_x.conj().T
        
        # Backward staple: U_ν†(x+μ-ν) U_μ†(x-ν) U_ν(x-ν)
        site_minus_nu = tuple((s - shift_nu[i]) % dims[i] for i, s in enumerate(site))
        site_plus_mu_minus_nu = tuple((s + shift_mu[i] - shift_nu[i]) % dims[i] for i, s in enumerate(site))
        
        U_nu_xpmumnu = U[site_plus_mu_minus_nu[0], site_plus_mu_minus_nu[1], 
                        site_plus_mu_minus_nu[2], site_plus_mu_minus_nu[3], nu]
        U_mu_xmnu = U[site_minus_nu[0], site_minus_nu[1], site_minus_nu[2], site_minus_nu[3], mu]
        U_nu_xmnu = U[site_minus_nu[0], site_minus_nu[1], site_minus_nu[2], site_minus_nu[3], nu]
        
        staple += U_nu_xpmumnu.conj().T @ U_mu_xmnu.conj().T @ U_nu_xmnu
    
    return staple


def cabibbo_marinari_update(U: torch.Tensor, site: Tuple[int, ...], mu: int, 
                            beta: float, n_hits: int = 1) -> torch.Tensor:
    """
    Cabibbo-Marinari SU(3) heatbath update for single link.
    
    Updates U_μ(x) by sequential SU(2) subgroup heatbath.
    
    Args:
        U: Gauge field [T, X, Y, Z, 4, 3, 3]
        site: Lattice site
        mu: Direction
        beta: Inverse coupling (β = 6/g²)
        n_hits: Number of SU(2) subgroup sweeps
        
    Returns:
        Updated link matrix
    """
    t, x, y, z = site
    link = U[t, x, y, z, mu].clone()
    staple = compute_staple(U, site, mu)
    
    # Effective staple for heatbath: W = U * Σ
    W = link @ staple
    
    for _ in range(n_hits):
        for subgroup in range(3):
            # Extract SU(2) subgroup of W
            W_su2 = extract_su2_subgroup(W, subgroup)
            
            # Heatbath for this subgroup
            su2_new = su2_heatbath(W_su2, beta / 3.0)  # β/3 for SU(3)
            
            # Embed back in SU(3)
            su3_update = embed_su2_in_su3(su2_new, subgroup)
            
            # Update link: U_new = su3_update * U_old
            link = su3_update @ link
            
            # Recompute W for next subgroup
            W = link @ staple
    
    # Project back to SU(3) to remove numerical drift
    link = project_to_su3(link)
    
    return link


def project_to_su3(M: torch.Tensor) -> torch.Tensor:
    """Project matrix to SU(3) using Gram-Schmidt + det normalization."""
    device = M.device
    
    # Gram-Schmidt orthonormalization
    u0 = M[:, 0]
    u0 = u0 / torch.norm(u0)
    
    u1 = M[:, 1] - torch.dot(u0.conj(), M[:, 1]) * u0
    u1 = u1 / torch.norm(u1)
    
    u2 = M[:, 2] - torch.dot(u0.conj(), M[:, 2]) * u0 - torch.dot(u1.conj(), M[:, 2]) * u1
    u2 = u2 / torch.norm(u2)
    
    U = torch.stack([u0, u1, u2], dim=1)
    
    # Fix determinant to 1
    det = torch.linalg.det(U)
    phase = torch.angle(det) / 3
    U = U * torch.exp(-1j * phase)
    
    return U


def heatbath_sweep(U: torch.Tensor, beta: float, n_hits: int = 1) -> torch.Tensor:
    """
    Perform one heatbath sweep over all links.
    
    Args:
        U: Gauge field [T, X, Y, Z, 4, 3, 3]
        beta: Inverse coupling
        n_hits: Number of SU(2) hits per link
        
    Returns:
        Updated gauge field
    """
    dims = U.shape[:4]
    U_new = U.clone()
    
    # Checkerboard ordering for parallelization potential
    for parity in [0, 1]:
        for t in range(dims[0]):
            for x in range(dims[1]):
                for y in range(dims[2]):
                    for z in range(dims[3]):
                        if (t + x + y + z) % 2 != parity:
                            continue
                        
                        site = (t, x, y, z)
                        for mu in range(4):
                            U_new[t, x, y, z, mu] = cabibbo_marinari_update(
                                U_new, site, mu, beta, n_hits
                            )
    
    return U_new


def compute_plaquette(U: torch.Tensor) -> torch.Tensor:
    """Compute average plaquette for monitoring thermalization."""
    dims = U.shape[:4]
    total = 0.0
    count = 0
    
    for t in range(dims[0]):
        for x in range(dims[1]):
            for y in range(dims[2]):
                for z in range(dims[3]):
                    site = (t, x, y, z)
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            # Plaquette P_μν(x) = U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x)
                            shift_mu = [0, 0, 0, 0]
                            shift_mu[mu] = 1
                            shift_nu = [0, 0, 0, 0]
                            shift_nu[nu] = 1
                            
                            site_plus_mu = tuple((s + shift_mu[i]) % dims[i] for i, s in enumerate(site))
                            site_plus_nu = tuple((s + shift_nu[i]) % dims[i] for i, s in enumerate(site))
                            
                            P = (U[t, x, y, z, mu] @ 
                                 U[site_plus_mu[0], site_plus_mu[1], site_plus_mu[2], site_plus_mu[3], nu] @
                                 U[site_plus_nu[0], site_plus_nu[1], site_plus_nu[2], site_plus_nu[3], mu].conj().T @
                                 U[t, x, y, z, nu].conj().T)
                            
                            total += P.trace().real / 3.0
                            count += 1
    
    return total / count


def thermalize(U: torch.Tensor, beta: float, n_therm: int = 100, 
               n_hits: int = 1, verbose: bool = True) -> torch.Tensor:
    """
    Thermalize gauge configuration.
    
    Args:
        U: Initial gauge field (can be random or cold start)
        beta: Inverse coupling
        n_therm: Number of thermalization sweeps
        n_hits: SU(2) hits per link per sweep
        verbose: Print progress
        
    Returns:
        Thermalized gauge field
    """
    if verbose:
        print(f"Thermalizing: β={beta}, sweeps={n_therm}")
        print("-" * 40)
    
    for i in range(n_therm):
        U = heatbath_sweep(U, beta, n_hits)
        
        if verbose and (i + 1) % 10 == 0:
            plaq = compute_plaquette(U)
            print(f"  Sweep {i+1:4d}: <P> = {plaq:.6f}")
    
    if verbose:
        print("-" * 40)
        print(f"Final plaquette: {compute_plaquette(U):.6f}")
    
    return U


def generate_ensemble(lattice_dims: Tuple[int, ...], beta: float,
                      n_configs: int, n_therm: int = 100, n_skip: int = 10,
                      device: str = 'cuda', verbose: bool = True) -> torch.Tensor:
    """
    Generate ensemble of thermalized configurations.
    
    Args:
        lattice_dims: (T, X, Y, Z)
        beta: Inverse coupling
        n_configs: Number of configurations to generate
        n_therm: Initial thermalization sweeps
        n_skip: Sweeps between saved configurations
        device: 'cuda' or 'cpu'
        verbose: Print progress
        
    Returns:
        Ensemble tensor [n_configs, T, X, Y, Z, 4, 3, 3]
    """
    T, X, Y, Z = lattice_dims
    
    # Cold start (all links = identity)
    U = torch.eye(3, dtype=torch.complex64, device=device)
    U = U.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    U = U.expand(T, X, Y, Z, 4, 3, 3).clone()
    
    if verbose:
        print(f"Generating {n_configs} configs on {lattice_dims} at β={beta}")
        print(f"Thermalization: {n_therm}, Skip: {n_skip}")
        print("=" * 50)
    
    # Initial thermalization
    U = thermalize(U, beta, n_therm, verbose=verbose)
    
    # Generate ensemble
    ensemble = []
    
    for i in range(n_configs):
        # Skip sweeps for decorrelation
        for _ in range(n_skip):
            U = heatbath_sweep(U, beta)
        
        ensemble.append(U.clone())
        
        if verbose:
            plaq = compute_plaquette(U)
            print(f"Config {i+1}/{n_configs}: <P> = {plaq:.6f}")
    
    return torch.stack(ensemble)


# =============================================================================
# Quick validation test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SU(3) Heatbath MCMC Validation")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Small lattice for testing
    dims = (4, 4, 4, 4)
    beta = 6.0  # Weak coupling
    
    # Random start
    U = torch.randn(*dims, 4, 3, 3, dtype=torch.complex64, device=device)
    for t in range(dims[0]):
        for x in range(dims[1]):
            for y in range(dims[2]):
                for z in range(dims[3]):
                    for mu in range(4):
                        U[t, x, y, z, mu] = project_to_su3(U[t, x, y, z, mu])
    
    print(f"\nInitial plaquette: {compute_plaquette(U):.6f}")
    print(f"(Random configs have <P> ≈ 0)")
    print()
    
    # Thermalize
    U = thermalize(U, beta, n_therm=50, verbose=True)
    
    print(f"\n✓ Thermalization complete!")
    print(f"  For β=6.0, expect <P> ≈ 0.59")
    print(f"  Observed: <P> = {compute_plaquette(U):.4f}")
