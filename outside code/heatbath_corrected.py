"""
CORRECT SU(3) Heatbath MCMC for Lattice QCD
==========================================

Fixed implementation using the standard Creutz algorithm for SU(2) heatbath.

The bug in the previous version: incorrect rejection sampling that was
effectively generating uniform random SU(2) instead of thermal distribution.

Reference: M. Creutz, Phys. Rev. D 21, 2308 (1980)

Author: Bee Rosa Davis & Claude  
Date: December 2025
"""

import torch
import math
from typing import Tuple


def su2_heatbath_creutz(k: float, beta: float, device: torch.device) -> torch.Tensor:
    """
    Generate SU(2) matrix from heatbath distribution using Creutz algorithm.
    
    Distribution: P(a0) ∝ sqrt(1 - a0²) * exp(beta * k * a0)
    
    where k = sqrt(det(staple)) is the "effective coupling"
    
    The Creutz algorithm samples a0 = cos(θ/2) from this distribution.
    
    Args:
        k: Effective coupling (magnitude of staple)
        beta: Inverse coupling
        device: torch device
        
    Returns:
        SU(2) matrix [2, 2] complex
    """
    # Effective beta*k
    alpha = beta * k
    
    if alpha < 1e-6:
        # Very weak coupling: use random SU(2)
        return random_su2(device)
    
    # Creutz algorithm for sampling a0
    # We want P(a0) ∝ sqrt(1-a0²) * exp(α * a0) for a0 ∈ [-1, 1]
    
    max_iter = 1000
    for _ in range(max_iter):
        # Generate 4 uniform random numbers
        r1 = torch.rand(1, device=device)
        r2 = torch.rand(1, device=device)
        r3 = torch.rand(1, device=device)
        r4 = torch.rand(1, device=device)
        
        # Generate candidate for a0 using Creutz formula
        # x = cos²(θ) where a0 = 1 - 2*λ²
        # λ² = -1/(2α) * (ln(r1) + ln(r2)*cos²(2πr3))
        
        log_r1 = torch.log(r1 + 1e-10)
        log_r2 = torch.log(r2 + 1e-10)
        cos2_term = torch.cos(2 * math.pi * r3) ** 2
        
        lambda_sq = -(log_r1 + log_r2 * cos2_term) / (2.0 * alpha)
        
        # Reject if lambda² > 1 (a0 would be < -1)
        if lambda_sq > 1.0:
            continue
        
        # a0 = 1 - 2*lambda²
        a0 = 1.0 - 2.0 * lambda_sq
        
        # Accept/reject with probability sqrt(1 - a0²)
        # which equals sqrt(4*lambda² - 4*lambda⁴) = 2*lambda*sqrt(1 - lambda²)
        # We compare r4² with 1 - lambda² (simplified acceptance)
        
        if r4 ** 2 <= 1.0 - lambda_sq:
            break
    else:
        # Fallback: use mean value
        a0 = torch.tensor(1.0 - 1.0/alpha, device=device).clamp(-1, 1)
    
    # a0 is cos(θ/2), now generate random direction for the 3-vector part
    # SU(2) = a0*I + i*(a1*σ1 + a2*σ2 + a3*σ3)
    # Constraint: a0² + a1² + a2² + a3² = 1
    
    # Radius squared for spatial part
    r_sq = 1.0 - a0 ** 2
    r = torch.sqrt(torch.clamp(r_sq, 0, 1))
    
    # Random direction on 2-sphere
    cos_theta = 2 * torch.rand(1, device=device) - 1  # uniform in [-1, 1]
    sin_theta = torch.sqrt(1 - cos_theta ** 2)
    phi = 2 * math.pi * torch.rand(1, device=device)
    
    a1 = r * sin_theta * torch.cos(phi)
    a2 = r * sin_theta * torch.sin(phi)
    a3 = r * cos_theta
    
    # Construct SU(2) matrix
    # U = [[a0 + i*a3, a2 + i*a1], [-a2 + i*a1, a0 - i*a3]]
    U = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    U[0, 0] = a0 + 1j * a3
    U[0, 1] = a2 + 1j * a1
    U[1, 0] = -a2 + 1j * a1
    U[1, 1] = a0 - 1j * a3
    
    return U


def random_su2(device: torch.device) -> torch.Tensor:
    """Generate random SU(2) matrix using Haar measure (4-sphere)."""
    x = torch.randn(4, device=device)
    x = x / torch.norm(x)
    
    a0, a1, a2, a3 = x[0], x[1], x[2], x[3]
    
    U = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    U[0, 0] = a0 + 1j * a3
    U[0, 1] = a2 + 1j * a1
    U[1, 0] = -a2 + 1j * a1
    U[1, 1] = a0 - 1j * a3
    
    return U


def extract_su2_subgroup(M: torch.Tensor, subgroup: int) -> torch.Tensor:
    """Extract 2x2 SU(2) subgroup from 3x3 matrix."""
    if subgroup == 0:  # (0,1) subgroup
        return M[:2, :2].clone()
    elif subgroup == 1:  # (0,2) subgroup
        su2 = torch.zeros(2, 2, dtype=M.dtype, device=M.device)
        su2[0, 0] = M[0, 0]
        su2[0, 1] = M[0, 2]
        su2[1, 0] = M[2, 0]
        su2[1, 1] = M[2, 2]
        return su2
    else:  # (1,2) subgroup
        return M[1:, 1:].clone()


def embed_su2_in_su3(su2: torch.Tensor, subgroup: int, device: torch.device) -> torch.Tensor:
    """Embed SU(2) matrix into SU(3) identity."""
    su3 = torch.eye(3, dtype=torch.complex64, device=device)
    
    if subgroup == 0:  # (0,1) subgroup
        su3[0, 0] = su2[0, 0]
        su3[0, 1] = su2[0, 1]
        su3[1, 0] = su2[1, 0]
        su3[1, 1] = su2[1, 1]
    elif subgroup == 1:  # (0,2) subgroup
        su3[0, 0] = su2[0, 0]
        su3[0, 2] = su2[0, 1]
        su3[2, 0] = su2[1, 0]
        su3[2, 2] = su2[1, 1]
    else:  # (1,2) subgroup
        su3[1, 1] = su2[0, 0]
        su3[1, 2] = su2[0, 1]
        su3[2, 1] = su2[1, 0]
        su3[2, 2] = su2[1, 1]
    
    return su3


def project_to_su3(M: torch.Tensor) -> torch.Tensor:
    """Project matrix to SU(3) using modified Gram-Schmidt."""
    device = M.device
    
    # Gram-Schmidt
    u0 = M[:, 0].clone()
    u0 = u0 / (torch.norm(u0) + 1e-10)
    
    u1 = M[:, 1] - (u0.conj() @ M[:, 1]) * u0
    u1 = u1 / (torch.norm(u1) + 1e-10)
    
    u2 = M[:, 2] - (u0.conj() @ M[:, 2]) * u0 - (u1.conj() @ M[:, 2]) * u1
    u2 = u2 / (torch.norm(u2) + 1e-10)
    
    U = torch.stack([u0, u1, u2], dim=1)
    
    # Fix determinant to +1
    det = torch.linalg.det(U)
    phase = torch.angle(det) / 3.0
    U = U * torch.exp(-1j * phase)
    
    return U


def compute_staple(U: torch.Tensor, site: Tuple[int, ...], mu: int) -> torch.Tensor:
    """
    Compute staple sum for link U_μ(x).
    
    Staple_μ(x) = Σ_{ν≠μ} [U_ν(x+μ) U†_μ(x+ν) U†_ν(x) + U†_ν(x+μ-ν) U†_μ(x-ν) U_ν(x-ν)]
    """
    device = U.device
    dtype = U.dtype
    dims = U.shape[:4]
    
    staple = torch.zeros(3, 3, dtype=dtype, device=device)
    t, x, y, z = site
    
    for nu in range(4):
        if nu == mu:
            continue
        
        # Shift indices
        def shift(s, d):
            return tuple((s[i] + d[i]) % dims[i] for i in range(4))
        
        delta_mu = [0, 0, 0, 0]
        delta_mu[mu] = 1
        delta_nu = [0, 0, 0, 0]
        delta_nu[nu] = 1
        delta_minus_nu = [0, 0, 0, 0]
        delta_minus_nu[nu] = -1
        
        # Forward staple: U_ν(x+μ) @ U†_μ(x+ν) @ U†_ν(x)
        x_plus_mu = shift(site, delta_mu)
        x_plus_nu = shift(site, delta_nu)
        
        staple += (U[x_plus_mu[0], x_plus_mu[1], x_plus_mu[2], x_plus_mu[3], nu] @
                   U[x_plus_nu[0], x_plus_nu[1], x_plus_nu[2], x_plus_nu[3], mu].conj().T @
                   U[t, x, y, z, nu].conj().T)
        
        # Backward staple: U†_ν(x+μ-ν) @ U†_μ(x-ν) @ U_ν(x-ν)
        x_minus_nu = shift(site, delta_minus_nu)
        x_plus_mu_minus_nu = shift(site, [delta_mu[i] + delta_minus_nu[i] for i in range(4)])
        
        staple += (U[x_plus_mu_minus_nu[0], x_plus_mu_minus_nu[1], 
                    x_plus_mu_minus_nu[2], x_plus_mu_minus_nu[3], nu].conj().T @
                   U[x_minus_nu[0], x_minus_nu[1], x_minus_nu[2], x_minus_nu[3], mu].conj().T @
                   U[x_minus_nu[0], x_minus_nu[1], x_minus_nu[2], x_minus_nu[3], nu])
    
    return staple


def cabibbo_marinari_update(U: torch.Tensor, site: Tuple[int, ...], mu: int, 
                            beta: float) -> torch.Tensor:
    """
    Cabibbo-Marinari SU(3) heatbath update for single link.
    
    Update U_μ(x) by cycling through 3 SU(2) subgroups.
    """
    device = U.device
    t, x, y, z = site
    link = U[t, x, y, z, mu].clone()
    staple = compute_staple(U, site, mu)
    
    # Cycle through 3 SU(2) subgroups
    for subgroup in range(3):
        # Form W = link @ staple (the combination that appears in the action)
        W = link @ staple
        
        # Extract SU(2) subgroup of W
        W_su2 = extract_su2_subgroup(W, subgroup)
        
        # Compute effective coupling k = sqrt(|det(W_su2)|)
        det_W = W_su2[0, 0] * W_su2[1, 1] - W_su2[0, 1] * W_su2[1, 0]
        k = torch.sqrt(torch.abs(det_W) + 1e-10).real
        
        if k < 1e-8:
            # Degenerate: use random SU(2)
            su2_new = random_su2(device)
        else:
            # Normalize W_su2 to get V: W_su2 = k * V where V ∈ SU(2)
            V = W_su2 / (k + 1e-10)
            
            # Heatbath generates R from P(R) ∝ exp(β/3 * k * Re Tr(R))
            # Then new link subgroup is R @ V†
            R = su2_heatbath_creutz(k.item(), beta / 3.0, device)
            
            # The update matrix for the link
            su2_new = R @ V.conj().T
        
        # Embed in SU(3)
        su3_update = embed_su2_in_su3(su2_new, subgroup, device)
        
        # Update link: U_new = su3_update @ U_old
        link = su3_update @ link
    
    # Project to SU(3) to fix numerical drift
    link = project_to_su3(link)
    
    return link


def heatbath_sweep(U: torch.Tensor, beta: float) -> torch.Tensor:
    """One heatbath sweep over all links using checkerboard ordering."""
    dims = U.shape[:4]
    U_new = U.clone()
    
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
                                U_new, site, mu, beta
                            )
    
    return U_new


def compute_plaquette(U: torch.Tensor) -> float:
    """Compute average plaquette for monitoring."""
    dims = U.shape[:4]
    total = 0.0
    count = 0
    
    for t in range(dims[0]):
        for x in range(dims[1]):
            for y in range(dims[2]):
                for z in range(dims[3]):
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            def shift(s, d):
                                return tuple((s[i] + d[i]) % dims[i] for i in range(4))
                            
                            delta_mu = [0, 0, 0, 0]
                            delta_mu[mu] = 1
                            delta_nu = [0, 0, 0, 0]
                            delta_nu[nu] = 1
                            
                            site = (t, x, y, z)
                            x_plus_mu = shift(site, delta_mu)
                            x_plus_nu = shift(site, delta_nu)
                            
                            P = (U[t, x, y, z, mu] @ 
                                 U[x_plus_mu[0], x_plus_mu[1], x_plus_mu[2], x_plus_mu[3], nu] @
                                 U[x_plus_nu[0], x_plus_nu[1], x_plus_nu[2], x_plus_nu[3], mu].conj().T @
                                 U[t, x, y, z, nu].conj().T)
                            
                            total += P.trace().real / 3.0
                            count += 1
    
    return total / count


def main():
    """Test the corrected heatbath."""
    print("=" * 60)
    print("CORRECTED SU(3) Heatbath - Cold Start Test")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Small lattice for quick test
    dims = (4, 4, 4, 4)
    beta = 6.0
    
    # COLD START
    U = torch.eye(3, dtype=torch.complex64, device=device)
    U = U.view(1, 1, 1, 1, 1, 3, 3).expand(dims[0], dims[1], dims[2], dims[3], 4, 3, 3).clone()
    
    print(f"\nCOLD START: all links = identity")
    print(f"Initial plaquette: {compute_plaquette(U):.6f}")
    print(f"Expected at β=6.0 equilibrium: ~0.59")
    print()
    
    print(f"Thermalizing at β={beta}...")
    print("-" * 40)
    
    for i in range(50):
        U = heatbath_sweep(U, beta)
        
        if (i + 1) % 5 == 0:
            plaq = compute_plaquette(U)
            print(f"  Sweep {i+1:4d}: <P> = {plaq:.6f}")
    
    print("-" * 40)
    final_plaq = compute_plaquette(U)
    print(f"Final plaquette: {final_plaq:.6f}")
    print()
    
    if 0.55 < final_plaq < 0.65:
        print("✅ Thermalization working correctly!")
    elif final_plaq > 0.7:
        print("⚠️  Plaquette too high - may need more sweeps")
    else:
        print("❌ Plaquette too low - algorithm still has issues")


if __name__ == "__main__":
    main()
