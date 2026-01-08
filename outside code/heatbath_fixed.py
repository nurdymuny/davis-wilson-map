"""
FIXED SU(3) Heatbath MCMC for Lattice QCD
=========================================

THE BUG WAS: Missing factor of 2 in the effective coupling!

For SU(2): Tr(U) = 2*a0
So P(a0) ∝ sqrt(1-a0²) * exp(α * a0) needs α = 2*β*k/3, NOT β*k/3

This single factor of 2 was causing equilibrium at <P> ≈ 0.25 instead of 0.59!

Reference: M. Creutz, Phys. Rev. D 21, 2308 (1980)
Author: Bee Rosa Davis & Claude  
Date: December 2025
"""

import torch
import math
from typing import Tuple


def su2_heatbath_creutz(alpha: float, device: torch.device) -> torch.Tensor:
    """
    Generate SU(2) matrix from heatbath distribution using Creutz algorithm.
    
    Distribution: P(a0) ∝ sqrt(1 - a0²) * exp(alpha * a0)
    
    Args:
        alpha: The FULL effective coupling (should be 2*β*k/3 for SU(3) Wilson)
        device: torch device
        
    Returns:
        SU(2) matrix [2, 2] complex
    """
    if alpha < 1e-6:
        return random_su2(device)
    
    # Creutz algorithm for sampling a0
    # P(a0) ∝ sqrt(1-a0²) * exp(α * a0) for a0 ∈ [-1, 1]
    
    max_iter = 1000
    for _ in range(max_iter):
        r1 = torch.rand(1, device=device)
        r2 = torch.rand(1, device=device)
        r3 = torch.rand(1, device=device)
        r4 = torch.rand(1, device=device)
        
        # Creutz formula: λ² = -1/(2α) * (ln(r1) + ln(r2)*cos²(2πr3))
        log_r1 = torch.log(r1 + 1e-10)
        log_r2 = torch.log(r2 + 1e-10)
        cos2_term = torch.cos(2 * math.pi * r3) ** 2
        
        lambda_sq = -(log_r1 + log_r2 * cos2_term) / (2.0 * alpha)
        
        if lambda_sq > 1.0:
            continue
        
        # a0 = 1 - 2*λ²
        a0 = 1.0 - 2.0 * lambda_sq
        
        # Accept with probability sqrt(1 - a0²) ∝ sqrt(1 - λ²)
        if r4 ** 2 <= 1.0 - lambda_sq:
            break
    else:
        a0 = torch.tensor(1.0 - 1.0/alpha, device=device).clamp(-1, 1)
    
    # Generate random direction for the 3-vector part
    r_sq = 1.0 - a0 ** 2
    r = torch.sqrt(torch.clamp(r_sq, 0, 1))
    
    cos_theta = 2 * torch.rand(1, device=device) - 1
    sin_theta = torch.sqrt(1 - cos_theta ** 2)
    phi = 2 * math.pi * torch.rand(1, device=device)
    
    a1 = r * sin_theta * torch.cos(phi)
    a2 = r * sin_theta * torch.sin(phi)
    a3 = r * cos_theta
    
    # SU(2) = a0*I + i*(a1*σ1 + a2*σ2 + a3*σ3)
    U = torch.zeros(2, 2, dtype=torch.complex64, device=device)
    U[0, 0] = a0 + 1j * a3
    U[0, 1] = a2 + 1j * a1
    U[1, 0] = -a2 + 1j * a1
    U[1, 1] = a0 - 1j * a3
    
    return U


def random_su2(device: torch.device) -> torch.Tensor:
    """Generate random SU(2) matrix using Haar measure."""
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
    if subgroup == 0:  # (0,1)
        return M[:2, :2].clone()
    elif subgroup == 1:  # (0,2)
        su2 = torch.zeros(2, 2, dtype=M.dtype, device=M.device)
        su2[0, 0] = M[0, 0]
        su2[0, 1] = M[0, 2]
        su2[1, 0] = M[2, 0]
        su2[1, 1] = M[2, 2]
        return su2
    else:  # (1,2)
        return M[1:, 1:].clone()


def embed_su2_in_su3(su2: torch.Tensor, subgroup: int, device: torch.device) -> torch.Tensor:
    """Embed SU(2) matrix into SU(3)."""
    su3 = torch.eye(3, dtype=torch.complex64, device=device)
    
    if subgroup == 0:
        su3[0, 0] = su2[0, 0]
        su3[0, 1] = su2[0, 1]
        su3[1, 0] = su2[1, 0]
        su3[1, 1] = su2[1, 1]
    elif subgroup == 1:
        su3[0, 0] = su2[0, 0]
        su3[0, 2] = su2[0, 1]
        su3[2, 0] = su2[1, 0]
        su3[2, 2] = su2[1, 1]
    else:
        su3[1, 1] = su2[0, 0]
        su3[1, 2] = su2[0, 1]
        su3[2, 1] = su2[1, 0]
        su3[2, 2] = su2[1, 1]
    
    return su3


def project_to_su3(M: torch.Tensor) -> torch.Tensor:
    """Project matrix to SU(3) using Gram-Schmidt."""
    u0 = M[:, 0].clone()
    u0 = u0 / (torch.norm(u0) + 1e-10)
    
    u1 = M[:, 1] - (u0.conj() @ M[:, 1]) * u0
    u1 = u1 / (torch.norm(u1) + 1e-10)
    
    u2 = M[:, 2] - (u0.conj() @ M[:, 2]) * u0 - (u1.conj() @ M[:, 2]) * u1
    u2 = u2 / (torch.norm(u2) + 1e-10)
    
    U = torch.stack([u0, u1, u2], dim=1)
    
    det = torch.linalg.det(U)
    phase = torch.angle(det) / 3.0
    U = U * torch.exp(-1j * phase)
    
    return U


def compute_staple(U: torch.Tensor, site: Tuple[int, ...], mu: int) -> torch.Tensor:
    """
    Compute staple sum A_μ(x) for link U_μ(x).
    
    The local action is: exp(β/3 * Re Tr(U_μ(x) @ A_μ(x)))
    
    where A is what completes the plaquette when multiplied by U_μ(x):
      P = U_μ(x) @ A_μ(x)
    
    Forward (ν > 0):  A = U_ν(x+μ) @ U†_μ(x+ν) @ U†_ν(x)
    Backward (ν < 0): A = U†_ν(x+μ-ν) @ U†_μ(x-ν) @ U_ν(x-ν)
    """
    device = U.device
    dtype = U.dtype
    dims = U.shape[:4]
    
    staple = torch.zeros(3, 3, dtype=dtype, device=device)
    t, x, y, z = site
    
    for nu in range(4):
        if nu == mu:
            continue
        
        def shift(s, d):
            return tuple((s[i] + d[i]) % dims[i] for i in range(4))
        
        delta_mu = [0, 0, 0, 0]
        delta_mu[mu] = 1
        delta_nu = [0, 0, 0, 0]
        delta_nu[nu] = 1
        delta_minus_nu = [0, 0, 0, 0]
        delta_minus_nu[nu] = -1
        
        x_plus_mu = shift(site, delta_mu)
        x_plus_nu = shift(site, delta_nu)
        x_minus_nu = shift(site, delta_minus_nu)
        x_plus_mu_minus_nu = shift(site, [delta_mu[i] + delta_minus_nu[i] for i in range(4)])
        
        # Forward staple: U_ν(x+μ) @ U†_μ(x+ν) @ U†_ν(x)
        staple += (U[x_plus_mu[0], x_plus_mu[1], x_plus_mu[2], x_plus_mu[3], nu] @
                   U[x_plus_nu[0], x_plus_nu[1], x_plus_nu[2], x_plus_nu[3], mu].conj().T @
                   U[t, x, y, z, nu].conj().T)
        
        # Backward staple: U†_ν(x+μ-ν) @ U†_μ(x-ν) @ U_ν(x-ν)
        staple += (U[x_plus_mu_minus_nu[0], x_plus_mu_minus_nu[1], 
                    x_plus_mu_minus_nu[2], x_plus_mu_minus_nu[3], nu].conj().T @
                   U[x_minus_nu[0], x_minus_nu[1], x_minus_nu[2], x_minus_nu[3], mu].conj().T @
                   U[x_minus_nu[0], x_minus_nu[1], x_minus_nu[2], x_minus_nu[3], nu])
    
    return staple


def cabibbo_marinari_update(U: torch.Tensor, site: Tuple[int, ...], mu: int, 
                            beta: float) -> torch.Tensor:
    """
    Cabibbo-Marinari SU(3) heatbath update for single link.
    
    THE BUG WAS: su2_new = R @ V.conj().T (should be R @ V!)
    
    We want Re Tr(new_su2 @ V†) = Re Tr(R) to match sampled distribution.
    If su2_new = R @ V, then Re Tr(R @ V @ V†) = Re Tr(R) ✓
    """
    device = U.device
    t, x, y, z = site
    link = U[t, x, y, z, mu].clone()
    staple = compute_staple(U, site, mu)
    
    for subgroup in range(3):
        # W = link @ staple
        W = link @ staple
        W_su2 = extract_su2_subgroup(W, subgroup)
        
        # Decompose W_su2 = k * V where V ≈ SU(2)
        det_W = W_su2[0, 0] * W_su2[1, 1] - W_su2[0, 1] * W_su2[1, 0]
        k = torch.sqrt(torch.abs(det_W) + 1e-10).real
        
        if k < 1e-8:
            continue
        
        V = W_su2 / (k + 1e-10)
        
        # α = 2βk/3 (factor of 2 because Tr(SU(2)) = 2*a0)
        alpha = 2.0 * beta * k.item() / 3.0
        
        # Sample R from heatbath
        R = su2_heatbath_creutz(alpha, device)
        
        # THE FIX: R @ V (NOT R @ V†)
        # We want Tr(su2_update @ V†) = Tr(R @ V @ V†) = Tr(R)
        su2_update = R @ V
        su3_update = embed_su2_in_su3(su2_update, subgroup, device)
        link = su3_update @ link
    
    link = project_to_su3(link)
    return link


def heatbath_sweep(U: torch.Tensor, beta: float) -> torch.Tensor:
    """One heatbath sweep over all links."""
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
    """Compute average plaquette."""
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
    """Test the FIXED heatbath."""
    print("=" * 60)
    print("FIXED SU(3) Heatbath - Factor of 2 Correction")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
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
    
    plaq_history = []
    for i in range(100):
        U = heatbath_sweep(U, beta)
        
        if (i + 1) % 5 == 0:
            plaq = compute_plaquette(U)
            plaq_history.append(plaq)
            print(f"  Sweep {i+1:4d}: <P> = {plaq:.6f}")
    
    print("-" * 40)
    final_plaq = compute_plaquette(U)
    print(f"Final plaquette: {final_plaq:.6f}")
    print()
    
    # Check convergence
    if len(plaq_history) >= 10:
        last_10_avg = sum(plaq_history[-10:]) / 10
        print(f"Average of last 10 measurements: {last_10_avg:.6f}")
    
    if 0.55 < final_plaq < 0.65:
        print("✅ SUCCESS! Thermalization working correctly!")
    elif final_plaq > 0.7:
        print("⚠️  Plaquette high - may need more sweeps to equilibrate")
    elif 0.20 < final_plaq < 0.30:
        print("❌ STILL WRONG - getting ~0.25 means factor of 2 bug persists!")
    else:
        print(f"⚠️  Unexpected value - check algorithm")


if __name__ == "__main__":
    main()
