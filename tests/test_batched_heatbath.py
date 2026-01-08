"""
Minimal Batched Heatbath Test
=============================

This is a minimal implementation that uses the EXACT same logic as 
heatbath_lr_fix.py (which works!) but in batched form.

Run this first to verify the batched heatbath works before integrating
into the full validation script.
"""

import torch
import numpy as np
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def random_su3_batch(batch_size: int) -> torch.Tensor:
    """Generate batch of random SU(3) matrices."""
    A = torch.randn(batch_size, 3, 3, dtype=torch.complex64, device=device)
    Q, R = torch.linalg.qr(A)
    det = torch.linalg.det(Q)
    phase = torch.exp(-1j * torch.angle(det) / 3).unsqueeze(-1).unsqueeze(-1)
    return Q * phase


def random_su2_batch(batch_size: int) -> torch.Tensor:
    """Generate batch of random SU(2) matrices (Haar measure)."""
    x = torch.randn(batch_size, 4, device=device)
    x = x / torch.norm(x, dim=1, keepdim=True)
    a0, a1, a2, a3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    
    U = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=device)
    U[:, 0, 0] = a0 + 1j * a3
    U[:, 0, 1] = a2 + 1j * a1
    U[:, 1, 0] = -a2 + 1j * a1
    U[:, 1, 1] = a0 - 1j * a3
    return U


def su2_heatbath_creutz_batch(alpha: torch.Tensor) -> torch.Tensor:
    """
    Batched Creutz SU(2) heatbath.
    
    alpha: (batch_size,) tensor of effective couplings
    Returns: (batch_size, 2, 2) SU(2) matrices
    
    Distribution: P(a0) ∝ sqrt(1-a0²) * exp(α*a0)
    """
    batch_size = alpha.shape[0]
    a0 = torch.zeros(batch_size, device=device, dtype=torch.float32)
    
    remaining = torch.ones(batch_size, dtype=torch.bool, device=device)
    max_iterations = 100
    
    for _ in range(max_iterations):
        n_remaining = remaining.sum().item()
        if n_remaining == 0:
            break
        
        alpha_rem = alpha[remaining].clamp(min=1e-6)
        
        r1 = torch.rand(n_remaining, device=device)
        r2 = torch.rand(n_remaining, device=device)
        r3 = torch.rand(n_remaining, device=device)
        r4 = torch.rand(n_remaining, device=device)
        
        log_r1 = torch.log(r1 + 1e-10)
        log_r2 = torch.log(r2 + 1e-10)
        cos2_term = torch.cos(2 * np.pi * r3) ** 2
        
        lambda_sq = -(log_r1 + log_r2 * cos2_term) / (2.0 * alpha_rem)
        
        valid = lambda_sq <= 1.0
        x = 1.0 - 2.0 * lambda_sq
        accept = valid & (r4 ** 2 <= 1.0 - lambda_sq)
        
        idx = torch.where(remaining)[0]
        a0[idx[accept]] = x[accept]
        remaining[idx[accept]] = False
    
    # Fallback for any remaining
    if remaining.any():
        alpha_fallback = alpha[remaining].clamp(min=1e-6)
        a0[remaining] = (1.0 - 1.0 / alpha_fallback).clamp(-1, 1)
    
    # Generate random direction for spatial components
    r_sq = (1.0 - a0 * a0).clamp(min=0)
    r_mag = torch.sqrt(r_sq)
    
    phi = 2 * np.pi * torch.rand(batch_size, device=device)
    cos_theta = 2 * torch.rand(batch_size, device=device) - 1
    sin_theta = torch.sqrt((1 - cos_theta * cos_theta).clamp(min=0))
    
    a1 = r_mag * sin_theta * torch.cos(phi)
    a2 = r_mag * sin_theta * torch.sin(phi)
    a3 = r_mag * cos_theta
    
    # Construct SU(2) matrix
    R = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=device)
    R[:, 0, 0] = a0 + 1j * a3
    R[:, 0, 1] = a2 + 1j * a1
    R[:, 1, 0] = -a2 + 1j * a1
    R[:, 1, 1] = a0 - 1j * a3
    
    return R


class BatchedLattice:
    """Minimal batched lattice for testing heatbath."""
    
    def __init__(self, N: int, L: int, beta: float):
        self.N = N
        self.L = L
        self.T = L
        self.beta = beta
        
        # Initialize to identity (cold start)
        eye = torch.eye(3, dtype=torch.complex64, device=device)
        self.links = eye.view(1, 1, 1, 1, 1, 1, 3, 3).expand(
            N, L, L, L, L, 4, 3, 3
        ).clone()
        
        # Create checkerboard indices
        coords = torch.stack(torch.meshgrid(
            torch.arange(L, device=device),
            torch.arange(L, device=device),
            torch.arange(L, device=device),
            torch.arange(L, device=device),
            indexing='ij'
        ), dim=-1).reshape(-1, 4)
        
        parity = coords.sum(dim=1) % 2
        self.red_idx = coords[parity == 0]
        self.black_idx = coords[parity == 1]
        self.all_indices = coords
    
    def get_links(self, indices, mu):
        """Get links at indices for direction mu. Returns (N, n_sites, 3, 3)"""
        return self.links[:, indices[:, 0], indices[:, 1], 
                         indices[:, 2], indices[:, 3], mu]
    
    def set_links(self, indices, mu, values):
        """Set links at indices for direction mu."""
        self.links[:, indices[:, 0], indices[:, 1], 
                  indices[:, 2], indices[:, 3], mu] = values
    
    def compute_staples(self, indices, mu):
        """
        Compute staples using SAME convention as working heatbath_lr_fix.py.
        Returns (N, n_sites, 3, 3)
        """
        L = self.L
        n_sites = indices.shape[0]
        staples = torch.zeros(self.N, n_sites, 3, 3, 
                             dtype=torch.complex64, device=device)
        
        for nu in range(4):
            if nu == mu:
                continue
            
            # Compute shifted indices
            idx_plus_mu = indices.clone()
            idx_plus_mu[:, mu] = (idx_plus_mu[:, mu] + 1) % L
            
            idx_plus_nu = indices.clone()
            idx_plus_nu[:, nu] = (idx_plus_nu[:, nu] + 1) % L
            
            idx_minus_nu = indices.clone()
            idx_minus_nu[:, nu] = (idx_minus_nu[:, nu] - 1 + L) % L
            
            idx_plus_mu_minus_nu = indices.clone()
            idx_plus_mu_minus_nu[:, mu] = (idx_plus_mu_minus_nu[:, mu] + 1) % L
            idx_plus_mu_minus_nu[:, nu] = (idx_plus_mu_minus_nu[:, nu] - 1 + L) % L
            
            # Forward staple: U_ν(x+μ) @ U_μ†(x+ν) @ U_ν†(x)
            # (Same as heatbath_lr_fix.py)
            U1 = self.get_links(idx_plus_mu, nu)
            U2 = self.get_links(idx_plus_nu, mu).mH  # conjugate transpose
            U3 = self.get_links(indices, nu).mH
            staples = staples + U1 @ U2 @ U3
            
            # Backward staple: U_ν†(x+μ-ν) @ U_μ†(x-ν) @ U_ν(x-ν)
            U1 = self.get_links(idx_plus_mu_minus_nu, nu).mH
            U2 = self.get_links(idx_minus_nu, mu).mH
            U3 = self.get_links(idx_minus_nu, nu)
            staples = staples + U1 @ U2 @ U3
        
        return staples
    
    def extract_su2(self, M, subgroup):
        """Extract SU(2) subgroup from SU(3). M is (..., 3, 3)"""
        if subgroup == 0:
            i, j = 0, 1
        elif subgroup == 1:
            i, j = 0, 2
        else:
            i, j = 1, 2
        
        su2 = torch.stack([
            torch.stack([M[..., i, i], M[..., i, j]], dim=-1),
            torch.stack([M[..., j, i], M[..., j, j]], dim=-1)
        ], dim=-2)
        return su2
    
    def embed_su2(self, su2, subgroup):
        """Embed SU(2) into SU(3). su2 is (..., 2, 2)"""
        batch_shape = su2.shape[:-2]
        eye = torch.eye(3, dtype=torch.complex64, device=device)
        result = eye.expand(*batch_shape, 3, 3).clone()
        
        if subgroup == 0:
            i, j = 0, 1
        elif subgroup == 1:
            i, j = 0, 2
        else:
            i, j = 1, 2
        
        result[..., i, i] = su2[..., 0, 0]
        result[..., i, j] = su2[..., 0, 1]
        result[..., j, i] = su2[..., 1, 0]
        result[..., j, j] = su2[..., 1, 1]
        
        return result
    
    def project_su3(self, M):
        """Project to SU(3) via SVD."""
        U, S, Vh = torch.linalg.svd(M)
        result = U @ Vh
        det = torch.linalg.det(result)
        phase = torch.exp(-1j * torch.angle(det) / 3)
        return result * phase.unsqueeze(-1).unsqueeze(-1)
    
    def heatbath_sweep(self, n_hits=1):
        """
        One heatbath sweep using EXACT same logic as heatbath_lr_fix.py.
        
        The THREE FIXES:
        1. W = staples @ links (not links @ staples)
        2. links = links @ su3_update (not su3_update @ links)
        3. su2_new = R @ V† (with the dagger!)
        """
        # alpha = 2*beta*k/3 (factor of 2 for SU(2) trace)
        beta_eff = 2.0 * self.beta / 3.0
        
        for mu in range(4):
            for indices in [self.red_idx, self.black_idx]:
                staples = self.compute_staples(indices, mu)  # (N, sites, 3, 3)
                links = self.get_links(indices, mu)          # (N, sites, 3, 3)
                
                for _ in range(n_hits):
                    for subgroup in range(3):
                        # FIX 1: W = staples @ links
                        W = staples @ links
                        
                        # Extract SU(2) subgroup
                        W_su2 = self.extract_su2(W, subgroup)  # (N, sites, 2, 2)
                        
                        # Flatten for batched heatbath
                        batch_shape = W_su2.shape[:-2]  # (N, sites)
                        batch_size = W_su2.numel() // 4
                        W_flat = W_su2.reshape(batch_size, 2, 2)
                        
                        # Compute k = sqrt(|det(W)|)
                        det_W = W_flat[:, 0, 0] * W_flat[:, 1, 1] - W_flat[:, 0, 1] * W_flat[:, 1, 0]
                        k = torch.sqrt(torch.abs(det_W) + 1e-10).real
                        
                        # α = 2*β*k/3
                        alpha = beta_eff * k
                        
                        # Generate R from heatbath
                        R = su2_heatbath_creutz_batch(alpha)
                        
                        # Normalize W to get V: W = k*V
                        V = W_flat / k.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-10)
                        
                        # FIX 3: su2_new = R @ V† (WITH the dagger!)
                        su2_new = R @ V.mH  # .mH = conjugate transpose
                        
                        # Reshape back
                        su2_new = su2_new.reshape(*batch_shape, 2, 2)
                        
                        # Embed in SU(3)
                        su3_update = self.embed_su2(su2_new, subgroup)
                        
                        # FIX 2: links = links @ su3_update
                        links = links @ su3_update
                
                # Project and store
                links = self.project_su3(links)
                self.set_links(indices, mu, links)
    
    def plaquette(self):
        """Compute average plaquette for each config. Returns (N,)"""
        L = self.L
        total = torch.zeros(self.N, device=device)
        count = 0
        
        for mu in range(4):
            for nu in range(mu + 1, 4):
                # Get all plaquettes
                indices = self.all_indices
                
                idx_plus_mu = indices.clone()
                idx_plus_mu[:, mu] = (idx_plus_mu[:, mu] + 1) % L
                
                idx_plus_nu = indices.clone()
                idx_plus_nu[:, nu] = (idx_plus_nu[:, nu] + 1) % L
                
                U1 = self.get_links(indices, mu)
                U2 = self.get_links(idx_plus_mu, nu)
                U3 = self.get_links(idx_plus_nu, mu).mH
                U4 = self.get_links(indices, nu).mH
                
                P = U1 @ U2 @ U3 @ U4  # (N, sites, 3, 3)
                traces = torch.real(torch.diagonal(P, dim1=-2, dim2=-1).sum(dim=-1))
                total += traces.sum(dim=1)
                count += L * L * L * L
        
        return total / (3.0 * count)


def main():
    print("=" * 60)
    print("Minimal Batched Heatbath Test")
    print("=" * 60)
    
    N = 100  # Same as validation script
    L = 8    # Same as validation script
    beta = 6.0
    
    lattice = BatchedLattice(N, L, beta)
    
    print(f"COLD START: all links = identity")
    print(f"Initial plaquette: {lattice.plaquette().mean().item():.6f}")
    print(f"Expected at β=6.0: ~0.59")
    print()
    
    print(f"Thermalizing at β={beta}...")
    print("-" * 40)
    
    for sweep in range(50):
        lattice.heatbath_sweep(n_hits=1)
        
        if (sweep + 1) % 5 == 0:
            plaq = lattice.plaquette().mean().item()
            print(f"  Sweep {sweep+1:4d}: <P> = {plaq:.6f}")
    
    print("-" * 40)
    final_plaq = lattice.plaquette().mean().item()
    print(f"Final plaquette: {final_plaq:.6f}")
    print()
    
    if 0.55 < final_plaq < 0.65:
        print("✅ SUCCESS! Batched heatbath working correctly!")
    elif final_plaq > 0.85:
        print("⚠️  Plaquette too high - needs more sweeps")
    elif 0.20 < final_plaq < 0.35:
        print("❌ Getting ~0.25-0.30 - original bug still present")
    elif final_plaq < 0.1:
        print("❌ Plaquette near 0 - heatbath not updating properly")
    else:
        print(f"⚠️  Unexpected value: {final_plaq:.4f}")


if __name__ == "__main__":
    main()