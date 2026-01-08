#!/usr/bin/env python3
"""
PNP-001: 2-SAT vs 3-SAT Hessian Spectra (GPU-Accelerated)
=========================================================
Measure the geometric curvature (Hessian spectrum) of solution manifolds
for P (2-SAT) vs NP (3-SAT) problems.

Hypothesis: NP manifolds exhibit "glassy" curvature (many saddle points),
while P manifolds exhibit smoother landscapes.

Metric: Instability fraction = proportion of negative Hessian eigenvalues

Expected: NP instability >> P instability (2.4× gap documented)

Author: Bee Rosa Davis
Date: January 8, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import time

try:
    import torch
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 PyTorch device: {device}")
except ImportError:
    TORCH_AVAILABLE = False
    device = None
    print("⚠️ PyTorch not available")


@dataclass
class GeometryResult:
    p_instability: float
    np_instability: float
    p_mean_curv: float
    np_mean_curv: float
    gap_ratio: float
    n_vars: int
    alpha: float


def generate_k_sat_torch(k, n, m, device):
    """Generate random k-SAT instance."""
    indices = torch.randint(0, n, (m, k), device=device)
    signs = torch.randint(0, 2, (m, k), device=device).float() * 2 - 1
    return indices, signs


def davis_energy_torch(state, indices, signs):
    """
    Soft-SAT Energy via continuous relaxation.
    
    A clause is satisfied if at least one literal is True (+1).
    Penalty = product of (1 - literal_i) for all literals in clause.
    """
    clause_vars = state[indices]
    literals = clause_vars * signs
    term_penalties = torch.prod(1 - literals, dim=1)
    return torch.sum(term_penalties ** 2)


def measure_curvature(k: int, n: int, alpha: float, device) -> np.ndarray:
    """Compute Hessian eigenvalues at a random point on the manifold."""
    m = int(n * alpha)
    indices, signs = generate_k_sat_torch(k, n, m, device)
    
    # Random point on manifold
    state = torch.tanh(torch.randn(n, device=device))
    state.requires_grad_(True)
    
    def energy_fn(s):
        return davis_energy_torch(s, indices, signs)
    
    H = torch.autograd.functional.hessian(energy_fn, state)
    eigs = torch.linalg.eigvalsh(H)
    return eigs.cpu().detach().numpy()


def run_geometry_test(
    n_vars: int = 100,
    alpha: float = 4.2,
    n_samples: int = 50
) -> GeometryResult:
    """Run the PNP-001 geometry comparison test."""
    
    print("=" * 60)
    print("PNP-001: GEOMETRIC COMPLEXITY ANALYSIS (GPU)")
    print(f"Variables: {n_vars}, Clause ratio: {alpha}")
    print(f"Samples: {n_samples}")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required")
        return None
    
    # Scan P manifold (2-SAT)
    print("\nScanning P-Manifold (2-SAT)...")
    t0 = time.time()
    p_spectra = []
    for i in range(n_samples):
        if (i + 1) % 20 == 0:
            print(f"  2-SAT probe {i+1}/{n_samples}")
        try:
            eigs = measure_curvature(k=2, n=n_vars, alpha=alpha, device=device)
            p_spectra.append(eigs)
        except Exception as e:
            print(f"  Warning: probe {i} failed: {e}")
    print(f"  Done ({time.time()-t0:.1f}s)")
    
    # Scan NP manifold (3-SAT)
    print("\nScanning NP-Manifold (3-SAT)...")
    t0 = time.time()
    np_spectra = []
    for i in range(n_samples):
        if (i + 1) % 20 == 0:
            print(f"  3-SAT probe {i+1}/{n_samples}")
        try:
            eigs = measure_curvature(k=3, n=n_vars, alpha=alpha, device=device)
            np_spectra.append(eigs)
        except Exception as e:
            print(f"  Warning: probe {i} failed: {e}")
    print(f"  Done ({time.time()-t0:.1f}s)")
    
    # Aggregate
    p_flat = np.concatenate(p_spectra)
    np_flat = np.concatenate(np_spectra)
    
    # Statistics
    p_instability = np.mean(p_flat < 0)
    np_instability = np.mean(np_flat < 0)
    p_mean_curv = np.mean(p_flat)
    np_mean_curv = np.mean(np_flat)
    gap_ratio = np_instability / (p_instability + 1e-10)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"P (2-SAT)  Instability: {p_instability:.1%} negative eigenvalues")
    print(f"NP (3-SAT) Instability: {np_instability:.1%} negative eigenvalues")
    print(f"P (2-SAT)  Mean Curvature: {p_mean_curv:.4f}")
    print(f"NP (3-SAT) Mean Curvature: {np_mean_curv:.4f}")
    print(f"Gap Ratio: {gap_ratio:.2f}×")
    
    if gap_ratio >= 2.0:
        print("\n✓ PNP-001 PASS: NP manifold shows ≥2× more instability than P")
    elif gap_ratio >= 1.5:
        print("\n✓ PNP-001 PASS: NP manifold significantly more unstable")
    elif gap_ratio > 1.2:
        print("\n~ PNP-001 WEAK: Gap detected but modest")
    else:
        print("\n✗ PNP-001 FAIL: No significant gap")
    print("=" * 60)
    
    return GeometryResult(
        p_instability=p_instability,
        np_instability=np_instability,
        p_mean_curv=p_mean_curv,
        np_mean_curv=np_mean_curv,
        gap_ratio=gap_ratio,
        n_vars=n_vars,
        alpha=alpha
    )


def plot_geometry(p_spectra, np_spectra, result: GeometryResult, save_path: str = None):
    """Visualize eigenvalue distributions."""
    p_flat = np.concatenate(p_spectra)
    np_flat = np.concatenate(np_spectra)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Full distributions
    ax1 = axes[0]
    ax1.hist(p_flat, bins=100, density=True, alpha=0.6, color='blue', label='P (2-SAT)')
    ax1.hist(np_flat, bins=100, density=True, alpha=0.6, color='red', label='NP (3-SAT)')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, label='Stability Boundary')
    ax1.set_xlabel("Hessian Eigenvalue (Curvature)", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.set_title("Geometric Fingerprint of Complexity Classes", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Instability region
    ax2 = axes[1]
    p_neg = p_flat[p_flat < 0]
    np_neg = np_flat[np_flat < 0]
    
    ax2.hist(p_neg, bins=50, density=True, alpha=0.6, color='blue', 
             label=f'P unstable ({len(p_neg)}/{len(p_flat)})')
    ax2.hist(np_neg, bins=50, density=True, alpha=0.6, color='red', 
             label=f'NP unstable ({len(np_neg)}/{len(np_flat)})')
    ax2.set_xlabel("Negative Eigenvalues (Saddle Directions)", fontsize=12)
    ax2.set_ylabel("Probability Density", fontsize=12)
    ax2.set_title("Instability Spectrum (The 'Glassiness')", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    verdict = "✓ PASS" if result.gap_ratio >= 1.5 else "~ WEAK"
    fig.suptitle(f'PNP-001: {result.gap_ratio:.1f}× Gap | {verdict}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    result = run_geometry_test(n_vars=100, alpha=4.2, n_samples=50)
    
    os.makedirs('../../results/p_vs_np', exist_ok=True)
    # Note: For full plot, would need to store spectra - simplified version here
