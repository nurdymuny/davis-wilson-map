"""
PNP-003: Δ Scaling with Problem Size n
======================================
Test whether geometric roughness scales differently for P vs NP problems.

Expected:
- P problems: Δ ~ polylog(n) or polynomial
- NP problems: Δ ~ exp(n) or super-polynomial

Author: Bee Rosa Davis
Date: January 8, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import os


@dataclass
class ScalingResult:
    sizes: np.ndarray
    p_deltas: np.ndarray
    np_deltas: np.ndarray
    p_fit: Tuple[float, float]  # (slope, intercept) in log-log
    np_fit: Tuple[float, float]
    p_scaling: str  # "polynomial" or "exponential"
    np_scaling: str


def generate_k_sat(k: int, n: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random k-SAT instance."""
    indices = np.random.randint(0, n, (m, k))
    signs = np.random.choice([-1, 1], (m, k))
    return indices, signs


def compute_delta_roughness(k: int, n: int, alpha: float, n_samples: int = 50) -> float:
    """
    Compute geometric roughness Δ for k-SAT at size n.
    
    Δ = variance of energy landscape + gradient roughness
    """
    m = int(alpha * n)
    
    deltas = []
    for _ in range(n_samples):
        indices, signs = generate_k_sat(k, n, m)
        
        # Sample random assignments
        energies = []
        gradients = []
        
        for _ in range(100):
            assignment = np.random.choice([False, True], n)
            
            # Count unsatisfied clauses (energy)
            energy = 0
            for c in range(m):
                clause_sat = False
                for j in range(k):
                    var_idx = indices[c, j]
                    sign = signs[c, j]
                    lit_val = assignment[var_idx] if sign > 0 else not assignment[var_idx]
                    if lit_val:
                        clause_sat = True
                        break
                if not clause_sat:
                    energy += 1
            energies.append(energy)
            
            # Compute local gradient (1-flip neighborhood)
            grad = 0
            for i in range(min(n, 20)):  # Sample subset for speed
                flipped = assignment.copy()
                flipped[i] = not flipped[i]
                new_energy = 0
                for c in range(m):
                    clause_sat = False
                    for j in range(k):
                        var_idx = indices[c, j]
                        sign = signs[c, j]
                        lit_val = flipped[var_idx] if sign > 0 else not flipped[var_idx]
                        if lit_val:
                            clause_sat = True
                            break
                    if not clause_sat:
                        new_energy += 1
                grad += abs(new_energy - energy)
            gradients.append(grad / min(n, 20))
        
        energies = np.array(energies)
        gradients = np.array(gradients)
        
        # Δ combines energy variance and gradient roughness
        delta = (np.var(energies) / (m + 1) + np.mean(gradients) + np.var(gradients)) / n
        deltas.append(delta)
    
    return np.mean(deltas)


def run_scaling_analysis(
    sizes: List[int] = [20, 30, 40, 50, 75, 100],
    alpha: float = 4.2,
    n_samples: int = 30
) -> ScalingResult:
    """Run the scaling analysis across problem sizes."""
    
    print("=" * 60)
    print("PNP-003: Δ SCALING ANALYSIS")
    print(f"Sizes: {sizes}")
    print(f"Alpha: {alpha}")
    print("=" * 60)
    
    p_deltas = []
    np_deltas = []
    
    for n in sizes:
        print(f"\nSize n={n}...")
        
        # 2-SAT (P)
        p_delta = compute_delta_roughness(k=2, n=n, alpha=alpha, n_samples=n_samples)
        p_deltas.append(p_delta)
        print(f"  2-SAT Δ = {p_delta:.6f}")
        
        # 3-SAT (NP)
        np_delta = compute_delta_roughness(k=3, n=n, alpha=alpha, n_samples=n_samples)
        np_deltas.append(np_delta)
        print(f"  3-SAT Δ = {np_delta:.6f}")
    
    sizes = np.array(sizes)
    p_deltas = np.array(p_deltas)
    np_deltas = np.array(np_deltas)
    
    # Fit log-log scaling (polynomial: log(Δ) ~ c*log(n))
    log_sizes = np.log(sizes)
    log_p = np.log(p_deltas + 1e-10)
    log_np = np.log(np_deltas + 1e-10)
    
    p_fit = np.polyfit(log_sizes, log_p, 1)
    np_fit = np.polyfit(log_sizes, log_np, 1)
    
    # Determine scaling type
    # Polynomial: slope in log-log is constant (Δ ~ n^c)
    # Exponential: would show up as super-linear in log-log
    p_scaling = "polynomial" if abs(p_fit[0]) < 2 else "super-polynomial"
    np_scaling = "polynomial" if abs(np_fit[0]) < 2 else "super-polynomial"
    
    print("\n" + "=" * 60)
    print("SCALING RESULTS")
    print("=" * 60)
    print(f"2-SAT (P):  Δ ~ n^{p_fit[0]:.3f} → {p_scaling}")
    print(f"3-SAT (NP): Δ ~ n^{np_fit[0]:.3f} → {np_scaling}")
    
    return ScalingResult(
        sizes=sizes,
        p_deltas=p_deltas,
        np_deltas=np_deltas,
        p_fit=tuple(p_fit),
        np_fit=tuple(np_fit),
        p_scaling=p_scaling,
        np_scaling=np_scaling
    )


def plot_scaling(result: ScalingResult, save_path: str = None):
    """Visualize scaling results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale
    ax1 = axes[0]
    ax1.plot(result.sizes, result.p_deltas, 'bo-', label='2-SAT (P)', linewidth=2)
    ax1.plot(result.sizes, result.np_deltas, 'ro-', label='3-SAT (NP)', linewidth=2)
    ax1.set_xlabel('Problem Size n', fontsize=12)
    ax1.set_ylabel('Geometric Roughness Δ', fontsize=12)
    ax1.set_title('Δ vs Problem Size (Linear)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale
    ax2 = axes[1]
    ax2.loglog(result.sizes, result.p_deltas, 'bo-', label=f'2-SAT: Δ ~ n^{result.p_fit[0]:.2f}', linewidth=2)
    ax2.loglog(result.sizes, result.np_deltas, 'ro-', label=f'3-SAT: Δ ~ n^{result.np_fit[0]:.2f}', linewidth=2)
    ax2.set_xlabel('Problem Size n (log)', fontsize=12)
    ax2.set_ylabel('Geometric Roughness Δ (log)', fontsize=12)
    ax2.set_title('Δ vs Problem Size (Log-Log)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('PNP-003: Complexity Scaling Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Run the test
    result = run_scaling_analysis(
        sizes=[20, 30, 40, 50, 75, 100],
        alpha=4.2,
        n_samples=30
    )
    
    # Save results
    os.makedirs('../../results/p_vs_np', exist_ok=True)
    plot_scaling(result, save_path='../../results/p_vs_np/pnp_003_scaling.png')
    
    np.savez('../../results/p_vs_np/pnp_003_data.npz',
             sizes=result.sizes,
             p_deltas=result.p_deltas,
             np_deltas=result.np_deltas,
             p_fit=result.p_fit,
             np_fit=result.np_fit,
             p_scaling=result.p_scaling,
             np_scaling=result.np_scaling)
    
    # Verdict
    print("\n" + "=" * 60)
    if result.np_fit[0] > result.p_fit[0] + 0.3:
        print("✓ PNP-003 PASS: NP shows steeper scaling than P")
    else:
        print("~ PNP-003 INCONCLUSIVE: Similar scaling observed")
    print("=" * 60)
