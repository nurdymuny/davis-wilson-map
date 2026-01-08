"""
PNP-005: NP ∩ co-NP Intermediate Complexity
===========================================
Test whether problems in NP ∩ co-NP show intermediate geometric roughness.

NP ∩ co-NP problems:
- Integer factoring (believed not NP-complete)
- Primality testing (now known P, but good test case)
- Graph isomorphism (not known to be NP-complete)

Expected: Δ(P) < Δ(NP ∩ co-NP) < Δ(NP-complete)

Author: Bee Rosa Davis
Date: January 8, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import os


@dataclass
class IntermediateResult:
    p_delta: float       # 2-SAT
    inter_delta: float   # NP ∩ co-NP proxy
    np_delta: float      # 3-SAT
    ordering_correct: bool
    

def generate_k_sat(k: int, n: int, m: int):
    """Generate random k-SAT."""
    indices = np.random.randint(0, n, (m, k))
    signs = np.random.choice([-1, 1], (m, k))
    return indices, signs


def compute_sat_delta(k: int, n: int, alpha: float, n_samples: int = 50) -> float:
    """Compute Δ for k-SAT."""
    m = int(alpha * n)
    deltas = []
    
    for _ in range(n_samples):
        indices, signs = generate_k_sat(k, n, m)
        energies = []
        
        for _ in range(100):
            assignment = np.random.choice([False, True], n)
            energy = 0
            for c in range(m):
                clause_sat = any(
                    (assignment[indices[c, j]] if signs[c, j] > 0 
                     else not assignment[indices[c, j]])
                    for j in range(k)
                )
                if not clause_sat:
                    energy += 1
            energies.append(energy)
        
        deltas.append(np.var(energies) / (m + 1))
    
    return np.mean(deltas)


def compute_factoring_proxy_delta(n_bits: int, n_samples: int = 50) -> float:
    """
    Compute Δ for a factoring-like problem.
    
    We use a proxy: finding factors as a constraint satisfaction problem.
    Given N = p * q, find p, q such that their product equals N.
    
    This creates a landscape that's:
    - Not trivially solvable (like 2-SAT)
    - Not as fragmented as 3-SAT
    """
    deltas = []
    
    for _ in range(n_samples):
        # Generate a semiprime (product of two primes)
        # For simplicity, use random odd numbers as proxy
        p_true = np.random.randint(2**(n_bits//2 - 1), 2**(n_bits//2)) | 1
        q_true = np.random.randint(2**(n_bits//2 - 1), 2**(n_bits//2)) | 1
        N = p_true * q_true
        
        # Energy landscape: distance from correct factorization
        energies = []
        for _ in range(100):
            # Random guess for factors
            p_guess = np.random.randint(2, int(np.sqrt(N)) + 1)
            q_guess = N // p_guess if p_guess > 0 else N
            
            # Energy = how far from valid factorization
            product = p_guess * q_guess
            energy = abs(N - product) / N
            energies.append(energy)
        
        deltas.append(np.var(energies))
    
    return np.mean(deltas)


def compute_graph_iso_proxy_delta(n_vertices: int, n_samples: int = 50) -> float:
    """
    Compute Δ for graph isomorphism proxy.
    
    Two graphs G1, G2 are isomorphic if there's a bijection preserving edges.
    The search space is all permutations - n! possibilities.
    
    Energy = number of edge mismatches under a permutation.
    """
    deltas = []
    edge_prob = 0.3  # Erdos-Renyi parameter
    
    for _ in range(n_samples):
        # Generate random graph G1
        G1 = np.random.random((n_vertices, n_vertices)) < edge_prob
        G1 = np.triu(G1, 1)
        G1 = G1 + G1.T  # Symmetric
        
        # Generate G2 as isomorphic copy (random permutation of G1)
        perm = np.random.permutation(n_vertices)
        G2 = G1[perm][:, perm]
        
        # Sample random permutations and compute energy
        energies = []
        for _ in range(100):
            random_perm = np.random.permutation(n_vertices)
            G1_permuted = G1[random_perm][:, random_perm]
            
            # Energy = edge mismatches
            energy = np.sum(G1_permuted != G2) / (n_vertices * n_vertices)
            energies.append(energy)
        
        deltas.append(np.var(energies))
    
    return np.mean(deltas)


def run_intermediate_test(n: int = 50, alpha: float = 4.2, n_samples: int = 30) -> IntermediateResult:
    """Run the NP ∩ co-NP intermediate complexity test."""
    
    print("=" * 60)
    print("PNP-005: NP ∩ co-NP INTERMEDIATE COMPLEXITY TEST")
    print("=" * 60)
    
    # P class: 2-SAT
    print("\nComputing 2-SAT (P) roughness...")
    p_delta = compute_sat_delta(k=2, n=n, alpha=alpha, n_samples=n_samples)
    print(f"  Δ(2-SAT) = {p_delta:.6f}")
    
    # NP ∩ co-NP proxy: Average of factoring and graph-iso proxies
    print("\nComputing NP ∩ co-NP proxy roughness...")
    factor_delta = compute_factoring_proxy_delta(n_bits=n, n_samples=n_samples)
    print(f"  Δ(factoring proxy) = {factor_delta:.6f}")
    
    iso_delta = compute_graph_iso_proxy_delta(n_vertices=min(n, 20), n_samples=n_samples)
    print(f"  Δ(graph-iso proxy) = {iso_delta:.6f}")
    
    inter_delta = (factor_delta + iso_delta) / 2
    print(f"  Δ(NP∩co-NP avg) = {inter_delta:.6f}")
    
    # NP-complete: 3-SAT
    print("\nComputing 3-SAT (NP-complete) roughness...")
    np_delta = compute_sat_delta(k=3, n=n, alpha=alpha, n_samples=n_samples)
    print(f"  Δ(3-SAT) = {np_delta:.6f}")
    
    # Check ordering
    ordering_correct = p_delta < inter_delta < np_delta
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Δ(P) = {p_delta:.6f}")
    print(f"Δ(NP∩co-NP) = {inter_delta:.6f}")
    print(f"Δ(NP-complete) = {np_delta:.6f}")
    print(f"\nExpected ordering: Δ(P) < Δ(NP∩co-NP) < Δ(NP-complete)")
    print(f"Observed: {p_delta:.4f} {'<' if p_delta < inter_delta else '>='} {inter_delta:.4f} {'<' if inter_delta < np_delta else '>='} {np_delta:.4f}")
    
    if ordering_correct:
        print("\n✓ PNP-005 PASS: Intermediate complexity shows intermediate Δ")
    else:
        print("\n~ PNP-005 PARTIAL: Ordering not strictly satisfied")
    print("=" * 60)
    
    return IntermediateResult(
        p_delta=p_delta,
        inter_delta=inter_delta,
        np_delta=np_delta,
        ordering_correct=ordering_correct
    )


def plot_intermediate(result: IntermediateResult, save_path: str = None):
    """Visualize intermediate complexity results."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    classes = ['P\n(2-SAT)', 'NP∩co-NP\n(Factoring/GI)', 'NP-complete\n(3-SAT)']
    deltas = [result.p_delta, result.inter_delta, result.np_delta]
    colors = ['green', 'orange', 'red']
    
    bars = ax.bar(classes, deltas, color=colors, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Geometric Roughness Δ', fontsize=12)
    ax.set_title('PNP-005: Complexity Hierarchy via Geometric Roughness', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{delta:.4f}', ha='center', va='bottom', fontsize=11)
    
    verdict = "✓ PASS" if result.ordering_correct else "~ PARTIAL"
    ax.text(0.02, 0.98, f'Ordering: {verdict}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    result = run_intermediate_test(n=50, alpha=4.2, n_samples=30)
    
    os.makedirs('../../results/p_vs_np', exist_ok=True)
    plot_intermediate(result, save_path='../../results/p_vs_np/pnp_005_intermediate.png')
    
    np.savez('../../results/p_vs_np/pnp_005_data.npz',
             p_delta=result.p_delta,
             inter_delta=result.inter_delta,
             np_delta=result.np_delta,
             ordering_correct=result.ordering_correct)
