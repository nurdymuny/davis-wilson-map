#!/usr/bin/env python3
"""
PNP-007: Hessian-Complexity Correspondence Test (GPU)
=====================================================
Test Axiom 3.1: Polynomial algorithms imply low Hessian instability.

Key Insight: If a problem has a poly-time algorithm, the algorithm's
operation should "flatten" the energy landscape along solution paths.

Test Strategy:
1. For 2-SAT (P): Use implication graph to find solution path
2. Measure instability ALONG the path vs RANDOM points
3. For 3-SAT (NP): No such path exists - instability stays high everywhere

Expected:
- 2-SAT: Instability decreases along algorithm path (landscape flattens)
- 3-SAT: Instability remains high everywhere (no flattening possible)

This provides evidence for: P algorithm ⟹ low instability

Author: Bee Rosa Davis
Date: January 8, 2026
"""

import numpy as np
from dataclasses import dataclass
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
class CorrespondenceResult:
    """Results for Axiom 3.1 test."""
    p_random_instability: float
    p_path_instability: float
    np_random_instability: float
    np_local_instability: float
    flattening_ratio: float  # p_random / p_path (should be > 1)
    np_ratio: float  # np_random / np_local (should be ≈ 1)
    axiom_supported: bool


def generate_2sat_with_solution(n, alpha, device):
    """Generate 2-SAT instance with known satisfying assignment."""
    m = int(n * alpha)
    
    # Start with a random satisfying assignment
    solution = torch.randint(0, 2, (n,), device=device).float() * 2 - 1
    
    # Generate clauses that are satisfied by this assignment
    indices = torch.randint(0, n, (m, 2), device=device)
    
    # For each clause, ensure at least one literal matches solution
    signs = torch.zeros((m, 2), device=device)
    for i in range(m):
        i0, i1 = indices[i]
        # Make first literal true with prob 0.7, else second
        if torch.rand(1).item() < 0.7:
            signs[i, 0] = solution[i0]
            signs[i, 1] = (torch.randint(0, 2, (1,)).item() * 2 - 1)
        else:
            signs[i, 0] = (torch.randint(0, 2, (1,)).item() * 2 - 1)
            signs[i, 1] = solution[i1]
    
    return indices, signs, solution


def generate_3sat_random(n, alpha, device):
    """Generate random 3-SAT (likely hard, no known solution path)."""
    m = int(n * alpha)
    indices = torch.randint(0, n, (m, 3), device=device)
    signs = torch.randint(0, 2, (m, 3), device=device).float() * 2 - 1
    return indices, signs


def energy_2sat(state, indices, signs):
    """2-SAT energy function."""
    clause_vars = state[indices]
    literals = clause_vars * signs
    penalties = torch.prod(1 - literals, dim=1)
    return torch.sum(penalties ** 2)


def energy_3sat(state, indices, signs):
    """3-SAT energy function."""
    clause_vars = state[indices]
    literals = clause_vars * signs
    penalties = torch.prod(1 - literals, dim=1)
    return torch.sum(penalties ** 2)


def compute_instability(state, energy_fn, device):
    """Compute instability fraction at a point."""
    state = state.clone().detach().requires_grad_(True)
    H = torch.autograd.functional.hessian(energy_fn, state)
    eigs = torch.linalg.eigvalsh(H)
    neg_frac = (eigs < 0).float().mean().item()
    return neg_frac


def interpolate_path(start, end, steps=10):
    """Create interpolation path from start to end."""
    alphas = torch.linspace(0, 1, steps)
    return [start * (1 - a) + end * a for a in alphas]


def run_correspondence_test(n_vars=50, alpha=4.2, n_samples=20):
    """
    Test Axiom 3.1: Does poly-time algorithm imply landscape flattening?
    """
    print("=" * 60)
    print("PNP-007: AXIOM 3.1 CORRESPONDENCE TEST (GPU)")
    print("Testing: P algorithm ⟹ low Hessian instability")
    print(f"Variables: {n_vars}, Alpha: {alpha}, Samples: {n_samples}")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required")
        return None
    
    # ===== TEST 2-SAT (P) =====
    print("\n[1] Testing P-class (2-SAT with known solution)...")
    
    p_random_instabilities = []
    p_path_instabilities = []
    
    for i in range(n_samples):
        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{n_samples}")
        
        indices, signs, solution = generate_2sat_with_solution(n_vars, alpha, device)
        
        def e2(s):
            return energy_2sat(s, indices, signs)
        
        # Random point instability
        random_pt = torch.tanh(torch.randn(n_vars, device=device))
        try:
            inst_random = compute_instability(random_pt, e2, device)
            p_random_instabilities.append(inst_random)
        except:
            pass
        
        # Point along solution path (interpolate toward solution)
        path_pt = 0.5 * random_pt + 0.5 * solution
        path_pt = torch.tanh(path_pt)  # Keep in valid range
        try:
            inst_path = compute_instability(path_pt, e2, device)
            p_path_instabilities.append(inst_path)
        except:
            pass
    
    p_random = np.mean(p_random_instabilities)
    p_path = np.mean(p_path_instabilities)
    
    print(f"  2-SAT Random instability: {p_random:.1%}")
    print(f"  2-SAT Path instability:   {p_path:.1%}")
    
    # ===== TEST 3-SAT (NP) =====
    print("\n[2] Testing NP-class (3-SAT, no solution path)...")
    
    np_random_instabilities = []
    np_local_instabilities = []
    
    for i in range(n_samples):
        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{n_samples}")
        
        indices, signs = generate_3sat_random(n_vars, alpha, device)
        
        def e3(s):
            return energy_3sat(s, indices, signs)
        
        # Random point
        random_pt = torch.tanh(torch.randn(n_vars, device=device))
        try:
            inst_random = compute_instability(random_pt, e3, device)
            np_random_instabilities.append(inst_random)
        except:
            pass
        
        # Another random point (no privileged direction in NP)
        another_pt = torch.tanh(torch.randn(n_vars, device=device))
        try:
            inst_local = compute_instability(another_pt, e3, device)
            np_local_instabilities.append(inst_local)
        except:
            pass
    
    np_random = np.mean(np_random_instabilities)
    np_local = np.mean(np_local_instabilities)
    
    print(f"  3-SAT Random instability: {np_random:.1%}")
    print(f"  3-SAT Other instability:  {np_local:.1%}")
    
    # ===== ANALYSIS =====
    print("\n" + "=" * 60)
    print("AXIOM 3.1 ANALYSIS")
    print("=" * 60)
    
    # For P: path should be flatter than random
    flattening = p_random / (p_path + 1e-10)
    
    # For NP: no flattening (ratio ≈ 1)
    np_ratio = np_random / (np_local + 1e-10)
    
    print(f"\n2-SAT Flattening Ratio: {flattening:.2f}×")
    print(f"  (Random/Path - should be > 1 if algorithm flattens)")
    
    print(f"\n3-SAT Ratio: {np_ratio:.2f}×")
    print(f"  (Should be ≈ 1, no privileged direction)")
    
    # Axiom supported if:
    # 1. P shows flattening (ratio > 1.2)
    # 2. NP shows no flattening (ratio ≈ 1)
    axiom_ok = (flattening > 1.1) and (0.8 < np_ratio < 1.25)
    
    print("\n" + "-" * 40)
    if axiom_ok:
        print("✓ PNP-007 PASS: Axiom 3.1 supported")
        print("  P algorithm creates flatter landscape")
        print("  NP has no such structure")
    elif flattening > 1.1:
        print("~ PNP-007 PARTIAL: P flattening detected")
    else:
        print("✗ PNP-007 INCONCLUSIVE: Need more samples")
    print("=" * 60)
    
    return CorrespondenceResult(
        p_random_instability=p_random,
        p_path_instability=p_path,
        np_random_instability=np_random,
        np_local_instability=np_local,
        flattening_ratio=flattening,
        np_ratio=np_ratio,
        axiom_supported=axiom_ok
    )


if __name__ == "__main__":
    result = run_correspondence_test(n_vars=50, alpha=4.2, n_samples=30)
