"""
Diagnostic tests for SU(2) heatbath sampler
============================================

Test 1: Check E[a0] against I_2(α)/I_1(α) theory
Test 2: Check before/after trace diagnostic
Test 3: Test Frobenius k normalization

Author: Bee Rosa Davis
Date: December 2025
"""

import torch
import math
from scipy.special import iv as bessel_i  # Modified Bessel function I_n

# Import the sampler from heatbath_fixed
import sys
sys.path.insert(0, '.')
from heatbath_fixed import su2_heatbath_creutz, random_su2


def test_su2_sampler(alphas=(0.5, 1.0, 2.0, 4.0, 8.0, 12.0), N=10000, device="cuda"):
    """
    Test that E[a0] matches I_2(α)/I_1(α).
    
    For the distribution P(a0) ∝ sqrt(1-a0²) * exp(α*a0),
    the theoretical expectation is E[a0] = I_2(α)/I_1(α).
    """
    print("="*60)
    print("TEST 1: SU(2) Sampler - E[a0] vs I_2(α)/I_1(α)")
    print("="*60)
    print(f"Sampling {N} matrices per alpha value")
    print()
    
    device = torch.device(device)
    
    results = []
    for alpha in alphas:
        a0_samples = []
        
        for _ in range(N):
            U = su2_heatbath_creutz(alpha, device)
            # For SU(2): U = a0*I + i*(a1*σ1 + a2*σ2 + a3*σ3)
            # So Tr(U) = 2*a0, hence a0 = Re(Tr(U))/2
            a0 = (U.trace().real / 2.0).item()
            a0_samples.append(a0)
        
        mc_mean = sum(a0_samples) / len(a0_samples)
        mc_std = (sum((x - mc_mean)**2 for x in a0_samples) / len(a0_samples)) ** 0.5
        mc_err = mc_std / (len(a0_samples) ** 0.5)
        
        # Theoretical: E[a0] = I_2(α) / I_1(α)
        theo = bessel_i(2, alpha) / bessel_i(1, alpha)
        
        diff = mc_mean - theo
        status = "✓" if abs(diff) < 3 * mc_err else "✗"
        
        print(f"α={alpha:>5.1f}:  MC E[a0]={mc_mean:.4f}±{mc_err:.4f}   "
              f"theory={theo:.4f}   diff={diff:+.4f}  {status}")
        
        results.append((alpha, mc_mean, theo, diff))
    
    print()
    return results


def test_before_after_trace(device="cuda"):
    """
    Test that updates increase Re Tr(link @ staple).
    
    If updates are working correctly, they should align the link
    with the staple, increasing the trace (= decreasing action).
    """
    print("="*60)
    print("TEST 2: Before/After Trace Diagnostic")
    print("="*60)
    
    from heatbath_fixed import compute_staple, cabibbo_marinari_update, project_to_su3
    
    device = torch.device(device)
    dims = (4, 4, 4, 4)
    beta = 6.0
    
    # Start from slightly disordered state (not cold, not hot)
    # This is more representative of mid-thermalization
    U = torch.eye(3, dtype=torch.complex64, device=device)
    U = U.view(1, 1, 1, 1, 1, 3, 3).expand(dims[0], dims[1], dims[2], dims[3], 4, 3, 3).clone()
    
    # Add small noise to break perfect ordering
    for t in range(dims[0]):
        for x in range(dims[1]):
            for y in range(dims[2]):
                for z in range(dims[3]):
                    for mu in range(4):
                        noise = 0.1 * torch.randn(3, 3, dtype=torch.complex64, device=device)
                        U[t, x, y, z, mu] = project_to_su3(U[t, x, y, z, mu] + noise)
    
    # Track before/after for many updates
    increases = 0
    decreases = 0
    total = 0
    
    print(f"Testing single updates at β={beta}...")
    print()
    
    for t in range(2):
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    for mu in range(4):
                        site = (t, x, y, z)
                        link = U[t, x, y, z, mu]
                        staple = compute_staple(U, site, mu)
                        
                        before = (link @ staple).trace().real / 3.0
                        
                        # Do update
                        new_link = cabibbo_marinari_update(U, site, mu, beta)
                        
                        after = (new_link @ staple).trace().real / 3.0
                        
                        if after > before:
                            increases += 1
                        else:
                            decreases += 1
                        total += 1
                        
                        if total <= 10:
                            print(f"  Site {site} μ={mu}: before={before.item():.4f} "
                                  f"after={after.item():.4f} Δ={after.item()-before.item():+.4f}")
    
    print()
    print(f"Total updates: {total}")
    print(f"  Increases (good): {increases} ({100*increases/total:.1f}%)")
    print(f"  Decreases (bad):  {decreases} ({100*decreases/total:.1f}%)")
    print()
    
    if increases > decreases:
        print("✓ Updates are mostly aligning with staple (correct behavior)")
    else:
        print("✗ Updates are mostly anti-aligning (WRONG - check multiplication order)")
    
    return increases, decreases


def test_k_normalization(device="cuda"):
    """
    Compare det-based k vs Frobenius-based k.
    """
    print("="*60)
    print("TEST 3: k Normalization Methods")
    print("="*60)
    
    from heatbath_fixed import compute_staple, extract_su2_subgroup, project_to_su3
    
    device = torch.device(device)
    dims = (4, 4, 4, 4)
    
    # Slightly disordered config
    U = torch.eye(3, dtype=torch.complex64, device=device)
    U = U.view(1, 1, 1, 1, 1, 3, 3).expand(dims[0], dims[1], dims[2], dims[3], 4, 3, 3).clone()
    
    for t in range(dims[0]):
        for x in range(dims[1]):
            for y in range(dims[2]):
                for z in range(dims[3]):
                    for mu in range(4):
                        noise = 0.3 * torch.randn(3, 3, dtype=torch.complex64, device=device)
                        U[t, x, y, z, mu] = project_to_su3(U[t, x, y, z, mu] + noise)
    
    print("Comparing k values for various W_su2 blocks:")
    print()
    
    for trial in range(5):
        site = (trial % dims[0], (trial*2) % dims[1], 0, 0)
        mu = trial % 4
        
        link = U[site[0], site[1], site[2], site[3], mu]
        staple = compute_staple(U, site, mu)
        W = link @ staple
        
        for subgroup in range(3):
            W_su2 = extract_su2_subgroup(W, subgroup)
            
            # Det-based k (current)
            det_W = W_su2[0, 0] * W_su2[1, 1] - W_su2[0, 1] * W_su2[1, 0]
            k_det = torch.sqrt(torch.abs(det_W) + 1e-10).real.item()
            
            # Frobenius-based k (GPT suggestion)
            k_frob = torch.sqrt(0.5 * torch.trace(W_su2.conj().T @ W_su2).real + 1e-10).item()
            
            print(f"  Trial {trial} subgroup {subgroup}: k_det={k_det:.4f}  k_frob={k_frob:.4f}  ratio={k_det/k_frob:.4f}")
    
    print()
    return


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    # Test 1: SU(2) sampler
    test_su2_sampler(device=device)
    
    # Test 2: Before/after trace
    test_before_after_trace(device=device)
    
    # Test 3: k normalization
    test_k_normalization(device=device)
    
    print("="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
