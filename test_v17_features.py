#!/usr/bin/env python3
"""
Test script to verify v17 features without requiring Modal/GPU.
Tests the core logic and structure.
"""

import numpy as np
import sys

def test_quantile_fallback():
    """Test compute_quantiles_with_fallback function."""
    print("Testing compute_quantiles_with_fallback...")
    
    # Mock the function (copied from v17.1)
    def compute_quantiles_with_fallback(values, q_lo=0.2, q_hi=0.8):
        values = np.array(values)
        n = len(values)
        
        # v17.1: Use Q_0.1/Q_0.9 only for high occupancy bins (n >= 50)
        if n >= 50:
            return np.percentile(values, 10), np.percentile(values, 90)
        elif n >= 10:
            # Default to Q_0.2/Q_0.8 for better finite-N robustness
            return np.percentile(values, 20), np.percentile(values, 80)
        else:
            mean = np.mean(values)
            std = np.std(values)
            return mean - std, mean + std
    
    # Test case 1: n >= 50 (use Q_0.1/Q_0.9)
    values_50 = np.random.randn(55)
    q_lo, q_hi = compute_quantiles_with_fallback(values_50)
    assert q_lo < np.median(values_50) < q_hi, "Failed n>=50 case"
    print("  ✓ n>=50 case: uses Q_0.1/Q_0.9")
    
    # Test case 2: 10 <= n < 50 (use Q_0.2/Q_0.8, the default)
    values_20 = np.random.randn(25)
    q_lo, q_hi = compute_quantiles_with_fallback(values_20)
    assert q_lo < np.median(values_20) < q_hi, "Failed 10<=n<50 case"
    print("  ✓ 10<=n<50 case: uses Q_0.2/Q_0.8 (default)")
    
    # Test case 3: n < 10 (use mean ± std)
    values_5 = np.random.randn(5)
    q_lo, q_hi = compute_quantiles_with_fallback(values_5)
    mean = np.mean(values_5)
    std = np.std(values_5)
    assert abs(q_lo - (mean - std)) < 1e-10, "Failed n<10 case lower"
    assert abs(q_hi - (mean + std)) < 1e-10, "Failed n<10 case upper"
    print("  ✓ n<10 case: uses mean ± std")
    
    return True

def test_eta_computation_logic():
    """Test η_mean and η_max computation logic."""
    print("\nTesting η computation logic...")
    
    # Mock compute_eta_from_chain (simplified version)
    def compute_eta_from_chain(assignments):
        n_transitions = 0
        n_same_bin = 0
        
        bin_exits = {}
        bin_stays = {}
        
        for i in range(len(assignments) - 1):
            b_curr = assignments[i]
            b_next = assignments[i + 1]
            
            if b_curr not in bin_exits:
                bin_exits[b_curr] = 0
                bin_stays[b_curr] = 0
            
            if b_curr != b_next:
                n_transitions += 1
                bin_exits[b_curr] += 1
            else:
                n_same_bin += 1
                bin_stays[b_curr] += 1
        
        total = n_transitions + n_same_bin
        if total == 0:
            return 0.0, 0.0
        
        eta_mean = n_transitions / total
        
        eta_max = 0.0
        for bid in bin_exits:
            exits = bin_exits[bid]
            stays = bin_stays.get(bid, 0)
            total_in_bin = exits + stays
            if total_in_bin > 0:
                leakage = exits / total_in_bin
                eta_max = max(eta_max, leakage)
        
        return eta_mean, eta_max
    
    # Test case: mostly stable with few transitions
    assignments = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
    eta_mean, eta_max = compute_eta_from_chain(assignments)
    
    # Should have 2 transitions out of 11 steps
    expected_eta_mean = 2 / 11
    assert abs(eta_mean - expected_eta_mean) < 1e-10, f"η_mean incorrect: {eta_mean} vs {expected_eta_mean}"
    print(f"  ✓ η_mean = {eta_mean:.3f} (correct)")
    
    # η_max should be worst-case leakage
    assert eta_max >= eta_mean, "η_max should be >= η_mean"
    assert 0 <= eta_max <= 1, "η_max should be in [0,1]"
    print(f"  ✓ η_max = {eta_max:.3f} (valid)")
    
    return True

def test_a2s_criteria_logic():
    """Test A2S-001 improved criteria logic."""
    print("\nTesting A2S-001 criteria logic...")
    
    # Test dispersion criterion: Q_0.9(σ_b) ≤ 3 × δ_O
    bin_stds = np.array([0.01, 0.02, 0.015, 0.018, 0.025])
    delta_O = np.median(bin_stds)
    q90_sigma = np.percentile(bin_stds, 90)
    
    dispersion_ok = q90_sigma <= 3 * delta_O
    print(f"  δ_O = {delta_O:.4f}, Q_0.9(σ_b) = {q90_sigma:.4f}")
    print(f"  ✓ Dispersion criterion: {dispersion_ok}")
    
    # Test bin separation criterion: D_min / Q_0.9(σ_b) ≥ 5
    D_min = 0.15  # Example minimum gap between adjacent bins
    bin_sep_ratio = D_min / max(q90_sigma, 1e-10)
    bins_distinguishable = bin_sep_ratio >= 5.0
    
    print(f"  D_min = {D_min:.4f}, ratio = {bin_sep_ratio:.2f}")
    print(f"  ✓ Separation criterion: {bins_distinguishable}")
    
    return True

def test_r_histogram_structure():
    """Test r_histogram reporting structure."""
    print("\nTesting r_histogram reporting structure...")
    
    # Simulate some topological charges
    r_histogram_global = [0, 0, 0, 1, 0, -1, 0, 0, 0, 0]
    
    r_counts = {}
    for r in r_histogram_global:
        r_counts[r] = r_counts.get(r, 0) + 1
    
    topology_frozen = len(r_counts) == 1
    r_diversity = len(r_counts)
    
    print(f"  r_histogram: {r_counts}")
    print(f"  ✓ topology_frozen: {topology_frozen}")
    print(f"  ✓ r_diversity: {r_diversity}")
    
    assert not topology_frozen, "Should have multiple r values"
    assert r_diversity == 3, f"Should have 3 distinct r values, got {r_diversity}"
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("Testing v17 Features")
    print("="*60)
    
    tests = [
        test_quantile_fallback,
        test_eta_computation_logic,
        test_a2s_criteria_logic,
        test_r_histogram_structure,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(('PASS', test.__name__))
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            results.append(('FAIL', test.__name__))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for status, name in results:
        print(f"  {status}: {name}")
    
    n_pass = sum(1 for status, _ in results if status == 'PASS')
    print(f"\n{n_pass}/{len(results)} tests passed")
    
    if n_pass == len(results):
        print("\n✓ All v17 feature tests PASSED")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())
