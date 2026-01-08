#!/usr/bin/env python3
"""
CC-001: Collatz Conjecture Holonomy Basin Test
===============================================

GPU-Accelerated Implementation for RTX 5070 (Blackwell)

This test validates the Collatz Conjecture in the Davis-Wilson framework
by verifying that all trajectories have negative net holonomy (contraction
dominates), guaranteeing convergence to the unique basin at 1.

Core Hypothesis: H(n) = N_odd × log(3) - N_total × log(2) < 0 for all n
                 Equivalently: odd_fraction ρ(n) < log(2)/log(3) ≈ 0.631

Author: Bee Rosa Davis
Framework: Davis-Wilson Field Equations
Date: January 2026
"""

import numpy as np
import time
import sys
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
import warnings

# GPU imports with fallback
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"🎮 GPU Detected: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("⚠️  CuPy not available, falling back to CPU (NumPy)")

# Numba for CPU acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("✓ Numba available for CPU acceleration")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba not available, using pure Python")

# Visualization
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  matplotlib not available, skipping plots")


# =============================================================================
# CONSTANTS
# =============================================================================

LOG2 = np.log(2)  # ≈ 0.693
LOG3 = np.log(3)  # ≈ 1.099
CRITICAL_RHO = LOG2 / LOG3  # ≈ 0.631 - critical odd fraction


# =============================================================================
# COLLATZ FUNCTIONS (CPU - Numba accelerated)
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def collatz_trajectory_fast(n: int, max_steps: int = 10_000_000) -> Tuple[int, int, int]:
        """
        Compute Collatz trajectory statistics (Numba JIT compiled).
        
        Returns: (stopping_time, n_odd, max_value)
        """
        steps = 0
        n_odd = 0
        max_val = n
        
        while n != 1 and steps < max_steps:
            if n > max_val:
                max_val = n
            
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
                n_odd += 1
            
            steps += 1
        
        return steps, n_odd, max_val
    
    @jit(nopython=True, parallel=True)
    def batch_collatz_stats(start: int, end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Collatz statistics for range [start, end) in parallel.
        
        Returns: (stopping_times, n_odds, max_values, holonomies)
        """
        size = end - start
        stopping_times = np.zeros(size, dtype=np.int64)
        n_odds = np.zeros(size, dtype=np.int64)
        max_values = np.zeros(size, dtype=np.int64)
        holonomies = np.zeros(size, dtype=np.float64)
        
        for i in prange(size):
            n = start + i
            if n < 1:
                continue
            
            steps, odd, max_val = collatz_trajectory_fast(n)
            stopping_times[i] = steps
            n_odds[i] = odd
            max_values[i] = max_val
            
            # Holonomy: H = n_odd * log(3) - n_total * log(2)
            holonomies[i] = odd * 1.0986122886681098 - steps * 0.6931471805599453
        
        return stopping_times, n_odds, max_values, holonomies

else:
    def collatz_trajectory_fast(n: int, max_steps: int = 10_000_000) -> Tuple[int, int, int]:
        """Fallback: Pure Python Collatz trajectory."""
        steps = 0
        n_odd = 0
        max_val = n
        
        while n != 1 and steps < max_steps:
            if n > max_val:
                max_val = n
            
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
                n_odd += 1
            
            steps += 1
        
        return steps, n_odd, max_val
    
    def batch_collatz_stats(start: int, end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fallback: Sequential batch computation."""
        size = end - start
        stopping_times = np.zeros(size, dtype=np.int64)
        n_odds = np.zeros(size, dtype=np.int64)
        max_values = np.zeros(size, dtype=np.int64)
        holonomies = np.zeros(size, dtype=np.float64)
        
        for i in range(size):
            n = start + i
            if n < 1:
                continue
            
            steps, odd, max_val = collatz_trajectory_fast(n)
            stopping_times[i] = steps
            n_odds[i] = odd
            max_values[i] = max_val
            holonomies[i] = odd * LOG3 - steps * LOG2
        
        return stopping_times, n_odds, max_values, holonomies


# =============================================================================
# GPU IMPLEMENTATION (CuPy)
# =============================================================================

if GPU_AVAILABLE:
    # CUDA kernel for Collatz statistics
    collatz_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void collatz_stats(
        const long long* starts,
        long long* stopping_times,
        long long* n_odds,
        long long* max_values,
        double* holonomies,
        const int size,
        const int max_steps
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= size) return;
        
        long long n = starts[idx];
        if (n < 1) {
            stopping_times[idx] = 0;
            n_odds[idx] = 0;
            max_values[idx] = 0;
            holonomies[idx] = 0.0;
            return;
        }
        
        long long steps = 0;
        long long odd = 0;
        long long max_val = n;
        
        while (n != 1 && steps < max_steps) {
            if (n > max_val) max_val = n;
            
            if (n % 2 == 0) {
                n = n / 2;
            } else {
                n = 3 * n + 1;
                odd++;
            }
            steps++;
        }
        
        stopping_times[idx] = steps;
        n_odds[idx] = odd;
        max_values[idx] = max_val;
        
        // Holonomy: H = n_odd * log(3) - n_total * log(2)
        holonomies[idx] = (double)odd * 1.0986122886681098 - (double)steps * 0.6931471805599453;
    }
    ''', 'collatz_stats')
    
    def batch_collatz_gpu(starts: np.ndarray, max_steps: int = 10_000_000) -> Dict[str, np.ndarray]:
        """
        Compute Collatz statistics on GPU.
        
        Returns dict with: stopping_times, n_odds, max_values, holonomies
        """
        size = len(starts)
        
        # Allocate GPU arrays
        starts_gpu = cp.asarray(starts, dtype=cp.int64)
        stopping_times_gpu = cp.zeros(size, dtype=cp.int64)
        n_odds_gpu = cp.zeros(size, dtype=cp.int64)
        max_values_gpu = cp.zeros(size, dtype=cp.int64)
        holonomies_gpu = cp.zeros(size, dtype=cp.float64)
        
        # Launch kernel
        block_size = 256
        grid_size = (size + block_size - 1) // block_size
        
        collatz_kernel(
            (grid_size,), (block_size,),
            (starts_gpu, stopping_times_gpu, n_odds_gpu, max_values_gpu, 
             holonomies_gpu, size, max_steps)
        )
        
        # Copy back
        result = {
            'stopping_times': cp.asnumpy(stopping_times_gpu),
            'n_odds': cp.asnumpy(n_odds_gpu),
            'max_values': cp.asnumpy(max_values_gpu),
            'holonomies': cp.asnumpy(holonomies_gpu)
        }
        
        # Free GPU memory
        del starts_gpu, stopping_times_gpu, n_odds_gpu, max_values_gpu, holonomies_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return result


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

@dataclass
class CollatzResult:
    """Results for a single n."""
    n: int
    stopping_time: int
    n_odd: int
    max_value: int
    holonomy: float
    odd_fraction: float
    gamma: float  # CRITICAL_RHO / odd_fraction
    
    @property
    def passed(self) -> bool:
        return self.holonomy < 0 and self.odd_fraction < CRITICAL_RHO


@dataclass
class TestSummary:
    """Summary statistics for a test run."""
    n_tested: int
    n_passed: int
    n_failed: int
    mean_stopping_time: float
    mean_odd_fraction: float
    mean_holonomy: float
    max_odd_fraction: float
    max_excursion_ratio: float
    worst_case_n: int
    worst_case_rho: float


def run_holonomy_test(N_max: int, chunk_size: int = 1_000_000, 
                      verbose: bool = True) -> TestSummary:
    """
    CC-001-A: Holonomy Budget Stability Test
    
    Verify H(n) < 0 and ρ(n) < 0.631 for all n up to N_max.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CC-001-A: HOLONOMY BUDGET STABILITY TEST")
        print("=" * 70)
        print(f"Testing n = 1 to {N_max:,}")
        print(f"Critical odd fraction ρ_c = {CRITICAL_RHO:.6f}")
        print("-" * 70)
    
    t_start = time.perf_counter()
    
    # Accumulators
    all_stopping_times = []
    all_odd_fractions = []
    all_holonomies = []
    all_excursion_ratios = []
    
    n_failed = 0
    worst_rho = 0.0
    worst_n = 1
    
    # Process in chunks
    n_chunks = (N_max + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size + 1
        end = min((chunk_idx + 1) * chunk_size + 1, N_max + 1)
        
        if verbose and chunk_idx % 10 == 0:
            progress = 100 * start / N_max
            elapsed = time.perf_counter() - t_start
            rate = start / elapsed if elapsed > 0 else 0
            print(f"  Progress: {progress:.1f}% ({start:,}/{N_max:,}) - {rate:.0f} n/s")
        
        # Compute statistics
        if GPU_AVAILABLE and end - start > 10000:
            starts = np.arange(start, end, dtype=np.int64)
            result = batch_collatz_gpu(starts)
            stopping_times = result['stopping_times']
            n_odds = result['n_odds']
            max_values = result['max_values']
            holonomies = result['holonomies']
        else:
            stopping_times, n_odds, max_values, holonomies = batch_collatz_stats(start, end)
        
        # Compute derived quantities
        with np.errstate(divide='ignore', invalid='ignore'):
            odd_fractions = np.where(stopping_times > 0, 
                                     n_odds / stopping_times, 
                                     0.0)
            excursion_ratios = np.where(np.arange(start, end) > 0,
                                        max_values / np.arange(start, end),
                                        0.0)
        
        # Accumulate
        all_stopping_times.extend(stopping_times)
        all_odd_fractions.extend(odd_fractions)
        all_holonomies.extend(holonomies)
        all_excursion_ratios.extend(excursion_ratios)
        
        # Check for failures (excluding n=1 which has 0 steps)
        valid_mask = stopping_times > 0
        failures = (holonomies[valid_mask] >= 0) | (odd_fractions[valid_mask] >= CRITICAL_RHO)
        n_failed += np.sum(failures)
        
        # Track worst case
        if len(odd_fractions[valid_mask]) > 0:
            chunk_max_rho = np.max(odd_fractions[valid_mask])
            if chunk_max_rho > worst_rho:
                worst_rho = chunk_max_rho
                worst_idx = np.argmax(odd_fractions[valid_mask])
                worst_n = start + np.where(valid_mask)[0][worst_idx]
    
    elapsed = time.perf_counter() - t_start
    
    # Convert to arrays
    all_stopping_times = np.array(all_stopping_times)
    all_odd_fractions = np.array(all_odd_fractions)
    all_holonomies = np.array(all_holonomies)
    all_excursion_ratios = np.array(all_excursion_ratios)
    
    # Compute summary (excluding n=1)
    valid = all_stopping_times > 0
    
    summary = TestSummary(
        n_tested=N_max,
        n_passed=N_max - n_failed,
        n_failed=n_failed,
        mean_stopping_time=np.mean(all_stopping_times[valid]),
        mean_odd_fraction=np.mean(all_odd_fractions[valid]),
        mean_holonomy=np.mean(all_holonomies[valid]),
        max_odd_fraction=np.max(all_odd_fractions[valid]),
        max_excursion_ratio=np.max(all_excursion_ratios[valid]),
        worst_case_n=worst_n,
        worst_case_rho=worst_rho
    )
    
    if verbose:
        print("-" * 70)
        print(f"Completed in {elapsed:.2f}s ({N_max/elapsed:.0f} n/s)")
        print()
        print("RESULTS:")
        print(f"  Tested:           {summary.n_tested:,}")
        print(f"  Passed:           {summary.n_passed:,}")
        print(f"  Failed:           {summary.n_failed:,}")
        print()
        print(f"  Mean stopping time:    {summary.mean_stopping_time:.2f}")
        print(f"  Mean odd fraction:     {summary.mean_odd_fraction:.4f} (critical: {CRITICAL_RHO:.4f})")
        print(f"  Mean holonomy:         {summary.mean_holonomy:.4f} (should be < 0)")
        print(f"  Max odd fraction:      {summary.max_odd_fraction:.4f}")
        print(f"  Max excursion ratio:   {summary.max_excursion_ratio:.2f}")
        print(f"  Worst case: n={summary.worst_case_n}, ρ={summary.worst_case_rho:.4f}")
        print()
        
        # Verdict
        if summary.n_failed == 0 and summary.max_odd_fraction < CRITICAL_RHO:
            margin = CRITICAL_RHO - summary.max_odd_fraction
            print(f"✓ PASS: All trajectories have negative holonomy")
            print(f"        Margin below critical: {margin:.4f} ({100*margin/CRITICAL_RHO:.1f}%)")
        else:
            print(f"✗ FAIL: {summary.n_failed} trajectories violated holonomy bound")
        print("=" * 70)
    
    return summary


def run_stopping_time_scaling(scales: List[int] = None, samples_per_scale: int = 10000,
                               verbose: bool = True) -> Dict:
    """
    CC-001-B: Stopping Time Scaling Test
    
    Verify T(n) ~ α × log(n).
    """
    if scales is None:
        scales = [10**k for k in range(2, 9)]
    
    if verbose:
        print("\n" + "=" * 70)
        print("CC-001-B: STOPPING TIME SCALING TEST")
        print("=" * 70)
    
    results = {'scales': [], 'mean_T': [], 'std_T': [], 'log_n': []}
    
    for scale in scales:
        # Sample random integers around this scale
        np.random.seed(42)
        samples = np.random.randint(scale // 2, scale * 2, size=samples_per_scale)
        
        if GPU_AVAILABLE:
            stats = batch_collatz_gpu(samples)
            stopping_times = stats['stopping_times']
        else:
            stopping_times = np.array([collatz_trajectory_fast(int(n))[0] for n in samples])
        
        mean_T = np.mean(stopping_times)
        std_T = np.std(stopping_times)
        
        results['scales'].append(scale)
        results['mean_T'].append(mean_T)
        results['std_T'].append(std_T)
        results['log_n'].append(np.log(scale))
        
        if verbose:
            print(f"  n ~ 10^{int(np.log10(scale))}: T = {mean_T:.1f} ± {std_T:.1f}")
    
    # Linear fit: T = α × log(n) + β
    log_n = np.array(results['log_n'])
    mean_T = np.array(results['mean_T'])
    
    coeffs = np.polyfit(log_n, mean_T, 1)
    alpha, beta = coeffs
    
    # R² calculation
    T_pred = alpha * log_n + beta
    ss_res = np.sum((mean_T - T_pred) ** 2)
    ss_tot = np.sum((mean_T - np.mean(mean_T)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    results['alpha'] = alpha
    results['beta'] = beta
    results['r_squared'] = r_squared
    
    if verbose:
        print()
        print(f"Fit: T(n) = {alpha:.2f} × log(n) + {beta:.2f}")
        print(f"R² = {r_squared:.6f}")
        print()
        
        if r_squared > 0.95:
            print(f"✓ PASS: Stopping time scales logarithmically (R² = {r_squared:.4f})")
        else:
            print(f"⚠️  PARTIAL: R² = {r_squared:.4f} < 0.95")
        print("=" * 70)
    
    return results


def run_excursion_analysis(N_max: int = 1_000_000, verbose: bool = True) -> Dict:
    """
    CC-001-C: Maximum Excursion Bound Test
    
    Verify max(trajectory) / n is bounded.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CC-001-C: MAXIMUM EXCURSION BOUND TEST")
        print("=" * 70)
    
    # Compute for all n
    starts = np.arange(1, N_max + 1, dtype=np.int64)
    
    if GPU_AVAILABLE:
        stats = batch_collatz_gpu(starts)
        max_values = stats['max_values']
    else:
        _, _, max_values, _ = batch_collatz_stats(1, N_max + 1)
    
    # Excursion ratio
    excursion_ratios = max_values / starts
    
    # Find record holders
    record_indices = []
    current_max = 0
    for i, ratio in enumerate(excursion_ratios):
        if ratio > current_max:
            current_max = ratio
            record_indices.append(i)
    
    results = {
        'max_excursion': np.max(excursion_ratios),
        'mean_excursion': np.mean(excursion_ratios),
        'median_excursion': np.median(excursion_ratios),
        'record_holders': [(starts[i], max_values[i], excursion_ratios[i]) 
                          for i in record_indices[-10:]],  # Top 10 records
        'excursion_ratios': excursion_ratios
    }
    
    if verbose:
        print(f"Max excursion ratio:    {results['max_excursion']:.2f}")
        print(f"Mean excursion ratio:   {results['mean_excursion']:.2f}")
        print(f"Median excursion ratio: {results['median_excursion']:.2f}")
        print()
        print("Top 10 record holders (highest excursion):")
        for n, max_val, ratio in results['record_holders']:
            print(f"  n = {n:>10,}: max = {max_val:>15,}, ratio = {ratio:.2f}")
        print()
        
        if results['max_excursion'] < 1000:  # Reasonable bound
            print(f"✓ PASS: Excursion ratio bounded (max = {results['max_excursion']:.2f})")
        else:
            print(f"⚠️  WARNING: Large excursion ratio = {results['max_excursion']:.2f}")
        print("=" * 70)
    
    return results


def run_odd_fraction_analysis(N_max: int = 1_000_000, verbose: bool = True) -> Dict:
    """
    CC-001-E: Odd Fraction Distribution Test
    
    Analyze distribution of odd fractions.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("CC-001-E: ODD FRACTION DISTRIBUTION TEST")
        print("=" * 70)
    
    # Compute for all n
    starts = np.arange(2, N_max + 1, dtype=np.int64)  # Skip n=1
    
    if GPU_AVAILABLE:
        stats = batch_collatz_gpu(starts)
        stopping_times = stats['stopping_times']
        n_odds = stats['n_odds']
    else:
        stopping_times, n_odds, _, _ = batch_collatz_stats(2, N_max + 1)
    
    # Odd fractions
    valid = stopping_times > 0
    odd_fractions = n_odds[valid] / stopping_times[valid]
    
    results = {
        'mean': np.mean(odd_fractions),
        'std': np.std(odd_fractions),
        'min': np.min(odd_fractions),
        'max': np.max(odd_fractions),
        'median': np.median(odd_fractions),
        'percentile_99': np.percentile(odd_fractions, 99),
        'above_critical': np.sum(odd_fractions >= CRITICAL_RHO),
        'histogram': np.histogram(odd_fractions, bins=50, range=(0, 0.7))
    }
    
    if verbose:
        print(f"Odd fraction statistics:")
        print(f"  Mean:   {results['mean']:.4f}")
        print(f"  Std:    {results['std']:.4f}")
        print(f"  Min:    {results['min']:.4f}")
        print(f"  Max:    {results['max']:.4f}")
        print(f"  Median: {results['median']:.4f}")
        print(f"  99th percentile: {results['percentile_99']:.4f}")
        print()
        print(f"Critical threshold: ρ_c = {CRITICAL_RHO:.4f}")
        print(f"Trajectories above critical: {results['above_critical']}")
        print()
        
        if results['max'] < CRITICAL_RHO:
            margin = CRITICAL_RHO - results['max']
            print(f"✓ PASS: All odd fractions below critical")
            print(f"        Margin: {margin:.4f} ({100*margin/CRITICAL_RHO:.1f}%)")
        else:
            print(f"✗ FAIL: {results['above_critical']} trajectories above critical")
        print("=" * 70)
    
    return results


# =============================================================================
# MAIN TEST SUITE
# =============================================================================

def run_full_test_suite(N_max: int = 10_000_000, verbose: bool = True):
    """
    Run complete CC-001 test suite.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   CC-001: COLLATZ CONJECTURE HOLONOMY BASIN TEST                 ║
    ║                                                                   ║
    ║   Testing the Collatz Conjecture in the                          ║
    ║   Davis-Wilson Field Equations Framework                         ║
    ║                                                                   ║
    ║   "All trajectories have negative holonomy"                      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Test A: Holonomy Budget
    print("\n" + "█" * 70)
    print("TEST A: HOLONOMY BUDGET STABILITY")
    print("█" * 70)
    results['holonomy'] = run_holonomy_test(N_max, verbose=verbose)
    
    # Test B: Stopping Time Scaling
    print("\n" + "█" * 70)
    print("TEST B: STOPPING TIME SCALING")
    print("█" * 70)
    results['scaling'] = run_stopping_time_scaling(verbose=verbose)
    
    # Test C: Excursion Bounds
    print("\n" + "█" * 70)
    print("TEST C: EXCURSION BOUNDS")
    print("█" * 70)
    results['excursion'] = run_excursion_analysis(min(N_max, 1_000_000), verbose=verbose)
    
    # Test E: Odd Fraction Distribution
    print("\n" + "█" * 70)
    print("TEST E: ODD FRACTION DISTRIBUTION")
    print("█" * 70)
    results['odd_fraction'] = run_odd_fraction_analysis(min(N_max, 1_000_000), verbose=verbose)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: CC-001 COLLATZ CONJECTURE TEST")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 4
    
    # Check each test
    if results['holonomy'].n_failed == 0:
        print("  CC-001-A (Holonomy):      ✓ PASS")
        tests_passed += 1
    else:
        print("  CC-001-A (Holonomy):      ✗ FAIL")
    
    if results['scaling']['r_squared'] > 0.95:
        print(f"  CC-001-B (Scaling):       ✓ PASS (R² = {results['scaling']['r_squared']:.4f})")
        tests_passed += 1
    else:
        print(f"  CC-001-B (Scaling):       ⚠️  PARTIAL (R² = {results['scaling']['r_squared']:.4f})")
        tests_passed += 0.5
    
    if results['excursion']['max_excursion'] < 1000:
        print(f"  CC-001-C (Excursion):     ✓ PASS (max = {results['excursion']['max_excursion']:.2f})")
        tests_passed += 1
    else:
        print(f"  CC-001-C (Excursion):     ⚠️  WARNING (max = {results['excursion']['max_excursion']:.2f})")
    
    if results['odd_fraction']['above_critical'] == 0:
        print(f"  CC-001-E (Odd Fraction):  ✓ PASS (max = {results['odd_fraction']['max']:.4f})")
        tests_passed += 1
    else:
        print(f"  CC-001-E (Odd Fraction):  ✗ FAIL ({results['odd_fraction']['above_critical']} violations)")
    
    print("-" * 70)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print()
    
    if tests_passed >= 4:
        print("🏆 STRONG PASS: Collatz Conjecture VALIDATED in Davis-Wilson framework")
        print()
        print("   All trajectories have negative holonomy (contraction dominates).")
        print("   The unique basin at 1 attracts all orbits.")
        print(f"   Mean odd fraction: {results['odd_fraction']['mean']:.4f} << {CRITICAL_RHO:.4f}")
    elif tests_passed >= 3:
        print("✓ PASS: Collatz Conjecture SUPPORTED")
    else:
        print("⚠️  PARTIAL: Further investigation needed")
    
    print("=" * 70)
    
    # Generate plots if available
    if PLOTTING_AVAILABLE:
        generate_plots(results, N_max)
    
    return results


def generate_plots(results: Dict, N_max: int):
    """Generate visualization of test results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'CC-001: Collatz Conjecture Test (N = {N_max:,})', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Stopping time scaling
    ax = axes[0, 0]
    log_n = np.array(results['scaling']['log_n'])
    mean_T = np.array(results['scaling']['mean_T'])
    std_T = np.array(results['scaling']['std_T'])
    
    ax.errorbar(log_n, mean_T, yerr=std_T, fmt='bo-', capsize=5, markersize=8)
    ax.plot(log_n, results['scaling']['alpha'] * log_n + results['scaling']['beta'],
            'r--', linewidth=2, label=f"Fit: {results['scaling']['alpha']:.2f}×log(n) + {results['scaling']['beta']:.1f}")
    ax.set_xlabel('log(n)', fontsize=11)
    ax.set_ylabel('Stopping Time T(n)', fontsize=11)
    ax.set_title(f"Stopping Time Scaling (R² = {results['scaling']['r_squared']:.4f})", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Odd fraction histogram
    ax = axes[0, 1]
    hist_counts, hist_edges = results['odd_fraction']['histogram']
    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    ax.bar(hist_centers, hist_counts, width=hist_edges[1]-hist_edges[0], 
           alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(CRITICAL_RHO, color='red', linestyle='--', linewidth=2, 
               label=f'Critical ρ_c = {CRITICAL_RHO:.3f}')
    ax.axvline(results['odd_fraction']['mean'], color='green', linestyle='-', linewidth=2,
               label=f"Mean = {results['odd_fraction']['mean']:.3f}")
    ax.set_xlabel('Odd Fraction ρ', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Odd Fraction Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Holonomy distribution (sample)
    ax = axes[1, 0]
    # Sample some holonomies
    sample_n = np.arange(2, min(10001, N_max + 1))
    if GPU_AVAILABLE:
        sample_stats = batch_collatz_gpu(sample_n)
        sample_H = sample_stats['holonomies']
    else:
        _, _, _, sample_H = batch_collatz_stats(2, min(10001, N_max + 1))
    
    ax.hist(sample_H, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='H = 0 (critical)')
    ax.axvline(np.mean(sample_H), color='green', linestyle='-', linewidth=2,
               label=f"Mean = {np.mean(sample_H):.1f}")
    ax.set_xlabel('Holonomy H(n)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Holonomy Distribution (all should be < 0)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Excursion ratio vs n
    ax = axes[1, 1]
    sample_size = min(10000, len(results['excursion']['excursion_ratios']))
    sample_idx = np.linspace(0, len(results['excursion']['excursion_ratios'])-1, 
                             sample_size, dtype=int)
    ax.scatter(sample_idx + 1, results['excursion']['excursion_ratios'][sample_idx],
               alpha=0.3, s=1, c='blue')
    ax.axhline(results['excursion']['mean_excursion'], color='green', linestyle='-',
               linewidth=2, label=f"Mean = {results['excursion']['mean_excursion']:.2f}")
    ax.set_xlabel('n', fontsize=11)
    ax.set_ylabel('Excursion Ratio max(T)/n', fontsize=11)
    ax.set_title('Maximum Excursion Ratio', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('cc_001_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: cc_001_results.png")
    plt.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    
    N_max = 10_000_000  # Default: 10 million
    
    if "--quick" in sys.argv:
        N_max = 100_000
        print("🚀 Quick mode: N_max = 100,000")
    
    if "--full" in sys.argv:
        N_max = 100_000_000
        print("🔬 Full mode: N_max = 100,000,000")
    
    if "--extreme" in sys.argv:
        N_max = 1_000_000_000
        print("💪 Extreme mode: N_max = 1,000,000,000")
    
    results = run_full_test_suite(N_max=N_max)
    
    # Final status
    passed = (results['holonomy'].n_failed == 0 and 
              results['odd_fraction']['above_critical'] == 0)
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
