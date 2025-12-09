"""
TVR C.4: Universality Analysis
==============================

Analyze multi-loop harvest data to prove the Davis Term effect
is universal across different Wilson loop geometries.

If rectification is a real vacuum property (not loop artifact),
all loop shapes should show the same J-D correlation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats


def analyze_universality(filename: str):
    """
    Analyze multi-loop harvest for universality.
    """
    print("=" * 60)
    print("TVR C.4: UNIVERSALITY ANALYSIS")
    print("Testing: Does rectification depend on loop geometry?")
    print("=" * 60)

    data = np.load(filename)
    
    # Extract metadata
    L = int(data['L'])
    beta = float(data['beta'])
    n_configs = int(data['n_configs'])
    loops = data['loops']  # Array of (R, T) pairs
    
    print(f"\nDataset: L={L}, β={beta}, N={n_configs}")
    print(f"Loop geometries: {[tuple(l) for l in loops]}")

    # Extract J and D for each loop
    results = {}
    for R, T in loops:
        key_j = f"J_{R}x{T}"
        key_d = f"D_{R}x{T}"
        J = data[key_j]
        D = data[key_d]
        
        # Compute statistics
        corr = np.corrcoef(J, D)[0, 1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(D, J)
        
        results[(R, T)] = {
            'J': J,
            'D': D,
            'corr': corr,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
        }
        
        print(f"\n{R}×{T} Loop:")
        print(f"  <J> = {J.mean():.4f} ± {J.std():.4f}")
        print(f"  <D> = {D.mean():.4f} ± {D.std():.4f}")
        print(f"  ρ(J,D) = {corr:.4f}")
        print(f"  Slope = {slope:.4f} ± {std_err:.4f}")

    # ========================================================================
    # UNIVERSALITY TEST: Are all slopes consistent?
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("UNIVERSALITY TEST")
    print("=" * 60)
    
    slopes = [results[loop]['slope'] for loop in results]
    slope_errs = [results[loop]['std_err'] for loop in results]
    
    # Weighted mean slope
    weights = 1 / np.array(slope_errs)**2
    mean_slope = np.sum(weights * slopes) / np.sum(weights)
    mean_slope_err = 1 / np.sqrt(np.sum(weights))
    
    print(f"\nSlopes by loop geometry:")
    for loop in results:
        s = results[loop]['slope']
        e = results[loop]['std_err']
        dev = abs(s - mean_slope) / e
        print(f"  {loop[0]}×{loop[1]}: {s:.4f} ± {e:.4f} ({dev:.1f}σ from mean)")
    
    print(f"\nWeighted mean slope: {mean_slope:.4f} ± {mean_slope_err:.4f}")
    
    # Chi-squared test for consistency
    chi2 = np.sum(weights * (np.array(slopes) - mean_slope)**2)
    dof = len(slopes) - 1
    p_chi2 = 1 - stats.chi2.cdf(chi2, dof)
    
    print(f"\nConsistency test:")
    print(f"  χ² = {chi2:.2f} (dof={dof})")
    print(f"  p-value = {p_chi2:.4f}")
    
    if p_chi2 > 0.05:
        print(f"  >>> PASS: Slopes are consistent (p > 0.05)")
        universal = True
    else:
        print(f"  >>> FAIL: Slopes differ significantly (p < 0.05)")
        universal = False

    # ========================================================================
    # VISUALIZATION: The Holy Trinity
    # ========================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for idx, ((R, T), res) in enumerate(results.items()):
        ax = axes[idx]
        J, D = res['J'], res['D']
        
        # Scatter plot
        ax.scatter(D, J, alpha=0.3, s=10, c=colors[idx], label=f'{R}×{T}')
        
        # Regression line
        D_line = np.linspace(D.min(), D.max(), 100)
        J_line = res['slope'] * D_line + res['intercept']
        ax.plot(D_line, J_line, 'k--', linewidth=2, 
                label=f'slope={res["slope"]:.3f}±{res["std_err"]:.3f}')
        
        ax.set_xlabel('Davis Term (D)', fontsize=12)
        ax.set_ylabel('Rectified Current (J)', fontsize=12)
        ax.set_title(f'{R}×{T} Loop (Area={R*T})\nρ={res["corr"]:.3f}', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('C.4: Universality Test - J vs D Across Loop Geometries', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save to results/figures if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results", "figures")
    if os.path.isdir(results_dir):
        outfile = os.path.join(results_dir, "tvr_universality.png")
    else:
        outfile = "tvr_universality.png"
    
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {outfile}")
    plt.close()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    corrs = [results[loop]['corr'] for loop in results]
    
    print(f"\nCorrelations: {[f'{c:.3f}' for c in corrs]}")
    print(f"Mean correlation: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")
    print(f"Slopes consistent: {'YES' if universal else 'NO'} (p={p_chi2:.4f})")
    
    if universal and all(c > 0.5 for c in corrs):
        print("\n>>> VERDICT: UNIVERSALITY CONFIRMED <<<")
        print("The J-D coupling is independent of loop geometry.")
        print("This proves the Davis Term is a genuine vacuum property,")
        print("not an artifact of the specific Wilson loop shape.")
    elif all(c > 0.3 for c in corrs):
        print("\n>>> VERDICT: PARTIAL UNIVERSALITY <<<")
        print("Correlations present in all loops but with some variation.")
    else:
        print("\n>>> VERDICT: UNIVERSALITY NOT CONFIRMED <<<")
        print("The effect appears geometry-dependent.")

    return {
        'universal': universal,
        'mean_slope': mean_slope,
        'mean_slope_err': mean_slope_err,
        'chi2': chi2,
        'p_value': p_chi2,
        'correlations': {f"{l[0]}x{l[1]}": results[l]['corr'] for l in results},
        'slopes': {f"{l[0]}x{l[1]}": results[l]['slope'] for l in results},
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "harvest_multiloop.npz"
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        print("Usage: python analyze_universality.py <harvest_multiloop.npz>")
        print("\nTo run the harvest first:")
        print("  modal run extended_capabilities/tvr_harvest_multiloop.py")
        sys.exit(1)
    
    results = analyze_universality(filename)
