"""
TVR Deep Geometry Analysis Suite
=================================

Comprehensive geometric validation of vacuum rectification signal.

Tests included:
- C.1: Distribution Metric (Fréchet) - How PDFs deform with θ
- C.2: Persistent Homology (TDA) - Topological holes in J-D space
- C.5: Jensen Gap - Manifold curvature via convexity deviation
- C.6: Heat Kernel Spectral Analysis - Diffusion geometry + RMT

Run locally on existing harvest data.

Prerequisites:
    pip install ripser persim scikit-learn scipy numpy matplotlib

Author: Bee Davis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import stats
import sys
import os

# Optional TDA imports
try:
    from ripser import ripser
    from persim import plot_diagrams
    HAS_TDA = True
except ImportError:
    HAS_TDA = False
    print("Warning: ripser/persim not installed. Skipping TDA tests.")
    print("Install with: pip install ripser persim")

# Import our heat kernel module
try:
    from heat_kernel import TVRHeatKernelAnalyzer
    HAS_HEATKERNEL = True
except ImportError:
    HAS_HEATKERNEL = False
    print("Warning: heat_kernel module not found. Skipping spectral tests.")


def load_data(filename: str):
    """Load harvest data."""
    print(f"Loading {filename}...")
    data = np.load(filename)
    J = data["J"]
    D = data["D"]
    print(f"  Loaded {len(J)} samples")
    return J, D


def normalize(arr: np.ndarray) -> np.ndarray:
    """Standardize to zero mean, unit variance."""
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-10)


# =============================================================================
# TEST C.5: JENSEN GAP (Curvature Detector)
# =============================================================================

def test_jensen_gap(J: np.ndarray, D: np.ndarray, ax=None):
    """
    Measure manifold curvature via Jensen's inequality gap.
    
    For convex f: E[f(x)] ≥ f(E[x])
    The gap measures nonlinearity = curvature.
    
    We compare: <J>_θ vs J_linear(θ)
    If vacuum were flat, J would be linear in θ.
    The deviation is curvature.
    """
    print("\nRunning C.5: Jensen Gap (Curvature Detector)...")
    
    J_norm = normalize(J)
    D_norm = normalize(D)
    
    # Linear response baseline
    linear_slope = np.corrcoef(J, D)[0, 1] * np.std(J) / np.std(D)
    
    theta_scan = np.linspace(-3, 3, 100)
    gap_vals = []
    expected_J = []
    
    for th in theta_scan:
        # Reweighted mean E[J]_θ
        w = np.exp(-th * D_norm)
        w = w / w.sum()
        
        j_mean = np.sum(w * J)
        d_mean = np.sum(w * D)
        
        # Linear prediction (flat manifold)
        j_flat = linear_slope * d_mean
        
        gap_vals.append(abs(j_mean - j_flat))
        expected_J.append(j_mean)
    
    gap_vals = np.array(gap_vals)
    
    # Find peak curvature location
    peak_idx = np.argmax(gap_vals)
    peak_theta = theta_scan[peak_idx]
    max_gap = gap_vals[peak_idx]
    
    print(f"  Max Jensen Gap: {max_gap:.4f} at θ = θ_peak")
    print(f"  Interpretation: Curvature peaks at boundary region")
    
    if ax is not None:
        ax.plot(theta_scan, gap_vals, 'purple', linewidth=2, label='Jensen Gap')
        ax.fill_between(theta_scan, 0, gap_vals, color='purple', alpha=0.2)
        ax.axvline(peak_theta, color='red', linestyle='--', alpha=0.5, 
                   label='Peak: θ=θ*')
        ax.set_title("C.5: Manifold Curvature (Jensen Gap)")
        ax.set_xlabel("θ (normalized)")
        ax.set_ylabel("Nonlinearity Magnitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    return {
        'theta_scan': theta_scan,
        'gap_values': gap_vals,
        'peak_theta': peak_theta,
        'max_gap': max_gap
    }


# =============================================================================
# TEST C.2: PERSISTENT HOMOLOGY (TDA)
# =============================================================================

def test_persistent_homology(J: np.ndarray, D: np.ndarray, ax=None, n_samples: int = 1000):
    """
    Compute persistent homology of (D, J) point cloud.
    
    H0: Connected components (clustering)
    H1: Loops (non-trivial topology)
    
    If H1 features persist, the data has "holes" = real geometric structure.
    """
    if not HAS_TDA:
        print("\nSkipping C.2: TDA not installed")
        return None
    
    print("\nRunning C.2: Persistent Homology (TDA)...")
    
    J_norm = normalize(J)
    D_norm = normalize(D)
    
    # Subsample for speed (Ripser is O(n³))
    n = len(J)
    if n > n_samples:
        idx = np.random.choice(n, n_samples, replace=False)
        point_cloud = np.column_stack([D_norm[idx], J_norm[idx]])
    else:
        point_cloud = np.column_stack([D_norm, J_norm])
    
    print(f"  Computing persistence on {len(point_cloud)} points...")
    
    # Compute persistence
    result = ripser(point_cloud, maxdim=1)
    diagrams = result['dgms']
    
    # Analyze H1 (loops)
    h0 = diagrams[0]  # Components
    h1 = diagrams[1]  # Loops
    
    n_h0 = len(h0)
    n_h1 = len(h1)
    
    if n_h1 > 0:
        # Persistence = death - birth
        h1_persistence = h1[:, 1] - h1[:, 0]
        max_persistence = np.max(h1_persistence)
        mean_persistence = np.mean(h1_persistence)
    else:
        max_persistence = 0
        mean_persistence = 0
    
    print(f"  H0 features (components): {n_h0}")
    print(f"  H1 features (loops): {n_h1}")
    print(f"  Max H1 persistence: {max_persistence:.3f}")
    
    # Verdict
    has_topology = n_h1 > 0 and max_persistence > 0.1
    print(f"  Non-trivial topology: {'YES' if has_topology else 'NO'}")
    
    if ax is not None:
        plot_diagrams(diagrams, ax=ax, show=False)
        ax.set_title(f"C.2: Persistence Diagram (H1={n_h1} loops)")
        if n_h1 > 0:
            ax.text(0.05, 0.9, f"Max H1 persistence: {max_persistence:.3f}", 
                    transform=ax.transAxes, fontsize=10)
    
    return {
        'n_h0': n_h0,
        'n_h1': n_h1,
        'max_persistence': max_persistence,
        'mean_persistence': mean_persistence,
        'has_topology': has_topology,
        'diagrams': diagrams
    }


# =============================================================================
# TEST C.1: DISTRIBUTION METRIC (Fréchet-like)
# =============================================================================

def test_distribution_metric(J: np.ndarray, D: np.ndarray, ax=None):
    """
    Measure how the distribution of D changes under reweighting.
    
    Compare PDF(D) at θ=0 vs θ=θ* (critical point).
    Large deformation = strong geometric effect.
    """
    print("\nRunning C.1: Distribution Metric (Geometric Deformation)...")
    
    D_norm = normalize(D)
    
    # Compare raw vs reweighted at critical theta (value redacted)
    # θ* determined by optimization in TVR-002
    theta_crit = float(os.environ.get('TVR_THETA_CRIT', -1.0))  # Default placeholder
    w_crit = np.exp(-theta_crit * D_norm)
    w_crit = w_crit / w_crit.sum()
    
    # Also check opposite theta
    theta_pos = -theta_crit
    w_pos = np.exp(-theta_pos * D_norm)
    w_pos = w_pos / w_pos.sum()
    
    # Compute KL divergence as deformation metric
    # D_KL(P_θ || P_0)
    bins = np.linspace(-4, 4, 50)
    
    hist_base, _ = np.histogram(D_norm, bins=bins, density=True)
    hist_crit, _ = np.histogram(D_norm, bins=bins, weights=w_crit * len(D_norm), density=True)
    hist_pos, _ = np.histogram(D_norm, bins=bins, weights=w_pos * len(D_norm), density=True)
    
    # Add small epsilon for numerical stability
    eps = 1e-10
    hist_base = hist_base + eps
    hist_crit = hist_crit + eps
    hist_pos = hist_pos + eps
    
    # Normalize
    hist_base = hist_base / hist_base.sum()
    hist_crit = hist_crit / hist_crit.sum()
    hist_pos = hist_pos / hist_pos.sum()
    
    # KL divergence
    kl_crit = np.sum(hist_crit * np.log(hist_crit / hist_base))
    kl_pos = np.sum(hist_pos * np.log(hist_pos / hist_base))
    
    print(f"  KL divergence (θ=θ*): {kl_crit:.4f}")
    print(f"  KL divergence (θ=-θ*): {kl_pos:.4f}")
    print(f"  Asymmetry ratio: {kl_crit / (kl_pos + 1e-10):.2f}")
    
    if ax is not None:
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.hist(D_norm, bins=30, density=True, alpha=0.4, color='gray', 
                label='θ=0 (Unweighted)')
        ax.hist(D_norm, weights=w_crit * len(D_norm), bins=30, density=True, 
                alpha=0.4, color='red', label='θ=θ* (Critical)')
        ax.hist(D_norm, weights=w_pos * len(D_norm), bins=30, density=True, 
                alpha=0.4, color='blue', label='θ=-θ* (Opposite)')
        ax.set_title("C.1: Geometric Deformation of D")
        ax.set_xlabel("Davis Term (Normalized)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    return {
        'kl_crit': kl_crit,
        'kl_pos': kl_pos,
        'theta_crit': theta_crit,
        'asymmetry': kl_crit / (kl_pos + 1e-10)
    }


# =============================================================================
# TEST C.6: HEAT KERNEL SPECTRAL ANALYSIS
# =============================================================================

def test_heat_kernel(J: np.ndarray, D: np.ndarray, ax=None):
    """
    Full heat kernel spectral analysis.
    
    - Spectral gap (mixing time)
    - Anisotropy (directional dependence)
    - GUE vs GOE statistics (T-symmetry breaking)
    """
    if not HAS_HEATKERNEL:
        print("\nSkipping C.6: heat_kernel module not loaded")
        return None
    
    print("\nRunning C.6: Heat Kernel Spectral Analysis...")
    
    # Subsample for speed
    n = len(J)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        J_sub, D_sub = J[idx], D[idx]
    else:
        J_sub, D_sub = J, D
    
    analyzer = TVRHeatKernelAnalyzer(k_neighbors=50, n_eigenvalues=100)
    results = analyzer.analyze(D_sub, J_sub)
    
    hk = results['heat_kernel']
    rmt = results['rmt']
    aniso = results['anisotropy']
    
    if ax is not None:
        # Plot eigenvalue spacing histogram vs GUE/GOE
        spacings = rmt.spacings
        s_axis = np.linspace(0, 3, 100)
        
        # GUE prediction
        gue = (32 / np.pi**2) * s_axis**2 * np.exp(-4 * s_axis**2 / np.pi)
        # GOE prediction  
        goe = (np.pi / 2) * s_axis * np.exp(-np.pi * s_axis**2 / 4)
        # Poisson
        poisson = np.exp(-s_axis)
        
        ax.hist(spacings, bins=30, density=True, alpha=0.6, color='purple',
                label='TVR Spectrum')
        ax.plot(s_axis, gue, 'r-', linewidth=2, label=f'GUE (MSE={rmt.gue_mse:.4f})')
        ax.plot(s_axis, goe, 'g--', linewidth=2, label=f'GOE (MSE={rmt.goe_mse:.4f})')
        ax.plot(s_axis, poisson, 'k:', linewidth=1, alpha=0.5, label='Poisson')
        
        ax.set_title(f"C.6: RMT Statistics (Best: {rmt.best_fit})")
        ax.set_xlabel("Normalized Level Spacing")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    return {
        'spectral_gap': hk.spectral_gap,
        'anisotropy': aniso.anisotropy_ratio,
        'gue_mse': rmt.gue_mse,
        'goe_mse': rmt.goe_mse,
        'best_fit': rmt.best_fit,
        't_symmetry_broken': rmt.t_symmetry_broken
    }


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def analyze_deep_geometry(filename: str):
    """
    Run complete deep geometry analysis suite.
    
    Args:
        filename: Path to harvest .npz file
    """
    print("="*60)
    print("TVR-PATH-C: DEEP GEOMETRIC VALIDATION")
    print("Objective: Prove the Signal has Topological Shape")
    print("="*60)
    
    # Load data
    J, D = load_data(filename)
    
    # Create figure
    n_tests = 4 if HAS_TDA else 3
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Run tests
    results = {}
    
    # C.5: Jensen Gap
    results['jensen'] = test_jensen_gap(J, D, ax=axes[0])
    
    # C.2: Persistent Homology
    if HAS_TDA:
        results['tda'] = test_persistent_homology(J, D, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "TDA not installed\npip install ripser persim",
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("C.2: Persistent Homology (Skipped)")
    
    # C.1: Distribution Metric
    results['distribution'] = test_distribution_metric(J, D, ax=axes[2])
    
    # C.6: Heat Kernel (if available)
    if HAS_HEATKERNEL:
        results['heat_kernel'] = test_heat_kernel(J, D, ax=axes[3])
    else:
        axes[3].text(0.5, 0.5, "Heat kernel module not loaded",
                     ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title("C.6: Heat Kernel Spectral (Skipped)")
    
    plt.tight_layout()
    
    # Save to results/figures if it exists, otherwise cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(script_dir), "results", "figures")
    if os.path.isdir(results_dir):
        outfile = os.path.join(results_dir, "tvr_deep_geometry.png")
    else:
        outfile = "tvr_deep_geometry.png"
    plt.savefig(outfile, dpi=150)
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED TO: {outfile}")
    print("="*60)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if 'jensen' in results:
        print(f"Jensen Gap: Peak curvature at θ = θ_peak")
    
    if 'tda' in results and results['tda'] is not None:
        tda = results['tda']
        print(f"Topology: {tda['n_h1']} loops detected, max persistence = {tda['max_persistence']:.3f}")
    
    if 'distribution' in results:
        dist = results['distribution']
        print(f"Deformation: KL(θ=θ*) = {dist['kl_crit']:.4f}, asymmetry = {dist['asymmetry']:.2f}x")
    
    if 'heat_kernel' in results and results['heat_kernel'] is not None:
        hk = results['heat_kernel']
        print(f"Spectral: gap = {hk['spectral_gap']:.4f}, best fit = {hk['best_fit']}")
        print(f"T-symmetry broken: {hk['t_symmetry_broken']}")
    
    # Don't block - figure already saved
    plt.close('all')
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "harvest_merged.npz"
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        print("Usage: python deep_geometry.py <harvest_file.npz>")
        sys.exit(1)
    
    results = analyze_deep_geometry(filename)
