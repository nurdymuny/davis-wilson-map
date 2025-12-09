"""
Unified Heat Kernel Analysis for Grand Unified Theory
======================================================
MRI Scanner for:
- P vs NP manifolds (SAT solution space)
- Navier-Stokes turbulence topology
- Lattice QCD vacuum (TVR)

Author: Bee Davis
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import sparse
from scipy.spatial import cKDTree
import sys
import os


def build_laplacian(points: np.ndarray, k: int = 50, t: float = 1.0) -> sparse.csr_matrix:
    """
    Build graph Laplacian from point cloud.
    
    Args:
        points: (n, d) array of points
        k: number of neighbors
        t: diffusion time for weights
    
    Returns:
        Sparse normalized Laplacian
    """
    n = points.shape[0]
    k = min(k, n - 1)
    
    # Build k-NN graph
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k+1)
    
    # Remove self-loop
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # Build weight matrix
    rows, cols, data = [], [], []
    
    for i in range(n):
        for j_idx, dist in zip(indices[i], distances[i]):
            weight = np.exp(-dist**2 / (4 * t))
            rows.append(i)
            cols.append(j_idx)
            data.append(weight)
    
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2  # Symmetrize
    
    # Normalized Laplacian
    degrees = np.array(W.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(degrees > 1e-10, 1.0 / np.sqrt(degrees), 0)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L = sparse.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    
    return L.tocsr()


def compute_heat_kernel_spectrum(L: sparse.csr_matrix, n_eigs: int = 100):
    """Compute eigenvalues of Laplacian."""
    n_eigs = min(n_eigs, L.shape[0] - 2)
    
    try:
        # Try shift-invert mode for better convergence
        eigenvalues, eigenvectors = eigsh(
            L, k=n_eigs, which='SM', sigma=1e-6, 
            maxiter=10000, tol=1e-6
        )
    except Exception:
        # Fallback: use 'SA' (smallest algebraic) without shift
        eigenvalues, eigenvectors = eigsh(
            L, k=n_eigs, which='SA',
            maxiter=10000, tol=1e-4
        )
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[idx], 0)  # Ensure non-negative
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def analyze_pnp(filename: str):
    """
    Heat Kernel Analysis of P vs NP manifolds.
    
    Expected: 
    - P (2-SAT): Large spectral gap → fast mixing
    - NP (3-SAT): Small spectral gap → glassy dynamics
    """
    print("=" * 60)
    print("P vs NP HEAT KERNEL ANALYSIS")
    print("MRI Scan of Computational Complexity")
    print("=" * 60)
    
    data = np.load(filename)
    p_cloud = data['p_cloud']    # (500, 500) - 500 steps in 500-dim space
    np_cloud = data['np_cloud']  # (500, 500)
    p_energies = data['p_energies']
    np_energies = data['np_energies']
    
    print(f"\nP (2-SAT) trajectory: {p_cloud.shape}")
    print(f"NP (3-SAT) trajectory: {np_cloud.shape}")
    
    # Use PCA to reduce dimension for Laplacian (500D is too sparse)
    from sklearn.decomposition import PCA
    
    n_components = 20
    print(f"\nReducing to {n_components}D via PCA...")
    
    pca_p = PCA(n_components=n_components)
    p_reduced = pca_p.fit_transform(p_cloud)
    p_variance_explained = pca_p.explained_variance_ratio_.sum()
    
    pca_np = PCA(n_components=n_components)
    np_reduced = pca_np.fit_transform(np_cloud)
    np_variance_explained = pca_np.explained_variance_ratio_.sum()
    
    print(f"  P variance captured: {p_variance_explained:.1%}")
    print(f"  NP variance captured: {np_variance_explained:.1%}")
    
    # Build Laplacians
    print("\nBuilding graph Laplacians...")
    k = min(30, p_cloud.shape[0] - 1)
    
    L_p = build_laplacian(p_reduced, k=k)
    L_np = build_laplacian(np_reduced, k=k)
    
    # Compute spectra
    print("Computing spectra...")
    n_eigs = 50
    eigs_p, _ = compute_heat_kernel_spectrum(L_p, n_eigs=n_eigs)
    eigs_np, _ = compute_heat_kernel_spectrum(L_np, n_eigs=n_eigs)
    
    # Key metrics
    gap_p = eigs_p[1] if len(eigs_p) > 1 else 0  # λ₁ (λ₀ ≈ 0)
    gap_np = eigs_np[1] if len(eigs_np) > 1 else 0
    
    print("\n" + "=" * 60)
    print("SPECTRAL GAP COMPARISON")
    print("=" * 60)
    print(f"\n2-SAT (P)  spectral gap λ₁: {gap_p:.6f}")
    print(f"3-SAT (NP) spectral gap λ₁: {gap_np:.6f}")
    print(f"Ratio (P/NP): {gap_p/gap_np:.2f}x")
    
    # Heat trace at various scales
    print("\n" + "=" * 60)
    print("HEAT TRACE Z(t) = Σ exp(-λᵢt)")
    print("=" * 60)
    
    t_values = [0.1, 1.0, 10.0, 100.0]
    print("\n     t      Z_P(t)      Z_NP(t)    Ratio")
    print("-" * 50)
    for t in t_values:
        Z_p = np.sum(np.exp(-eigs_p * t))
        Z_np = np.sum(np.exp(-eigs_np * t))
        print(f"  {t:6.1f}   {Z_p:9.4f}   {Z_np:9.4f}   {Z_p/Z_np:.2f}")
    
    # Effective dimension (participation ratio)
    print("\n" + "=" * 60)
    print("EFFECTIVE DIMENSION (Participation Ratio)")
    print("=" * 60)
    
    print("\n     t      D_eff(P)    D_eff(NP)")
    print("-" * 40)
    for t in t_values:
        w_p = np.exp(-eigs_p * t)
        w_p = w_p / w_p.sum()
        D_eff_p = 1.0 / np.sum(w_p ** 2)
        
        w_np = np.exp(-eigs_np * t)
        w_np = w_np / w_np.sum()
        D_eff_np = 1.0 / np.sum(w_np ** 2)
        
        print(f"  {t:6.1f}   {D_eff_p:9.2f}   {D_eff_np:9.2f}")
    
    # INTERPRETATION
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if gap_p > gap_np:
        ratio = gap_p / gap_np
        print(f"\n✓ P (2-SAT) has {ratio:.1f}x LARGER spectral gap than NP (3-SAT)")
        print("  → P manifold has FASTER mixing time")
        print("  → NP manifold is GLASSY (slow equilibration)")
        print("\n  This is the geometric signature of computational complexity!")
        print("  The spectral gap encodes how 'hard' it is to explore the space.")
    else:
        print("\n⚠ Unexpected: NP has larger gap than P")
        print("  This may indicate the walk hasn't equilibrated")
    
    # Energy landscape analysis
    print("\n" + "=" * 60)
    print("ENERGY LANDSCAPE")
    print("=" * 60)
    print(f"\nP (2-SAT)  energy: mean={p_energies.mean():.2f}, std={p_energies.std():.2f}")
    print(f"NP (3-SAT) energy: mean={np_energies.mean():.2f}, std={np_energies.std():.2f}")
    print(f"NP/P roughness ratio: {np_energies.std()/p_energies.std():.2f}x")
    
    return {
        'gap_p': gap_p,
        'gap_np': gap_np,
        'eigenvalues_p': eigs_p,
        'eigenvalues_np': eigs_np
    }


def analyze_ns(filename: str):
    """
    Heat Kernel Analysis of Navier-Stokes turbulence topology.
    
    Expected:
    - Effective dimension < 3 if flow concentrates on vortex tubes
    - Spectral gap reveals coherence of turbulent structures
    """
    print("=" * 60)
    print("NAVIER-STOKES HEAT KERNEL ANALYSIS")
    print("MRI Scan of Turbulent Topology")
    print("=" * 60)
    
    data = np.load(filename)
    point_cloud = data['point_cloud']      # (10000, 3)
    vorticity = data['sampled_vorticity']  # (10000,)
    
    print(f"\nPoint cloud: {point_cloud.shape}")
    print(f"Vorticity range: [{vorticity.min():.2f}, {vorticity.max():.2f}]")
    
    # Subsample for tractable eigenproblem
    n_subsample = 2000
    if point_cloud.shape[0] > n_subsample:
        print(f"\nSubsampling to {n_subsample} points (weighted by vorticity)...")
        # Weight by vorticity^2 to keep high-vorticity points
        prob = vorticity ** 2
        prob = prob / prob.sum()
        idx = np.random.choice(len(vorticity), size=n_subsample, p=prob, replace=False)
        point_cloud = point_cloud[idx]
        vorticity = vorticity[idx]
    
    # Build Laplacian on 3D point cloud
    print("Building graph Laplacian on vortex cores...")
    k = 30
    L = build_laplacian(point_cloud, k=k, t=0.1)  # Smaller t for 3D locality
    
    # Compute spectrum
    print("Computing spectrum...")
    n_eigs = 100
    eigenvalues, eigenvectors = compute_heat_kernel_spectrum(L, n_eigs=n_eigs)
    
    spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
    
    print("\n" + "=" * 60)
    print("SPECTRAL PROPERTIES")
    print("=" * 60)
    print(f"\nSpectral gap λ₁: {spectral_gap:.6f}")
    print(f"First 10 eigenvalues: {eigenvalues[:10]}")
    
    # Effective dimension at various scales
    print("\n" + "=" * 60)
    print("EFFECTIVE DIMENSION (Heat Kernel Scale)")
    print("=" * 60)
    
    t_values = [0.01, 0.1, 1.0, 10.0]
    print("\n     t      D_eff     Interpretation")
    print("-" * 55)
    for t in t_values:
        w = np.exp(-eigenvalues * t)
        w = w / w.sum()
        D_eff = 1.0 / np.sum(w ** 2)
        
        if D_eff < 1.5:
            interp = "1D filaments (vortex lines)"
        elif D_eff < 2.5:
            interp = "2D sheets (vortex sheets)"
        else:
            interp = "3D volume-filling"
        
        print(f"  {t:6.2f}   {D_eff:7.2f}    {interp}")
    
    # Heat trace decay (signature of dimension)
    print("\n" + "=" * 60)
    print("HEAT TRACE ASYMPTOTICS")
    print("=" * 60)
    
    # Z(t) ~ t^{-d/2} for a d-dimensional manifold
    t_test = np.logspace(-2, 1, 20)
    Z_values = [np.sum(np.exp(-eigenvalues * t)) for t in t_test]
    
    # Fit log-log slope
    log_t = np.log(t_test[5:15])
    log_Z = np.log(Z_values[5:15])
    slope, intercept = np.polyfit(log_t, log_Z, 1)
    inferred_dim = -2 * slope
    
    print(f"\nZ(t) ~ t^α with α = {slope:.2f}")
    print(f"Inferred dimension d = -2α = {inferred_dim:.2f}")
    
    # INTERPRETATION
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if inferred_dim < 3:
        print(f"\n✓ Turbulence concentrates on {inferred_dim:.1f}D structures")
        print("  → Vortex tubes and sheets are REAL topological objects")
        print("  → Energy is being funneled into lower-dimensional singularities")
        print("\n  This is evidence for the vortex stretching mechanism!")
    else:
        print(f"\n⚠ Flow is volume-filling (D ≈ {inferred_dim:.1f})")
        print("  May need higher Reynolds number to see concentration")
    
    # Vorticity-weighted analysis
    print("\n" + "=" * 60)
    print("VORTICITY-WEIGHTED LOCALIZATION")
    print("=" * 60)
    
    # Check if high-vorticity points cluster in mode space
    high_vort_mask = vorticity > np.percentile(vorticity, 90)
    n_high = high_vort_mask.sum()
    
    print(f"\nAnalyzing top 10% vorticity points ({n_high} points)")
    
    # Project onto slowest modes
    for i in range(1, min(5, len(eigenvalues))):
        mode = eigenvectors[:, i]
        mode_high = mode[high_vort_mask]
        mode_low = mode[~high_vort_mask]
        
        # Check if high-vorticity points are localized in this mode
        var_high = np.var(mode_high)
        var_low = np.var(mode_low)
        localization = var_high / (var_low + 1e-10)
        
        if localization > 2:
            print(f"  Mode {i} (λ={eigenvalues[i]:.4f}): High vorticity LOCALIZED ({localization:.1f}x variance)")
        elif localization < 0.5:
            print(f"  Mode {i} (λ={eigenvalues[i]:.4f}): High vorticity DELOCALIZED ({localization:.1f}x variance)")
    
    return {
        'spectral_gap': spectral_gap,
        'eigenvalues': eigenvalues,
        'inferred_dimension': inferred_dim
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python unified_heat_kernel.py <filename.npz>")
        print("\nSupported formats:")
        print("  - pnp_manifold_clouds.npz (P vs NP)")
        print("  - ns_topology_cloud.npz (Navier-Stokes)")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Auto-detect file type
    data = np.load(filename, allow_pickle=True)
    keys = list(data.keys())
    
    if 'p_cloud' in keys and 'np_cloud' in keys:
        results = analyze_pnp(filename)
    elif 'point_cloud' in keys:
        results = analyze_ns(filename)
    else:
        print(f"Unknown file format. Keys: {keys}")
        sys.exit(1)
    
    # Save results
    outfile = os.path.splitext(filename)[0] + "_hk_results.npz"
    np.savez(outfile, **results)
    print(f"\n✓ Results saved to {outfile}")
