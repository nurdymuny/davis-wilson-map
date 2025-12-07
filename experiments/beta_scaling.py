"""
β-Scaling Experiment
====================
Run the gap ratio analysis at multiple β values to demonstrate
that the mass gap signature scales correctly with lattice spacing.

Physical expectation:
- Higher β → finer lattice → smoother configs → potentially sharper gaps
- Gap ratio should remain significant across the scaling window
- This validates that the gap is a continuum phenomenon, not a lattice artifact
"""

import modal
import numpy as np

app = modal.App("beta-scaling")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "numba", "scipy", "scikit-learn", "h5py", "matplotlib")
    .add_local_dir("./lattice", remote_path="/root/lattice")
    .add_local_dir("./analysis", remote_path="/root/analysis")
)

@app.function(image=image, gpu="T4", timeout=3600)
def beta_scaling_analysis():
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import io
    import base64
    
    import sys
    sys.path.insert(0, "/root")
    
    from lattice.gauge_config import hot_start, heatbath_sweep
    from lattice.wilson_loops import average_plaquette
    from lattice.topological import compute_topological_charge
    
    print("=" * 70)
    print("β-SCALING EXPERIMENT")
    print("=" * 70)
    print("Testing gap ratio across multiple β values")
    print("=" * 70)
    
    # β values to test (standard lattice QCD range)
    beta_values = [5.7, 5.85, 6.0, 6.1]
    L = 6  # Keep lattice small for speed
    n_configs = 40
    thermalization_sweeps = 150
    separation_sweeps = 15
    
    print(f"\nParameters:")
    print(f"  Lattice: {L}^4")
    print(f"  β values: {beta_values}")
    print(f"  Configs per β: {n_configs}")
    print()
    
    results = {}
    
    for beta in beta_values:
        print(f"\n{'='*60}")
        print(f"β = {beta}")
        print(f"{'='*60}")
        
        # Generate configs
        config = hot_start(L, beta)
        
        # Thermalize
        print("Thermalizing...")
        for i in range(thermalization_sweeps):
            config = heatbath_sweep(config)
        
        plaq = average_plaquette(config.U)
        print(f"After thermalization: <P> = {plaq:.4f}")
        
        # Collect configurations
        features = []
        charges = []
        plaquettes = []
        
        for i in range(n_configs):
            for _ in range(separation_sweeps):
                config = heatbath_sweep(config)
            
            # Features
            feat = []
            for mu in range(4):
                link_traces = np.real(np.trace(config.U[mu], axis1=-2, axis2=-1))
                feat.extend([np.mean(link_traces), np.std(link_traces)])
            
            plaq = average_plaquette(config.U)
            Q = compute_topological_charge(config)
            
            feat.append(plaq)
            feat.append(Q)
            
            features.append(feat)
            charges.append(Q)
            plaquettes.append(plaq)
        
        features = np.array(features)
        charges = np.array(charges)
        
        # Compute gap ratio
        pca = PCA(n_components=3)
        coords = pca.fit_transform(features)
        
        sectors = np.round(charges).astype(int)
        trivial_mask = sectors == 0
        n_trivial = np.sum(trivial_mask)
        
        if n_trivial >= 5:
            vacuum_center = np.mean(coords[trivial_mask], axis=0)
            radii = np.linalg.norm(coords - vacuum_center, axis=1)
            trivial_radii = radii[trivial_mask]
            
            sorted_radii = np.sort(trivial_radii)
            gaps = np.diff(sorted_radii)
            median_gap = np.median(gaps)
            max_gap = np.max(gaps)
            gap_ratio = max_gap / max(median_gap, 1e-10)
        else:
            gap_ratio = 0.0
            max_gap = 0.0
        
        # Sector distribution
        sector_counts = {}
        for q in [-2, -1, 0, 1, 2]:
            sector_counts[q] = np.sum(sectors == q)
        
        results[beta] = {
            'gap_ratio': gap_ratio,
            'max_gap': max_gap,
            'plaq_mean': np.mean(plaquettes),
            'plaq_std': np.std(plaquettes),
            'Q_std': np.std(charges),
            'n_trivial': n_trivial,
            'sector_counts': sector_counts,
            'variance_explained': sum(pca.explained_variance_ratio_),
            'coords': coords,
            'charges': charges
        }
        
        print(f"\nResults for β = {beta}:")
        print(f"  <P> = {np.mean(plaquettes):.4f} ± {np.std(plaquettes):.4f}")
        print(f"  σ(Q) = {np.std(charges):.3f}")
        print(f"  Gap Ratio = {gap_ratio:.1f}")
        print(f"  Sectors: {sector_counts}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: β-SCALING RESULTS")
    print("=" * 70)
    print(f"\n{'β':>6} | {'<P>':>8} | {'σ(Q)':>6} | {'Gap Ratio':>10} | {'Q=0':>4}")
    print("-" * 50)
    for beta in beta_values:
        r = results[beta]
        print(f"{beta:>6.2f} | {r['plaq_mean']:>8.4f} | {r['Q_std']:>6.3f} | {r['gap_ratio']:>10.1f} | {r['n_trivial']:>4}")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Gap ratio vs β
    ax1 = axes[0, 0]
    gap_ratios = [results[b]['gap_ratio'] for b in beta_values]
    ax1.bar(range(len(beta_values)), gap_ratios, color='steelblue', edgecolor='black')
    ax1.set_xticks(range(len(beta_values)))
    ax1.set_xticklabels([f'β={b}' for b in beta_values])
    ax1.axhline(5, color='red', linestyle='--', label='Threshold (5)')
    ax1.set_ylabel('Gap Ratio', fontsize=12)
    ax1.set_title('Gap Ratio vs β (Lattice Coupling)', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # 2. Plaquette vs β
    ax2 = axes[0, 1]
    plaq_means = [results[b]['plaq_mean'] for b in beta_values]
    plaq_stds = [results[b]['plaq_std'] for b in beta_values]
    ax2.errorbar(beta_values, plaq_means, yerr=plaq_stds, fmt='o-', 
                 capsize=5, color='green', markersize=8)
    ax2.set_xlabel('β', fontsize=12)
    ax2.set_ylabel('Average Plaquette <P>', fontsize=12)
    ax2.set_title('Plaquette vs β', fontsize=14, fontweight='bold')
    
    # 3. Q fluctuations vs β
    ax3 = axes[1, 0]
    Q_stds = [results[b]['Q_std'] for b in beta_values]
    ax3.plot(beta_values, Q_stds, 'o-', color='purple', markersize=8)
    ax3.set_xlabel('β', fontsize=12)
    ax3.set_ylabel('σ(Q)', fontsize=12)
    ax3.set_title('Topological Charge Fluctuations vs β', fontsize=14, fontweight='bold')
    
    # 4. PCA scatter for each β
    ax4 = axes[1, 1]
    colors = ['red', 'orange', 'blue', 'green']
    for i, beta in enumerate(beta_values):
        coords = results[beta]['coords']
        ax4.scatter(coords[:, 0], coords[:, 1], c=colors[i], 
                    label=f'β={beta}', alpha=0.6, s=30)
    ax4.set_xlabel('PC1', fontsize=12)
    ax4.set_ylabel('PC2', fontsize=12)
    ax4.set_title('Configuration Space (PCA)', fontsize=14, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Return summary
    return {
        'beta_values': beta_values,
        'gap_ratios': [results[b]['gap_ratio'] for b in beta_values],
        'plaq_means': [results[b]['plaq_mean'] for b in beta_values],
        'Q_stds': [results[b]['Q_std'] for b in beta_values],
        'image_base64': img_data
    }


@app.local_entrypoint()
def main():
    result = beta_scaling_analysis.remote()
    
    # Save image locally
    import base64
    img_data = base64.b64decode(result['image_base64'])
    img_path = "results/figures/beta_scaling.png"
    with open(img_path, 'wb') as f:
        f.write(img_data)
    print(f"\n✅ Saved: {img_path}")
    
    print("\n" + "=" * 70)
    print("β-SCALING FINAL RESULTS")
    print("=" * 70)
    for i, beta in enumerate(result['beta_values']):
        print(f"  β = {beta}: Gap Ratio = {result['gap_ratios'][i]:.1f}, <P> = {result['plaq_means'][i]:.4f}")
