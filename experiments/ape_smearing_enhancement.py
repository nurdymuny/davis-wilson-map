"""
APE Smearing Enhancement Experiment
====================================
Compare gap ratio before and after APE smearing to demonstrate
signal-to-noise improvement.

APE smearing acts as a UV low-pass filter, suppressing lattice artifacts
while preserving topological structure. It's highly correlated with
Wilson flow (>95% agreement on Q) but much faster to compute.
"""

import modal
import numpy as np

app = modal.App("ape-smearing-enhancement")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "numba", "scipy", "scikit-learn", "h5py", "matplotlib")
    .add_local_dir("./lattice", remote_path="/root/lattice")
    .add_local_dir("./analysis", remote_path="/root/analysis")
)

@app.function(image=image, gpu="T4", timeout=1800)
def smearing_comparison():
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import io
    import base64
    
    import sys
    sys.path.insert(0, "/root")
    
    from lattice.gauge_config import hot_start, heatbath_sweep
    from lattice.wilson_loops import average_plaquette
    from lattice.topological import compute_topological_charge, apply_smearing
    
    print("=" * 70)
    print("APE SMEARING ENHANCEMENT EXPERIMENT")
    print("=" * 70)
    print("Comparing gap ratio before vs after APE smearing")
    print("=" * 70)
    
    # Parameters - use lower beta for noisier raw Q
    L, beta = 6, 5.7  # Lower beta = rougher lattice, more UV noise
    n_configs = 30
    thermalization_sweeps = 150
    separation_sweeps = 15
    smearing_steps = 10  # More smearing needed at lower beta
    smearing_rho = 0.12
    
    print(f"\nParameters:")
    print(f"  Lattice: {L}^4")
    print(f"  β = {beta}")
    print(f"  Configurations: {n_configs}")
    print(f"  APE smearing: {smearing_steps} steps, ρ = {smearing_rho}")
    print()
    
    # Storage for both raw and smeared data
    raw_features = []
    smeared_features = []
    raw_charges = []
    smeared_charges = []
    raw_plaquettes = []
    smeared_plaquettes = []
    
    # Generate and thermalize initial config
    print("Thermalizing initial configuration...")
    config = hot_start(L, beta)
    for i in range(thermalization_sweeps):
        config = heatbath_sweep(config)
        if (i + 1) % 50 == 0:
            plaq = average_plaquette(config.U)
            print(f"  Sweep {i+1}: <P> = {plaq:.4f}")
    
    print(f"\nGenerating {n_configs} configurations...")
    
    for i in range(n_configs):
        # Separation sweeps
        for _ in range(separation_sweeps):
            config = heatbath_sweep(config)
        
        # === RAW (unsmeared) measurements ===
        raw_plaq = average_plaquette(config.U)
        raw_Q = compute_topological_charge(config)
        
        # Feature extraction (link traces)
        raw_feat = []
        for mu in range(4):
            link_traces = np.real(np.trace(config.U[mu], axis1=-2, axis2=-1))
            raw_feat.extend([np.mean(link_traces), np.std(link_traces)])
        raw_feat.append(raw_plaq)
        raw_feat.append(raw_Q)
        
        raw_features.append(raw_feat)
        raw_charges.append(raw_Q)
        raw_plaquettes.append(raw_plaq)
        
        # === SMEARED measurements ===
        smeared_config = apply_smearing(config, n_steps=smearing_steps, rho=smearing_rho)
        
        smeared_plaq = average_plaquette(smeared_config.U)
        smeared_Q = compute_topological_charge(smeared_config)
        
        smeared_feat = []
        for mu in range(4):
            link_traces = np.real(np.trace(smeared_config.U[mu], axis1=-2, axis2=-1))
            smeared_feat.extend([np.mean(link_traces), np.std(link_traces)])
        smeared_feat.append(smeared_plaq)
        smeared_feat.append(smeared_Q)
        
        smeared_features.append(smeared_feat)
        smeared_charges.append(smeared_Q)
        smeared_plaquettes.append(smeared_plaq)
        
        if (i + 1) % 10 == 0:
            print(f"  Config {i+1}/{n_configs}: Q_raw={raw_Q:.2f} → Q_smeared={smeared_Q:.2f}")
    
    # Convert to arrays
    raw_features = np.array(raw_features)
    smeared_features = np.array(smeared_features)
    raw_charges = np.array(raw_charges)
    smeared_charges = np.array(smeared_charges)
    
    print("\n" + "=" * 70)
    print("TOPOLOGICAL CHARGE ANALYSIS")
    print("=" * 70)
    
    # Compare charge distributions
    print(f"\nRaw charges:")
    print(f"  Mean: {np.mean(raw_charges):.3f}")
    print(f"  Std:  {np.std(raw_charges):.3f}")
    print(f"  Min:  {np.min(raw_charges):.3f}")
    print(f"  Max:  {np.max(raw_charges):.3f}")
    
    print(f"\nSmeared charges:")
    print(f"  Mean: {np.mean(smeared_charges):.3f}")
    print(f"  Std:  {np.std(smeared_charges):.3f}")
    print(f"  Min:  {np.min(smeared_charges):.3f}")
    print(f"  Max:  {np.max(smeared_charges):.3f}")
    
    # Check how close to integers
    raw_int_deviation = np.mean(np.abs(raw_charges - np.round(raw_charges)))
    smeared_int_deviation = np.mean(np.abs(smeared_charges - np.round(smeared_charges)))
    
    print(f"\nDeviation from integers:")
    print(f"  Raw:     {raw_int_deviation:.4f}")
    print(f"  Smeared: {smeared_int_deviation:.4f}")
    print(f"  Improvement: {raw_int_deviation / max(smeared_int_deviation, 0.001):.1f}×")
    
    # Sector counts
    raw_sectors = np.round(raw_charges).astype(int)
    smeared_sectors = np.round(smeared_charges).astype(int)
    
    print(f"\nSector distribution:")
    for Q in [-2, -1, 0, 1, 2]:
        raw_count = np.sum(raw_sectors == Q)
        smeared_count = np.sum(smeared_sectors == Q)
        print(f"  Q={Q:+d}: raw={raw_count}, smeared={smeared_count}")
    
    print("\n" + "=" * 70)
    print("GAP RATIO ANALYSIS")
    print("=" * 70)
    
    def compute_gap_ratio(features, charges):
        """Compute radial gap ratio for trivial sector."""
        # PCA to 3D
        pca = PCA(n_components=3)
        coords = pca.fit_transform(features)
        
        # Find vacuum center (mean of Q=0 configs)
        sectors = np.round(charges).astype(int)
        trivial_mask = sectors == 0
        
        if np.sum(trivial_mask) < 5:
            return 0.0, 0.0, coords, pca.explained_variance_ratio_
        
        vacuum_center = np.mean(coords[trivial_mask], axis=0)
        
        # Radial distances
        radii = np.linalg.norm(coords - vacuum_center, axis=1)
        trivial_radii = radii[trivial_mask]
        
        # Gap metrics
        sorted_radii = np.sort(trivial_radii)
        gaps = np.diff(sorted_radii)
        median_gap = np.median(gaps)
        max_gap = np.max(gaps)
        
        gap_ratio = max_gap / max(median_gap, 1e-10)
        
        return gap_ratio, max_gap, coords, pca.explained_variance_ratio_
    
    raw_gap_ratio, raw_max_gap, raw_coords, raw_var = compute_gap_ratio(raw_features, raw_charges)
    smeared_gap_ratio, smeared_max_gap, smeared_coords, smeared_var = compute_gap_ratio(smeared_features, smeared_charges)
    
    print(f"\nRaw configurations:")
    print(f"  Gap Ratio: {raw_gap_ratio:.1f}")
    print(f"  Max Gap:   {raw_max_gap:.4f}")
    print(f"  PCA Variance: {sum(raw_var)*100:.1f}%")
    
    print(f"\nSmeared configurations ({smearing_steps} APE steps):")
    print(f"  Gap Ratio: {smeared_gap_ratio:.1f}")
    print(f"  Max Gap:   {smeared_max_gap:.4f}")
    print(f"  PCA Variance: {sum(smeared_var)*100:.1f}%")
    
    improvement = smeared_gap_ratio / max(raw_gap_ratio, 1)
    print(f"\n🎯 GAP RATIO IMPROVEMENT: {improvement:.1f}×")
    
    if smeared_gap_ratio > 100:
        print("✅ Signal-to-noise is now UNDENIABLE")
    
    print("\n" + "=" * 70)
    print("PLAQUETTE COMPARISON")
    print("=" * 70)
    print(f"\nRaw:     <P> = {np.mean(raw_plaquettes):.4f} ± {np.std(raw_plaquettes):.4f}")
    print(f"Smeared: <P> = {np.mean(smeared_plaquettes):.4f} ± {np.std(smeared_plaquettes):.4f}")
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Charge distribution before/after
    ax1 = axes[0, 0]
    ax1.hist(raw_charges, bins=30, alpha=0.6, label='Raw', color='red', edgecolor='black')
    ax1.hist(smeared_charges, bins=30, alpha=0.6, label=f'APE ({smearing_steps} steps)', color='blue', edgecolor='black')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(-1, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(1, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Topological Charge Q', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Topological Charge Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # 2. Scatter: Q_raw vs Q_smeared
    ax2 = axes[0, 1]
    ax2.scatter(raw_charges, smeared_charges, c=np.abs(smeared_charges - np.round(smeared_charges)), 
                cmap='viridis', s=50, alpha=0.7)
    ax2.plot([-2, 2], [-2, 2], 'k--', alpha=0.3, label='y=x')
    ax2.axhline(-1, color='gray', linestyle=':', alpha=0.3)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax2.axhline(1, color='gray', linestyle=':', alpha=0.3)
    ax2.set_xlabel('Q (raw)', fontsize=12)
    ax2.set_ylabel('Q (smeared)', fontsize=12)
    ax2.set_title('Charge Stabilization', fontsize=14, fontweight='bold')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    
    # 3. Radial distribution comparison
    ax3 = axes[1, 0]
    raw_sectors = np.round(raw_charges).astype(int)
    smeared_sectors = np.round(smeared_charges).astype(int)
    
    # Compute radii for trivial sector
    raw_trivial = raw_sectors == 0
    smeared_trivial = smeared_sectors == 0
    
    vacuum_raw = np.mean(raw_coords[raw_trivial], axis=0) if np.any(raw_trivial) else np.zeros(3)
    vacuum_smeared = np.mean(smeared_coords[smeared_trivial], axis=0) if np.any(smeared_trivial) else np.zeros(3)
    
    raw_radii = np.linalg.norm(raw_coords - vacuum_raw, axis=1)
    smeared_radii = np.linalg.norm(smeared_coords - vacuum_smeared, axis=1)
    
    ax3.hist(raw_radii[raw_trivial], bins=20, alpha=0.6, label='Raw Q=0', color='red', edgecolor='black')
    ax3.hist(smeared_radii[smeared_trivial], bins=20, alpha=0.6, label='Smeared Q=0', color='blue', edgecolor='black')
    ax3.set_xlabel('Radial Distance from Vacuum', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Radial Distribution (Trivial Sector)', fontsize=14, fontweight='bold')
    ax3.legend()
    
    # 4. Summary metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    APE SMEARING ENHANCEMENT RESULTS
    =================================
    
    Configuration: {n_configs} configs, {L}⁴ lattice, β={beta}
    APE Smearing: {smearing_steps} steps, ρ = {smearing_rho}
    
    TOPOLOGICAL CHARGE:
      Raw deviation from integers:     {raw_int_deviation:.4f}
      Smeared deviation from integers: {smeared_int_deviation:.4f}
      Improvement: {raw_int_deviation / max(smeared_int_deviation, 0.001):.1f}×
    
    GAP RATIO:
      Raw:     {raw_gap_ratio:.1f}
      Smeared: {smeared_gap_ratio:.1f}
      Improvement: {improvement:.1f}×
    
    PLAQUETTE:
      Raw:     {np.mean(raw_plaquettes):.4f} ± {np.std(raw_plaquettes):.4f}
      Smeared: {np.mean(smeared_plaquettes):.4f} ± {np.std(smeared_plaquettes):.4f}
    
    CONCLUSION:
      APE smearing {'dramatically improves' if improvement > 2 else 'maintains'}
      the signal-to-noise ratio for topological structure.
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return {
        'n_configs': n_configs,
        'smearing_steps': smearing_steps,
        'raw_gap_ratio': float(raw_gap_ratio),
        'smeared_gap_ratio': float(smeared_gap_ratio),
        'improvement': float(improvement),
        'raw_int_deviation': float(raw_int_deviation),
        'smeared_int_deviation': float(smeared_int_deviation),
        'raw_plaq_mean': float(np.mean(raw_plaquettes)),
        'smeared_plaq_mean': float(np.mean(smeared_plaquettes)),
        'image_base64': img_data
    }


@app.local_entrypoint()
def main():
    result = smearing_comparison.remote()
    
    # Save image locally
    import base64
    img_data = base64.b64decode(result['image_base64'])
    img_path = "results/figures/ape_smearing_comparison.png"
    with open(img_path, 'wb') as f:
        f.write(img_data)
    print(f"\n✅ Saved: {img_path}")
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Raw Gap Ratio:     {result['raw_gap_ratio']:.1f}")
    print(f"  Smeared Gap Ratio: {result['smeared_gap_ratio']:.1f}")
    print(f"  IMPROVEMENT:       {result['improvement']:.1f}×")
    print()
    print(f"  Raw Q deviation:     {result['raw_int_deviation']:.4f}")
    print(f"  Smeared Q deviation: {result['smeared_int_deviation']:.4f}")
