"""
Modal deployment for Davis-Wilson Lattice Verification.

Run the full experiment on Modal cloud infrastructure:
    modal run modal_app.py

This will:
    1. Generate gauge configurations (or use pre-existing)
    2. Compute Davis-Wilson map for all configs
    3. Perform clustering analysis
    4. Generate visualization report

Estimated cost: ~$20 for 10k configurations on A100
Estimated time: ~5 hours wall clock
"""

from pathlib import Path
from typing import Optional, Any
import json

import modal


def convert_to_native(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ============================================================================
# Modal Setup
# ============================================================================

app = modal.App("davis-wilson-lattice")

# Container image with all dependencies + local packages
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core numerics
        "numpy>=1.24",
        "scipy>=1.10",
        "numba>=0.57",
        "h5py>=3.8",
        # ML/Clustering
        "umap-learn>=0.5",
        "hdbscan>=0.8",
        "scikit-learn>=1.2",
        # Visualization
        "matplotlib>=3.7",
        "plotly>=5.14",
        # Progress
        "tqdm>=4.65",
        # TDA (Topological Data Analysis)
        "ripser>=0.6",
        "persim>=0.3",
    )
    # Add local packages to the image
    .add_local_dir("./lattice", remote_path="/root/lattice")
    .add_local_dir("./analysis", remote_path="/root/analysis")
)

# Persistent volume for storing configurations and results
volume = modal.Volume.from_name("davis-wilson-data", create_if_missing=True)

# ============================================================================
# Configuration Generation
# ============================================================================

@app.function(
    image=image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/data": volume},
)
def generate_configs(
    lattice_size: int = 16,
    beta: float = 6.0,
    n_configs: int = 1000,
    thermalization: int = 100,
    output_dir: str = "/data/configs",
) -> list[str]:
    """
    Generate gauge configurations using HMC.
    
    This is the most compute-intensive step.
    
    Args:
        lattice_size: L for L^4 lattice
        beta: Gauge coupling
        n_configs: Number of configurations to generate
        thermalization: Trajectories to discard
        output_dir: Where to save configs
    
    Returns:
        List of paths to saved configurations
    """
    import sys
    sys.path.insert(0, "/root")
    
    from lattice import generate_config_hmc, save_config
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate configs
    print(f"Generating {n_configs} configs on {lattice_size}^4 lattice at β={beta}")
    configs = generate_config_hmc(
        L=lattice_size,
        beta=beta,
        n_trajectories=n_configs,
        thermalization=thermalization,
    )
    
    # Save to volume
    paths = []
    for i, config in enumerate(configs):
        config.metadata["trajectory"] = i
        config.metadata["id"] = f"config_{i:05d}"
        path = output_path / f"config_{i:05d}.h5"
        save_config(config, path)
        paths.append(str(path))
    
    volume.commit()
    print(f"Saved {len(paths)} configs to {output_dir}")
    
    return paths


# ============================================================================
# Davis-Wilson Map Computation
# ============================================================================

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,  # 1 hour
    volumes={"/data": volume},
)
def compute_cache_single(
    config_path: str,
    skeleton_stride: int = 4,
    smearing_steps: int = 10,
) -> dict:
    """
    Compute Davis-Wilson map for a single configuration.
    
    Designed to be called in parallel via .map()
    
    Args:
        config_path: Path to configuration file
        skeleton_stride: Stride for skeleton construction
        smearing_steps: Number of smearing iterations
    
    Returns:
        Dictionary with phi, r, q_raw
    """
    import sys
    sys.path.insert(0, "/root")
    
    from lattice import load_config
    from lattice.skeleton import build_skeleton
    from analysis.davis_wilson import davis_wilson_map
    
    # Load config
    config = load_config(config_path)
    
    # Build skeleton
    skeleton = build_skeleton(config.L, stride=skeleton_stride)
    
    # Compute Davis-Wilson map
    result = davis_wilson_map(config, skeleton, smearing_steps=smearing_steps)
    
    return {
        "phi": result.phi.tolist(),
        "r": result.r,
        "q_raw": result.q_raw,
        "config_id": config_path,
    }


@app.function(
    image=image,
    timeout=7200,  # 2 hours
    volumes={"/data": volume},
)
def compute_cache_batch_parallel(
    config_paths: list[str],
    skeleton_stride: int = 4,
    smearing_steps: int = 10,
    output_path: str = "/data/cache_results.h5",
) -> str:
    """
    Compute Davis-Wilson map for all configurations in parallel.
    
    Uses Modal's .map() for parallel execution.
    
    Args:
        config_paths: List of config file paths
        skeleton_stride: Stride for skeleton
        smearing_steps: Smearing iterations
        output_path: Where to save results
    
    Returns:
        Path to results file
    """
    import numpy as np
    import h5py
    
    print(f"Computing cache for {len(config_paths)} configurations...")
    
    # Parallel computation using Modal
    results = list(compute_cache_single.map(
        config_paths,
        kwargs={"skeleton_stride": skeleton_stride, "smearing_steps": smearing_steps},
    ))
    
    # Aggregate results
    n = len(results)
    cache_dim = len(results[0]["phi"])
    
    Phi = np.zeros((n, cache_dim), dtype=np.float64)
    r = np.zeros(n, dtype=np.int32)
    q_raw = np.zeros(n, dtype=np.float64)
    
    for i, res in enumerate(results):
        Phi[i] = res["phi"]
        r[i] = res["r"]
        q_raw[i] = res["q_raw"]
    
    # Save to HDF5
    with h5py.File(output_path, "w") as f:
        f.create_dataset("Phi", data=Phi, compression="gzip")
        f.create_dataset("r", data=r)
        f.create_dataset("q_raw", data=q_raw)
        f.attrs["n_configs"] = n
        f.attrs["cache_dim"] = cache_dim
        f.attrs["skeleton_stride"] = skeleton_stride
    
    volume.commit()
    print(f"Saved cache results to {output_path}")
    
    return output_path


# ============================================================================
# Analysis
# ============================================================================

@app.function(
    image=image,
    timeout=1800,  # 30 min
    volumes={"/data": volume},
)
def analyze_results(
    cache_path: str = "/data/cache_results.h5",
    output_dir: str = "/data/results",
) -> dict:
    """
    Perform clustering analysis and generate visualizations.
    
    Args:
        cache_path: Path to cache results HDF5
        output_dir: Directory for output files
    
    Returns:
        Analysis results dictionary
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import h5py
    
    from analysis.clustering import compute_gap_visibility
    from analysis.visualization import (
        plot_cache_space_3d,
        plot_gap_visibility,
        plot_topological_sectors,
        create_summary_report,
        plot_radial_density,
        plot_pairwise_distance_histogram,
        plot_null_hypothesis_test,
    )
    from analysis.tda import analyze_gap_with_tda
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load cache results
    print(f"Loading cache from {cache_path}")
    with h5py.File(cache_path, "r") as f:
        Phi = f["Phi"][:]
        r = f["r"][:]
        q_raw = f["q_raw"][:]
    
    print(f"Loaded {len(Phi)} configurations with cache_dim={Phi.shape[1]}")
    
    # Compute gap visibility (clustering in HIGH-D space, not UMAP!)
    print("Computing gap visibility (high-D clustering)...")
    results = compute_gap_visibility(Phi, r)
    
    # Add raw data for visualization
    results["q_raw"] = q_raw.tolist()
    results["r"] = r.tolist()
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Standard plots
    plot_topological_sectors(r, q_raw, output_path / "sectors.png")
    create_summary_report(results, output_path)
    
    # New diagnostic plots
    print("Generating radial density plot...")
    plot_radial_density(Phi, results["cluster_labels"], output_path / "radial_density.png")
    
    print("Generating pairwise distance histogram...")
    plot_pairwise_distance_histogram(Phi, output_path / "pairwise_distances.png")
    
    print("Performing null hypothesis test...")
    null_test = plot_null_hypothesis_test(
        Phi, results["overall"]["gap_visibility"],
        n_permutations=50,  # Reduced for speed
        output_path=output_path / "null_hypothesis.png"
    )
    
    # Topological Data Analysis
    print("Computing persistent homology (TDA)...")
    tda_results = analyze_gap_with_tda(Phi, r, output_dir=output_path)
    
    # Save JSON results (convert numpy types to native Python types)
    json_results = convert_to_native({
        "overall": results["overall"],
        "sectors": results["sectors"],
        "null_hypothesis_test": null_test,
        "tda": tda_results,
    })
    with open(output_path / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    volume.commit()
    
    print(f"\n{'='*60}")
    print(f"DAVIS-WILSON ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Gap Visibility G = {results['overall']['gap_visibility']:.2f}")
    print(f"Number of Clusters = {results['overall']['n_clusters']}")
    print(f"Silhouette Score = {results['overall']['silhouette_score']:.3f}")
    print(f"Radial Gap = {results['overall'].get('radial_gap', 'N/A')}")
    print(f"Null Hypothesis p-value = {null_test['p_value']:.4f}")
    print(f"TDA Gap Persistence = {tda_results['tda_metrics']['gap_persistence']:.2f}")
    print(f"")
    if results['overall']['gap_visibility'] > 1 and null_test['reject_null']:
        print(">>> RESULT: STRONG EVIDENCE FOR MASS GAP <<<")
    elif results['overall']['gap_visibility'] > 1:
        print(">>> RESULT: SUPPORTS MASS GAP (weak significance) <<<")
    else:
        print(">>> RESULT: INCONCLUSIVE <<<")
    print(f"{'='*60}")
    print(f"Results saved to {output_dir}")
    
    return json_results


# ============================================================================
# Main Entrypoint
# ============================================================================

@app.local_entrypoint()
def main(
    lattice_size: int = 16,
    n_configs: int = 100,
    beta: float = 6.0,
    skeleton_stride: int = 4,
    skip_generation: bool = False,
    skip_cache: bool = False,
    analysis_only: bool = False,
):
    """
    Run the full Davis-Wilson verification experiment.
    
    Args:
        lattice_size: L for L^4 lattice (default 16)
        n_configs: Number of configurations (default 100)
        beta: Gauge coupling (default 6.0)
        skeleton_stride: Stride for Wilson loop skeleton (default 4)
        skip_generation: If True, use existing configs
        skip_cache: If True, skip cache computation (use existing)
        analysis_only: If True, only run analysis on existing cache
    """
    print("="*60)
    print("DAVIS-WILSON LATTICE VERIFICATION")
    print("="*60)
    print(f"Lattice size: {lattice_size}^4")
    print(f"Configurations: {n_configs}")
    print(f"Beta: {beta}")
    print(f"Skeleton stride: {skeleton_stride}")
    if analysis_only:
        print("MODE: Analysis only (reusing existing cache)")
    print("="*60)
    
    cache_path = "/data/cache_results.h5"
    
    # If analysis_only, skip to step 3
    if analysis_only:
        print("\n[1/3] Skipping config generation (analysis_only=True)")
        print("\n[2/3] Skipping cache computation (analysis_only=True)")
    else:
        # Step 1: Generate or find configurations
        if not skip_generation:
            print("\n[1/3] Generating gauge configurations...")
            config_paths = generate_configs.remote(
                lattice_size=lattice_size,
                beta=beta,
                n_configs=n_configs,
            )
        else:
            print("\n[1/3] Using existing configurations...")
            config_paths = [f"/data/configs/config_{i:05d}.h5" for i in range(n_configs)]
        
        print(f"    Found {len(config_paths)} configurations")
        
        # Step 2: Compute Davis-Wilson map
        if not skip_cache:
            print("\n[2/3] Computing Davis-Wilson map...")
            cache_path = compute_cache_batch_parallel.remote(
                config_paths=config_paths,
                skeleton_stride=skeleton_stride,
            )
            print(f"    Cache saved to {cache_path}")
        else:
            print("\n[2/3] Using existing cache...")
    
    # Step 3: Analyze and visualize
    print("\n[3/3] Analyzing results...")
    results = analyze_results.remote(cache_path=cache_path)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Gap Visibility G = {results['overall']['gap_visibility']:.2f}")
    print(f"  Verdict: {'SUPPORTS MASS GAP' if results['overall']['gap_visibility'] > 1 else 'INCONCLUSIVE'}")
    print("\nView full report at /data/results/summary.html")


# ============================================================================
# Beta Scan (Confinement/Deconfinement Transition)
# ============================================================================

@app.function(
    image=image,
    gpu="A100",
    timeout=28800,  # 8 hours
    volumes={"/data": volume},
)
def run_beta_scan(
    lattice_size: int = 16,
    n_configs_per_beta: int = 100,
    beta_min: float = 5.4,
    beta_max: float = 6.4,
    n_betas: int = 6,
    flow_time: float = 1.0,
) -> dict:
    """
    Run Davis-Wilson analysis across multiple beta values.
    
    The most convincing evidence for mass gap would be:
    - Confinement (low T, β ~ 6.0): Distinct clusters, high G
    - Deconfinement (high T, β ~ 5.5): Clusters merge, low G
    
    This function sweeps β to show the transition.
    
    Args:
        lattice_size: L for L^4 lattice
        n_configs_per_beta: Configs per β value
        beta_min: Minimum β value
        beta_max: Maximum β value
        n_betas: Number of β values to scan
        flow_time: Wilson flow time for RG scale
    
    Returns:
        Dictionary mapping β → gap visibility
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    
    from lattice import generate_config_hmc
    from lattice.skeleton import build_block_skeleton
    from analysis.davis_wilson import davis_wilson_map
    from analysis.clustering import compute_gap_visibility
    
    # Generate beta values
    beta_values = np.linspace(beta_min, beta_max, n_betas).tolist()
    
    results = {}
    
    for beta in beta_values:
        print(f"\n{'='*60}")
        print(f"Processing β = {beta}")
        print(f"{'='*60}")
        
        # Generate configs at this beta
        print(f"Generating {n_configs_per_beta} configurations...")
        configs = generate_config_hmc(
            L=lattice_size,
            beta=beta,
            n_trajectories=n_configs_per_beta,
            thermalization=50,
        )
        
        # Build skeleton (block-based for proper coarse-graining)
        skeleton = build_block_skeleton(lattice_size, block_size=2)
        
        # Compute Davis-Wilson map for all configs
        print("Computing Davis-Wilson map...")
        cache_results = []
        for config in configs:
            result = davis_wilson_map(
                config, skeleton,
                flow_time=flow_time,
                use_flow=True,
            )
            cache_results.append(result)
        
        # Aggregate
        Phi = np.array([r.phi for r in cache_results])
        r = np.array([r.r for r in cache_results], dtype=np.int32)
        
        # Compute gap visibility
        print("Computing gap visibility...")
        gap_results = compute_gap_visibility(Phi, r)
        
        results[beta] = {
            "gap_visibility": gap_results["overall"]["gap_visibility"],
            "n_clusters": gap_results["overall"]["n_clusters"],
            "silhouette_score": gap_results["overall"]["silhouette_score"],
            "n_configs": n_configs_per_beta,
        }
        
        print(f"β = {beta}: G = {results[beta]['gap_visibility']:.2f}, "
              f"n_clusters = {results[beta]['n_clusters']}")
    
    # Save results
    output_path = Path("/data/results/beta_scan.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    volume.commit()
    
    # Print summary
    print(f"\n{'='*60}")
    print("BETA SCAN COMPLETE")
    print(f"{'='*60}")
    print("\nResults:")
    print(f"{'β':>8} {'G':>10} {'Clusters':>10}")
    print("-" * 30)
    for beta in sorted(results.keys()):
        r = results[beta]
        print(f"{beta:>8.2f} {r['gap_visibility']:>10.2f} {r['n_clusters']:>10}")
    
    return results


@app.function(
    image=image,
    timeout=1800,
    volumes={"/data": volume},
)
def plot_beta_scan(results_path: str = "/data/results/beta_scan.json") -> None:
    """Generate visualization of beta scan results."""
    import matplotlib.pyplot as plt
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    betas = sorted([float(b) for b in results.keys()])
    G_values = [results[str(b)]["gap_visibility"] for b in betas]
    n_clusters = [results[str(b)]["n_clusters"] for b in betas]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gap visibility vs beta
    ax1 = axes[0]
    ax1.plot(betas, G_values, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='red', linestyle='--', label='Threshold G=1')
    ax1.set_xlabel("Coupling β = 6/g²")
    ax1.set_ylabel("Gap Visibility G")
    ax1.set_title("Gap Visibility vs Coupling\n(Higher β = weaker coupling)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Number of clusters vs beta
    ax2 = axes[1]
    ax2.plot(betas, n_clusters, 'gs-', linewidth=2, markersize=8)
    ax2.set_xlabel("Coupling β = 6/g²")
    ax2.set_ylabel("Number of Clusters")
    ax2.set_title("Cluster Count vs Coupling\n(Confinement → distinct clusters)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path("/data/results/beta_scan.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    volume.commit()
    print(f"Plot saved to {output_path}")


# ============================================================================
# Pooled Beta Analysis (Combine configs from different β values)
# ============================================================================

@app.function(
    image=image,
    gpu="A100",
    timeout=14400,  # 4 hours
    volumes={"/data": volume},
)
def run_pooled_beta(
    lattice_size: int = 8,
    n_configs_per_beta: int = 30,
    beta_values: list[float] = [5.7, 6.0, 6.3],
    flow_time: float = 0.0,
) -> dict:
    """
    Pool configs from multiple β values and analyze together.
    
    This is the key test: configs at different β have different plaquettes,
    so they SHOULD separate in cache space if the Davis-Wilson map works.
    
    Expected result:
    - HDBSCAN finds N clusters (one per β value)
    - G > 0 because clusters are separated
    - Cluster labels correlate with β labels
    
    Args:
        lattice_size: L for L^4 lattice
        n_configs_per_beta: Configs per β value
        beta_values: List of β values to use
        flow_time: Wilson flow time (0 = no flow)
    
    Returns:
        Analysis results with cluster vs β correlation
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    from lattice import generate_config_hmc
    from lattice.skeleton import build_block_skeleton
    from analysis.davis_wilson import davis_wilson_map
    from analysis.clustering import compute_gap_visibility
    
    print("="*60)
    print("POOLED BETA ANALYSIS")
    print("="*60)
    print(f"Lattice size: {lattice_size}^4")
    print(f"Beta values: {beta_values}")
    print(f"Configs per beta: {n_configs_per_beta}")
    print(f"Total configs: {len(beta_values) * n_configs_per_beta}")
    print("="*60)
    
    all_Phi = []
    all_r = []
    all_beta_labels = []
    all_plaquettes = []
    
    # Build skeleton once
    skeleton = build_block_skeleton(lattice_size, block_size=2)
    print(f"Skeleton: {skeleton.n_loops} loops, cache_dim={skeleton.cache_dim}")
    
    for beta_idx, beta in enumerate(beta_values):
        print(f"\n--- Generating configs at β = {beta} ---")
        
        configs = generate_config_hmc(
            L=lattice_size,
            beta=beta,
            n_trajectories=n_configs_per_beta,
            thermalization=100,
        )
        
        for config in configs:
            # Compute Davis-Wilson map
            result = davis_wilson_map(
                config, skeleton,
                flow_time=flow_time,
                use_flow=(flow_time > 0),
            )
            all_Phi.append(result.phi)
            all_r.append(result.r)
            all_beta_labels.append(beta_idx)
            all_plaquettes.append(config.metadata.get("plaquette", 0.0))
    
    # Convert to arrays
    Phi = np.array(all_Phi)
    r = np.array(all_r, dtype=np.int32)
    beta_labels = np.array(all_beta_labels)
    plaquettes = np.array(all_plaquettes)
    
    print(f"\n--- Pooled data ---")
    print(f"Phi shape: {Phi.shape}")
    print(f"Topological charges: {np.unique(r, return_counts=True)}")
    for i, beta in enumerate(beta_values):
        mask = beta_labels == i
        print(f"β={beta}: n={mask.sum()}, mean_plaq={plaquettes[mask].mean():.4f}")
    
    # Compute gap visibility on pooled data
    print("\n--- Computing gap visibility ---")
    results = compute_gap_visibility(Phi, r)
    
    cluster_labels = results["cluster_labels"]
    n_clusters = results["overall"]["n_clusters"]
    G = results["overall"]["gap_visibility"]
    
    print(f"Gap Visibility G = {G:.2f}")
    print(f"Number of clusters = {n_clusters}")
    
    # Compute correlation between clusters and β labels
    if n_clusters > 1:
        ari = adjusted_rand_score(beta_labels, cluster_labels)
        nmi = normalized_mutual_info_score(beta_labels, cluster_labels)
        print(f"\n--- Cluster vs β correlation ---")
        print(f"Adjusted Rand Index: {ari:.3f} (1.0 = perfect)")
        print(f"Normalized Mutual Info: {nmi:.3f} (1.0 = perfect)")
    else:
        ari = 0.0
        nmi = 0.0
        print("\n(No clusters found, cannot compute correlation)")
    
    # Create visualization
    output_path = Path("/data/results")
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. UMAP colored by β
    print("\n--- Computing UMAP projection ---")
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(Phi)
        
        ax1 = axes[0, 0]
        scatter = ax1.scatter(embedding[:, 0], embedding[:, 1], 
                             c=beta_labels, cmap='viridis', alpha=0.7, s=30)
        ax1.set_title(f"UMAP colored by β\n(Should show {len(beta_values)} clusters)")
        ax1.set_xlabel("UMAP 1")
        ax1.set_ylabel("UMAP 2")
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("β index")
        
        # 2. UMAP colored by cluster
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1],
                              c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
        ax2.set_title(f"UMAP colored by HDBSCAN cluster\n(n_clusters={n_clusters}, G={G:.2f})")
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
        plt.colorbar(scatter2, ax=ax2, label="Cluster")
    except Exception as e:
        print(f"UMAP failed: {e}")
        axes[0, 0].text(0.5, 0.5, f"UMAP failed: {e}", ha='center', va='center')
        axes[0, 1].text(0.5, 0.5, f"UMAP failed: {e}", ha='center', va='center')
    
    # 3. Plaquette distribution by β
    ax3 = axes[1, 0]
    for i, beta in enumerate(beta_values):
        mask = beta_labels == i
        ax3.hist(plaquettes[mask], bins=20, alpha=0.6, label=f"β={beta}")
    ax3.set_xlabel("Average Plaquette")
    ax3.set_ylabel("Count")
    ax3.set_title("Plaquette Distribution by β\n(Should be well-separated)")
    ax3.legend()
    
    # 4. Phi norm distribution by β
    ax4 = axes[1, 1]
    phi_norms = np.linalg.norm(Phi, axis=1)
    for i, beta in enumerate(beta_values):
        mask = beta_labels == i
        ax4.hist(phi_norms[mask], bins=20, alpha=0.6, label=f"β={beta}")
    ax4.set_xlabel("||Φ|| (cache space norm)")
    ax4.set_ylabel("Count")
    ax4.set_title("Cache Space Norm by β\n(Davis-Wilson map magnitude)")
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "pooled_beta.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    output = {
        "beta_values": beta_values,
        "n_configs_per_beta": n_configs_per_beta,
        "total_configs": len(beta_labels),
        "gap_visibility": float(G),
        "n_clusters": int(n_clusters),
        "adjusted_rand_index": float(ari),
        "normalized_mutual_info": float(nmi),
        "radial_gap": float(results["overall"].get("radial_gap", 0.0)),
        "silhouette_score": float(results["overall"]["silhouette_score"]),
        "plaquette_means": {str(b): float(plaquettes[beta_labels == i].mean()) 
                           for i, b in enumerate(beta_values)},
    }
    
    with open(output_path / "pooled_beta_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    volume.commit()
    
    print("\n" + "="*60)
    print("POOLED BETA ANALYSIS COMPLETE")
    print("="*60)
    print(f"Gap Visibility G = {G:.2f}")
    print(f"Clusters found: {n_clusters} (expected: {len(beta_values)})")
    print(f"Cluster-β correlation (ARI): {ari:.3f}")
    if G > 1 and n_clusters >= len(beta_values):
        print(">>> RESULT: CACHE MAP SUCCESSFULLY SEPARATES β VALUES <<<")
    elif n_clusters > 1:
        print(">>> RESULT: PARTIAL SEPARATION DETECTED <<<")
    else:
        print(">>> RESULT: NO SEPARATION - CACHE MAP MAY NEED IMPROVEMENT <<<")
    print("="*60)
    
    return output


# ============================================================================
# Development / Testing
# ============================================================================

@app.function(image=image, volumes={"/data": volume})
def test_small():
    """Quick test on tiny lattice (4^4, 10 configs)."""
    import sys
    sys.path.insert(0, "/root")
    
    from lattice import cold_start
    from lattice.skeleton import build_skeleton, build_block_skeleton
    from analysis.davis_wilson import davis_wilson_map
    
    print("Testing on 4^4 lattice...")
    
    config = cold_start(L=4, beta=6.0)
    
    # Test standard skeleton
    skeleton = build_skeleton(4, stride=1)
    print(f"Standard skeleton: {skeleton.n_loops} loops, cache_dim={skeleton.cache_dim}")
    
    # Test block skeleton
    block_skeleton = build_block_skeleton(4, block_size=2)
    print(f"Block skeleton: {block_skeleton.n_loops} loops, cache_dim={block_skeleton.cache_dim}")
    
    # Test Davis-Wilson map with Wilson flow
    result = davis_wilson_map(
        config, skeleton,
        flow_time=0.0,  # Skip flow for cold config test
        use_flow=False,
    )
    
    print(f"Phi shape: {result.phi.shape}")
    print(f"Topological charge r = {result.r} (raw Q = {result.q_raw:.4f})")
    print(f"First few Phi values: {result.phi[:10]}")
    
    print("\nTest PASSED")
    return True
