"""
Visualization tools for Davis-Wilson cache space analysis.

Generates plots to visualize:
- Cache space structure (3D scatter, colored by cluster)
- Topological sector distribution
- Gap visibility metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def plot_cache_space_3d(
    embedding: NDArray[np.float64],
    cluster_labels: NDArray[np.int32],
    r: Optional[NDArray[np.int32]] = None,
    title: str = "Cache Space Structure",
    output_path: Optional[Path] = None,
    interactive: bool = True,
) -> None:
    """
    Create 3D scatter plot of cache space.
    
    Args:
        embedding: UMAP embedding, shape (N, 3)
        cluster_labels: Cluster assignments
        r: Topological charges (for marker shape)
        title: Plot title
        output_path: If provided, save to file
        interactive: If True, use Plotly; else use Matplotlib
    """
    if interactive:
        _plot_3d_plotly(embedding, cluster_labels, r, title, output_path)
    else:
        _plot_3d_matplotlib(embedding, cluster_labels, r, title, output_path)


def _plot_3d_plotly(
    embedding: NDArray[np.float64],
    cluster_labels: NDArray[np.int32],
    r: Optional[NDArray[np.int32]],
    title: str,
    output_path: Optional[Path],
) -> None:
    """Interactive 3D plot with Plotly."""
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Create figure
    fig = go.Figure()
    
    # Color by cluster
    unique_clusters = np.unique(cluster_labels)
    colors = px.colors.qualitative.Set1
    
    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        color = colors[i % len(colors)] if cluster >= 0 else 'gray'
        name = f"Cluster {cluster}" if cluster >= 0 else "Noise"
        
        fig.add_trace(go.Scatter3d(
            x=embedding[mask, 0],
            y=embedding[mask, 1],
            z=embedding[mask, 2],
            mode='markers',
            marker=dict(size=3, color=color, opacity=0.7),
            name=name,
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3",
        ),
        showlegend=True,
    )
    
    if output_path:
        fig.write_html(str(output_path))
    else:
        fig.show()


def _plot_3d_matplotlib(
    embedding: NDArray[np.float64],
    cluster_labels: NDArray[np.int32],
    r: Optional[NDArray[np.int32]],
    title: str,
    output_path: Optional[Path],
) -> None:
    """Static 3D plot with Matplotlib."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=cluster_labels,
        cmap='tab10',
        s=5,
        alpha=0.6,
    )
    
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title(title)
    
    plt.colorbar(scatter, label="Cluster")
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_gap_visibility(
    results: dict,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot gap visibility summary.
    
    Args:
        results: Output from compute_gap_visibility()
        output_path: If provided, save to file
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Bar chart of gap visibility per sector
    sectors = results["sectors"]
    sector_labels = list(sectors.keys())
    gap_values = [
        sectors[s]["gap_visibility"] if sectors[s]["gap_visibility"] is not None else 0
        for s in sector_labels
    ]
    
    ax1 = axes[0]
    bars = ax1.bar(range(len(sector_labels)), gap_values, color='steelblue')
    ax1.set_xticks(range(len(sector_labels)))
    ax1.set_xticklabels([f"r={s}" for s in sector_labels])
    ax1.set_xlabel("Topological Sector")
    ax1.set_ylabel("Gap Visibility G")
    ax1.set_title("Gap Visibility by Sector")
    ax1.axhline(y=1.0, color='red', linestyle='--', label='Threshold')
    ax1.legend()
    
    # Right: Summary statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    overall = results["overall"]
    summary_text = f"""
    DAVIS-WILSON GAP ANALYSIS
    ========================
    
    Overall Gap Visibility: {overall['gap_visibility']:.2f}
    Number of Clusters: {overall['n_clusters']}
    Silhouette Score: {overall['silhouette_score']:.3f}
    Min Inter-Cluster Distance: {overall['min_inter_cluster_distance']:.3f}
    Void Density: {overall['void_density']:.4f}
    
    INTERPRETATION
    -------------
    G > 1: Evidence for mass gap
    G < 1: No clear gap structure
    
    Verdict: {'SUPPORTS MASS GAP' if overall['gap_visibility'] > 1 else 'INCONCLUSIVE'}
    """
    
    ax2.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_topological_sectors(
    r: NDArray[np.int32],
    q_raw: NDArray[np.float64],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot distribution of topological charges.
    
    Shows:
    - Histogram of rounded charges r
    - Scatter of raw Q values
    
    Args:
        r: Integer topological charges
        q_raw: Raw (unrounded) charges
        output_path: If provided, save to file
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Histogram of integer charges
    ax1 = axes[0]
    unique_r, counts = np.unique(r, return_counts=True)
    ax1.bar(unique_r, counts, color='steelblue', edgecolor='black')
    ax1.set_xlabel("Topological Charge r")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Instanton Number")
    
    # Right: Raw Q values (should cluster near integers)
    ax2 = axes[1]
    ax2.hist(q_raw, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    for i in range(-3, 4):
        ax2.axvline(x=i, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Raw Topological Charge Q")
    ax2.set_ylabel("Count")
    ax2.set_title("Raw Q Distribution (should peak at integers)")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_summary_report(
    results: dict,
    output_dir: Path,
) -> None:
    """
    Generate a complete HTML report of the analysis.
    
    Creates:
    - summary.html: Main report page
    - cache_space.html: Interactive 3D visualization
    - gap_visibility.png: Summary metrics
    - sectors.png: Topological charge distribution
    
    Args:
        results: Full analysis results
        output_dir: Directory to save report files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual plots
    plot_cache_space_3d(
        results["embedding"],
        results["cluster_labels"],
        title="Davis-Wilson Cache Space",
        output_path=output_dir / "cache_space.html",
        interactive=True,
    )
    
    plot_gap_visibility(results, output_path=output_dir / "gap_visibility.png")
    
    # Generate HTML summary
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Davis-Wilson Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .metric {{ font-size: 24px; margin: 20px 0; }}
            .verdict {{ font-size: 32px; font-weight: bold; }}
            .support {{ color: green; }}
            .inconclusive {{ color: orange; }}
            iframe {{ border: none; }}
        </style>
    </head>
    <body>
        <h1>Davis-Wilson Mass Gap Analysis Report</h1>
        
        <h2>Key Results</h2>
        <p class="metric">Gap Visibility G = {results['overall']['gap_visibility']:.2f}</p>
        <p class="metric">Number of Clusters = {results['overall']['n_clusters']}</p>
        <p class="metric">Silhouette Score = {results['overall']['silhouette_score']:.3f}</p>
        
        <h2>Verdict</h2>
        <p class="verdict {'support' if results['overall']['gap_visibility'] > 1 else 'inconclusive'}">
            {'SUPPORTS MASS GAP' if results['overall']['gap_visibility'] > 1 else 'INCONCLUSIVE'}
        </p>
        
        <h2>Visualizations</h2>
        <h3>Cache Space (3D)</h3>
        <iframe src="cache_space.html" width="100%" height="600"></iframe>
        
        <h3>Gap Visibility Metrics</h3>
        <img src="gap_visibility.png" width="800">
        
        <h2>Interpretation</h2>
        <p>If G > 1, configurations cluster discretely in cache space, supporting the mass gap hypothesis.</p>
        <p>If G ≈ 0, configurations form a continuous cloud, inconsistent with a gap.</p>
    </body>
    </html>
    """
    
    with open(output_dir / "summary.html", "w") as f:
        f.write(html_content)
    
    print(f"Report saved to {output_dir / 'summary.html'}")


def plot_radial_density(
    Phi: NDArray[np.float64],
    cluster_labels: NDArray[np.int32],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot radial density from vacuum state.
    
    This is a KEY diagnostic for mass gap:
    - Gap exists → density near vacuum is zero, then rises at R_gap
    - No gap → density is non-zero everywhere from R=0
    
    The "moat" around the vacuum state is the mass gap signal.
    
    Args:
        Phi: Point cloud (PCA-reduced)
        cluster_labels: Cluster assignments
        output_path: If provided, save to file
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Reduce to manageable dimensions for distance computation
    if Phi.shape[1] > 50:
        pca = PCA(n_components=50)
        Phi_reduced = pca.fit_transform(Phi)
    else:
        Phi_reduced = Phi
    
    # Find vacuum (centroid of largest cluster or overall mean)
    unique_labels = [l for l in np.unique(cluster_labels) if l >= 0]
    
    if len(unique_labels) > 0:
        cluster_sizes = [(l, (cluster_labels == l).sum()) for l in unique_labels]
        largest_cluster = max(cluster_sizes, key=lambda x: x[1])[0]
        vacuum = Phi_reduced[cluster_labels == largest_cluster].mean(axis=0)
    else:
        vacuum = Phi_reduced.mean(axis=0)
    
    # Compute radial distances
    radii = np.linalg.norm(Phi_reduced - vacuum, axis=1)
    
    # Compute radial density histogram
    n_bins = 50
    counts, bin_edges = np.histogram(radii, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Normalize by shell volume (approximate for high-D)
    dim = Phi_reduced.shape[1]
    shell_volumes = np.maximum(bin_centers ** (dim - 1) * bin_widths, 1e-10)
    density = counts / shell_volumes
    density = density / density.max() if density.max() > 0 else density
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Raw histogram of radii
    ax1 = axes[0]
    ax1.hist(radii, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Distance from Vacuum $R$")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Radial Distance from Vacuum")
    ax1.axvline(x=0, color='red', linestyle='--', label='Vacuum')
    ax1.legend()
    
    # Right: Radial density (normalized by shell volume)
    ax2 = axes[1]
    ax2.plot(bin_centers, density, 'b-', linewidth=2)
    ax2.fill_between(bin_centers, 0, density, alpha=0.3)
    ax2.set_xlabel("Distance from Vacuum $R$")
    ax2.set_ylabel("Normalized Density $\\rho(R)$")
    ax2.set_title("Radial Density Profile (Gap = dip near R=0)")
    
    # Mark the gap region
    threshold = 0.1
    gap_end_idx = np.argmax(density > threshold)
    if gap_end_idx > 0:
        gap_R = bin_centers[gap_end_idx]
        ax2.axvline(x=gap_R, color='red', linestyle='--', label=f'Gap ≈ {gap_R:.2f}')
        ax2.axvspan(0, gap_R, alpha=0.2, color='red', label='Void region')
        ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pairwise_distance_histogram(
    Phi: NDArray[np.float64],
    output_path: Optional[Path] = None,
    max_samples: int = 3000,
) -> None:
    """
    Plot histogram of pairwise distances.
    
    KEY DIAGNOSTIC: If there's a mass gap, this histogram should show
    a DIP (low density) near zero distance, because configurations
    are separated into distinct clusters.
    
    A gapless distribution shows monotonic increase from zero.
    
    Args:
        Phi: Point cloud
        output_path: If provided, save to file
        max_samples: Maximum points to sample for efficiency
    """
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA
    
    # Reduce dimensions if needed
    if Phi.shape[1] > 50:
        pca = PCA(n_components=50)
        Phi_reduced = pca.fit_transform(Phi)
    else:
        Phi_reduced = Phi
    
    # Subsample if too many points
    n = len(Phi_reduced)
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        Phi_sample = Phi_reduced[indices]
    else:
        Phi_sample = Phi_reduced
    
    # Compute pairwise distances
    distances = pdist(Phi_sample, metric='euclidean')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts, bin_edges, _ = ax.hist(distances, bins=100, color='steelblue', 
                                    edgecolor='black', alpha=0.7)
    
    ax.set_xlabel("Pairwise Distance $d_{ij}$")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise Distance Histogram\n(Dip near zero = mass gap signal)")
    
    # Check for dip near zero
    first_bins = counts[1:6]  # Skip bin 0
    later_bins = counts[10:20]
    
    if len(later_bins) > 0 and first_bins.mean() > 0:
        dip_ratio = later_bins.mean() / first_bins.mean()
        if dip_ratio > 2:
            ax.axvspan(bin_edges[0], bin_edges[6], alpha=0.3, color='green',
                      label=f'Potential gap region (dip ratio: {dip_ratio:.1f})')
            ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_null_hypothesis_test(
    Phi: NDArray[np.float64],
    gap_visibility: float,
    n_permutations: int = 100,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Perform and visualize null hypothesis test for gap detection.
    
    Null Hypothesis H0: Configurations are drawn from a GAPLESS distribution
                        (Gaussian ball, not uniform)
    
    Alternative H1: Configurations cluster discretely (gap exists)
    
    Test: Compare observed gap visibility to null distribution
    
    Args:
        Phi: Point cloud
        gap_visibility: Observed gap visibility metric
        n_permutations: Number of permutation samples
        output_path: If provided, save to file
    
    Returns:
        Dictionary with p-value and test results
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    # Generate null distribution by shuffling components
    # This creates a gapless distribution with same marginal statistics
    null_G_values = []
    
    # Reduce dimensions for efficiency
    if Phi.shape[1] > 50:
        pca = PCA(n_components=50)
        Phi_reduced = pca.fit_transform(Phi)
    else:
        Phi_reduced = Phi
    
    for _ in range(n_permutations):
        # Shuffle each column independently (destroys structure)
        Phi_shuffled = Phi_reduced.copy()
        for j in range(Phi_shuffled.shape[1]):
            np.random.shuffle(Phi_shuffled[:, j])
        
        # Compute gap visibility on shuffled data
        from .clustering import analyze_cache_space
        r_null = np.zeros(len(Phi_shuffled), dtype=np.int32)
        result = analyze_cache_space(Phi_shuffled, r_null)
        null_G_values.append(result.gap_visibility)
    
    null_G_values = np.array(null_G_values)
    
    # Compute p-value
    p_value = (null_G_values >= gap_visibility).mean()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(null_G_values, bins=30, color='gray', alpha=0.7, 
            label='Null distribution (gapless)')
    ax.axvline(x=gap_visibility, color='red', linewidth=2,
               label=f'Observed G = {gap_visibility:.2f}')
    
    # Mark percentiles
    percentile_95 = np.percentile(null_G_values, 95)
    ax.axvline(x=percentile_95, color='orange', linestyle='--',
               label=f'95th percentile = {percentile_95:.2f}')
    
    ax.set_xlabel("Gap Visibility G")
    ax.set_ylabel("Count")
    ax.set_title(f"Null Hypothesis Test (p-value = {p_value:.4f})")
    ax.legend()
    
    # Add verdict
    if p_value < 0.05:
        verdict = "REJECT H0 (p < 0.05): Evidence for mass gap"
        color = 'green'
    else:
        verdict = "FAIL TO REJECT H0: Insufficient evidence for gap"
        color = 'orange'
    
    ax.text(0.5, 0.95, verdict, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='center',
            color=color, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        "p_value": float(p_value),
        "observed_G": gap_visibility,
        "null_mean": float(null_G_values.mean()),
        "null_std": float(null_G_values.std()),
        "percentile_95": float(percentile_95),
        "reject_null": p_value < 0.05,
    }
