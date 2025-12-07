"""
Topological Data Analysis (TDA) for mass gap detection.

Persistent homology provides a rigorous way to detect topological
features (clusters, loops, voids) in the cache space that persist
across multiple scales.

Key insight: The mass gap should manifest as:
- Long-lived H_0 barcode features (well-separated clusters)
- A clear "moat" around the vacuum component

Reference: Edelsbrunner & Harer, "Computational Topology"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class PersistenceResult:
    """
    Results of persistent homology computation.
    
    Attributes:
        betti_0: Number of connected components at final scale
        betti_1: Number of loops at final scale
        barcode_h0: Birth-death pairs for H_0 (connected components)
        barcode_h1: Birth-death pairs for H_1 (loops)
        persistence_entropy: Entropy of persistence diagram
        gap_persistence: Longest gap in H_0 (separation between vacuum and first excited)
    """
    betti_0: int
    betti_1: int
    barcode_h0: List[Tuple[float, float]]
    barcode_h1: List[Tuple[float, float]]
    persistence_entropy: float
    gap_persistence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "persistence_entropy": self.persistence_entropy,
            "gap_persistence": self.gap_persistence,
            "n_h0_features": len(self.barcode_h0),
            "n_h1_features": len(self.barcode_h1),
        }


def compute_persistent_homology(
    Phi: NDArray[np.float64],
    max_dimension: int = 1,
    max_edge_length: Optional[float] = None,
    n_samples: int = 2000,
) -> PersistenceResult:
    """
    Compute persistent homology of the cache space point cloud.
    
    Uses Rips complex filtration to track topological features
    across scales.
    
    Args:
        Phi: Point cloud, shape (N, d)
        max_dimension: Maximum homology dimension (0 = clusters, 1 = loops)
        max_edge_length: Maximum edge length in Rips complex
        n_samples: Maximum points to use (for computational efficiency)
    
    Returns:
        PersistenceResult with Betti numbers and barcodes
    
    The mass gap signal in TDA:
        - H_0 barcode should show a long-lived separation between the
          vacuum component (dies last) and other components
        - Large gap_persistence indicates well-separated clusters
    """
    try:
        import ripser
        from sklearn.decomposition import PCA
    except ImportError:
        # Return empty result if ripser not installed
        return PersistenceResult(
            betti_0=0, betti_1=0,
            barcode_h0=[], barcode_h1=[],
            persistence_entropy=0.0,
            gap_persistence=0.0,
        )
    
    # Reduce dimensions if needed (TDA is expensive in high-D)
    if Phi.shape[1] > 30:
        pca = PCA(n_components=30)
        Phi_reduced = pca.fit_transform(Phi)
    else:
        Phi_reduced = Phi
    
    # Subsample if too many points
    n = len(Phi_reduced)
    if n > n_samples:
        indices = np.random.choice(n, n_samples, replace=False)
        Phi_sample = Phi_reduced[indices]
    else:
        Phi_sample = Phi_reduced
    
    # Set max edge length based on data if not provided
    if max_edge_length is None:
        from scipy.spatial.distance import pdist
        dists = pdist(Phi_sample[:min(500, len(Phi_sample))])
        max_edge_length = np.percentile(dists, 90)
    
    # Compute Rips filtration and persistent homology
    result = ripser.ripser(
        Phi_sample,
        maxdim=max_dimension,
        thresh=max_edge_length,
    )
    
    diagrams = result['dgms']
    
    # Extract H_0 (connected components)
    h0 = diagrams[0]
    barcode_h0 = [(float(b), float(d)) for b, d in h0 if not np.isinf(d)]
    # Include infinite death with max_edge_length
    barcode_h0.extend([(float(b), max_edge_length) for b, d in h0 if np.isinf(d)])
    
    # Extract H_1 (loops) if computed
    barcode_h1 = []
    if max_dimension >= 1 and len(diagrams) > 1:
        h1 = diagrams[1]
        barcode_h1 = [(float(b), float(d)) for b, d in h1 if not np.isinf(d)]
    
    # Compute Betti numbers at final scale
    betti_0 = sum(1 for b, d in barcode_h0 if d >= max_edge_length * 0.9)
    betti_1 = sum(1 for b, d in barcode_h1 if d >= max_edge_length * 0.9)
    
    # Compute persistence entropy (measure of feature significance)
    lifetimes = [d - b for b, d in barcode_h0 if d > b]
    if len(lifetimes) > 0:
        total_lifetime = sum(lifetimes)
        probs = [l / total_lifetime for l in lifetimes]
        persistence_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    else:
        persistence_entropy = 0.0
    
    # Compute gap persistence (longest persistence in H_0)
    # This indicates how well-separated the vacuum is from excited states
    if len(lifetimes) > 0:
        gap_persistence = max(lifetimes)
    else:
        gap_persistence = 0.0
    
    return PersistenceResult(
        betti_0=betti_0,
        betti_1=betti_1,
        barcode_h0=barcode_h0,
        barcode_h1=barcode_h1,
        persistence_entropy=persistence_entropy,
        gap_persistence=gap_persistence,
    )


def plot_persistence_diagram(
    result: PersistenceResult,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot persistence diagrams (birth-death plots).
    
    Features far from the diagonal indicate significant topological
    structure (mass gap signal).
    
    Args:
        result: PersistenceResult from compute_persistent_homology
        output_path: If provided, save to file
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # H_0 persistence diagram
    ax1 = axes[0]
    if len(result.barcode_h0) > 0:
        births = [b for b, d in result.barcode_h0]
        deaths = [d for b, d in result.barcode_h0]
        max_val = max(max(deaths), max(births)) * 1.1
        
        ax1.scatter(births, deaths, c='blue', alpha=0.6, s=30)
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Diagonal')
        ax1.set_xlim(-0.05 * max_val, max_val)
        ax1.set_ylim(-0.05 * max_val, max_val)
    
    ax1.set_xlabel("Birth")
    ax1.set_ylabel("Death")
    ax1.set_title(f"H₀ Persistence (Betti₀ = {result.betti_0})\nGap persistence = {result.gap_persistence:.2f}")
    ax1.legend()
    
    # H_1 persistence diagram
    ax2 = axes[1]
    if len(result.barcode_h1) > 0:
        births = [b for b, d in result.barcode_h1]
        deaths = [d for b, d in result.barcode_h1]
        max_val = max(max(deaths), max(births)) * 1.1
        
        ax2.scatter(births, deaths, c='red', alpha=0.6, s=30)
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Diagonal')
        ax2.set_xlim(-0.05 * max_val, max_val)
        ax2.set_ylim(-0.05 * max_val, max_val)
    
    ax2.set_xlabel("Birth")
    ax2.set_ylabel("Death")
    ax2.set_title(f"H₁ Persistence (Betti₁ = {result.betti_1})")
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_barcode(
    result: PersistenceResult,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot persistence barcodes.
    
    Long bars in H_0 indicate well-separated clusters (mass gap).
    
    Args:
        result: PersistenceResult from compute_persistent_homology
        output_path: If provided, save to file
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # H_0 barcode
    ax1 = axes[0]
    h0_sorted = sorted(result.barcode_h0, key=lambda x: x[1] - x[0], reverse=True)
    n_show = min(30, len(h0_sorted))
    
    for i, (birth, death) in enumerate(h0_sorted[:n_show]):
        ax1.barh(i, death - birth, left=birth, height=0.8, color='blue', alpha=0.7)
    
    ax1.set_xlabel("Scale")
    ax1.set_ylabel("Feature index")
    ax1.set_title(f"H₀ Barcode (Connected Components)\nLongest bar = {result.gap_persistence:.2f}")
    ax1.invert_yaxis()
    
    # H_1 barcode
    ax2 = axes[1]
    h1_sorted = sorted(result.barcode_h1, key=lambda x: x[1] - x[0], reverse=True)
    n_show = min(30, len(h1_sorted))
    
    for i, (birth, death) in enumerate(h1_sorted[:n_show]):
        ax2.barh(i, death - birth, left=birth, height=0.8, color='red', alpha=0.7)
    
    ax2.set_xlabel("Scale")
    ax2.set_ylabel("Feature index")
    ax2.set_title("H₁ Barcode (Loops)")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_gap_with_tda(
    Phi: NDArray[np.float64],
    r: NDArray[np.int32],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Complete TDA analysis for mass gap detection.
    
    Args:
        Phi: Cache space points
        r: Topological charges
        output_dir: Directory for saving plots
    
    Returns:
        Dictionary with TDA results
    """
    # Compute persistent homology
    result = compute_persistent_homology(Phi, max_dimension=1)
    
    # Generate plots if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_persistence_diagram(result, output_dir / "persistence_diagram.png")
        plot_barcode(result, output_dir / "barcode.png")
    
    # Interpret results
    interpretation = {
        "gap_detected": result.gap_persistence > 0.5,  # Threshold for significance
        "n_clusters": result.betti_0,
        "n_loops": result.betti_1,
        "gap_persistence": result.gap_persistence,
        "persistence_entropy": result.persistence_entropy,
    }
    
    return {
        "tda_metrics": result.to_dict(),
        "interpretation": interpretation,
    }
