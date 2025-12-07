"""
Clustering analysis and gap visibility metrics.

The central prediction of the Davis-Wilson framework:
    - Mass gap exists → configurations cluster discretely in cache space
    - No mass gap → configurations form continuous cloud

We test this by:
    1. Clustering directly in high-dimensional space (avoid UMAP distortion)
    2. Pairwise distance histogram analysis
    3. Radial density around vacuum state
    4. UMAP used ONLY for visualization

CRITICAL: UMAP can create spurious clusters in uniform noise. All gap
detection must use the original high-dimensional vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class ClusteringResult:
    """
    Results of clustering analysis on cache space.
    
    Attributes:
        gap_visibility: G metric (0 = no gap, >0 = gap detected)
        n_clusters: Number of detected clusters
        cluster_labels: Cluster assignment for each point
        cluster_centers: Centroids of each cluster
        silhouette_score: Clustering quality metric
        min_inter_cluster_distance: Smallest distance between clusters
        void_density: Estimated density in void regions
        embedding: Low-dimensional projection for visualization ONLY
        radial_gap: Gap detected in radial density from vacuum
        pairwise_histogram: Histogram of pairwise distances
        tda_distance_scale: ε range used for TDA (for matching to physical length)
    """
    gap_visibility: float
    n_clusters: int
    cluster_labels: NDArray[np.int32]
    cluster_centers: NDArray[np.float64]
    silhouette_score: float
    min_inter_cluster_distance: float
    void_density: float
    embedding: NDArray[np.float64]  # Shape (N, 3) - for visualization only
    radial_gap: Optional[float] = None
    pairwise_histogram: Optional[Tuple[NDArray, NDArray]] = None
    tda_distance_scale: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "gap_visibility": self.gap_visibility,
            "n_clusters": self.n_clusters,
            "silhouette_score": self.silhouette_score,
            "min_inter_cluster_distance": self.min_inter_cluster_distance,
            "void_density": self.void_density,
            "radial_gap": self.radial_gap,
            "tda_distance_scale": self.tda_distance_scale,
        }


def analyze_cache_space(
    Phi: NDArray[np.float64],
    r: NDArray[np.int32],
    sector: Optional[int] = None,
    pca_components: int = 50,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    hdbscan_min_cluster_size: int = 50,
    hdbscan_min_samples: int = 10,
) -> ClusteringResult:
    """
    Perform clustering analysis on cache space to detect gap structure.
    
    CRITICAL: Clustering is done in HIGH-DIMENSIONAL space (PCA-reduced for 
    computational efficiency, but preserving linear structure). UMAP is used
    ONLY for visualization, never for gap detection.
    
    Args:
        Phi: Continuous cache vectors, shape (N, d_Phi)
        r: Topological charges, shape (N,)
        sector: If provided, analyze only this topological sector
        pca_components: PCA dimensions for clustering (linear, preserves structure)
        umap_n_neighbors: UMAP parameter for visualization only
        umap_min_dist: UMAP parameter for visualization only
        hdbscan_min_cluster_size: Minimum points to form a cluster
        hdbscan_min_samples: Core point threshold
    
    Returns:
        ClusteringResult with all metrics and visualization data
    
    Algorithm:
        1. Filter to specified topological sector (if provided)
        2. PCA reduce for clustering (linear, preserves global structure)
        3. Cluster using HDBSCAN in PCA space
        4. Compute pairwise distance histogram (key gap diagnostic)
        5. Compute radial density from vacuum state
        6. UMAP reduce for visualization only (NEVER for clustering)
        7. Calculate gap visibility from multiple metrics
    """
    import hdbscan
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import pdist
    
    # Filter to sector if specified
    if sector is not None:
        mask = r == sector
        Phi_filtered = Phi[mask]
    else:
        Phi_filtered = Phi
    
    n_points = len(Phi_filtered)
    
    if n_points < hdbscan_min_cluster_size:
        # Not enough points for meaningful clustering
        return ClusteringResult(
            gap_visibility=0.0,
            n_clusters=0,
            cluster_labels=np.zeros(n_points, dtype=np.int32),
            cluster_centers=np.array([]),
            silhouette_score=0.0,
            min_inter_cluster_distance=0.0,
            void_density=1.0,
            embedding=np.zeros((n_points, 3)),
            radial_gap=None,
            pairwise_histogram=None,
        )
    
    # Step 1: PCA reduction for clustering (LINEAR, preserves global structure)
    n_pca = min(pca_components, Phi_filtered.shape[1], n_points - 1)
    pca = PCA(n_components=n_pca)
    Phi_pca = pca.fit_transform(Phi_filtered)
    
    # Step 2: Clustering in PCA space (NOT UMAP space!)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric='euclidean',
    )
    cluster_labels = clusterer.fit_predict(Phi_pca)
    
    # Step 3: Compute pairwise distance histogram (KEY DIAGNOSTIC)
    pairwise_hist = compute_pairwise_distance_histogram(Phi_pca)
    
    # Step 4: Compute radial density from vacuum
    radial_gap = compute_radial_gap(Phi_pca, cluster_labels)
    
    # Step 5: UMAP for visualization ONLY
    try:
        import umap
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=42,
        )
        embedding = reducer.fit_transform(Phi_pca)
    except ImportError:
        # Fallback to PCA if UMAP not available
        embedding = Phi_pca[:, :3] if Phi_pca.shape[1] >= 3 else np.zeros((n_points, 3))
    
    # Compute metrics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    if n_clusters < 2:
        # Even without multiple clusters, we can detect structure via radial gap
        # and pairwise histogram
        gap_visibility = compute_combined_gap_visibility(
            min_cluster_distance=0.0,
            void_density=1.0,
            radial_gap=radial_gap,
            pairwise_hist=pairwise_hist,
        )
        return ClusteringResult(
            gap_visibility=gap_visibility,
            n_clusters=n_clusters,
            cluster_labels=cluster_labels,
            cluster_centers=np.array([]),
            silhouette_score=0.0,
            min_inter_cluster_distance=0.0,
            void_density=1.0,
            embedding=embedding,
            radial_gap=radial_gap,
            pairwise_histogram=pairwise_hist,
        )
    
    # Cluster centers in PCA space
    cluster_centers = []
    for label in range(n_clusters):
        mask = cluster_labels == label
        center = Phi_pca[mask].mean(axis=0)
        cluster_centers.append(center)
    cluster_centers = np.array(cluster_centers)
    
    # Silhouette score (clustering quality)
    valid_mask = cluster_labels >= 0
    if valid_mask.sum() > n_clusters:
        sil_score = silhouette_score(Phi_pca[valid_mask], cluster_labels[valid_mask])
    else:
        sil_score = 0.0
    
    # Inter-cluster distance
    min_distance = compute_min_cluster_distance(cluster_centers)
    
    # Void density from pairwise histogram
    void_density = estimate_void_density_from_histogram(pairwise_hist)
    
    # Gap visibility: combine multiple metrics
    eps = 1e-10
    gap_visibility = compute_combined_gap_visibility(
        min_distance, void_density, radial_gap, pairwise_hist
    )
    
    return ClusteringResult(
        gap_visibility=gap_visibility,
        n_clusters=n_clusters,
        cluster_labels=cluster_labels,
        cluster_centers=cluster_centers,
        silhouette_score=sil_score,
        min_inter_cluster_distance=min_distance,
        void_density=void_density,
        embedding=embedding,
        radial_gap=radial_gap,
        pairwise_histogram=pairwise_hist,
    )


def compute_min_cluster_distance(centers: NDArray[np.float64]) -> float:
    """
    Compute minimum distance between cluster centers.
    
    Args:
        centers: Cluster centroids, shape (n_clusters, d)
    
    Returns:
        Minimum pairwise distance
    """
    from scipy.spatial.distance import pdist
    
    if len(centers) < 2:
        return 0.0
    
    distances = pdist(centers)
    return float(np.min(distances))


def compute_pairwise_distance_histogram(
    Phi: NDArray[np.float64],
    n_bins: int = 100,
    max_samples: int = 5000,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute histogram of pairwise distances.
    
    This is the KEY diagnostic for mass gap detection:
    - If there's a gap, the histogram P(d_ij) should have a DIP near zero
    - A gapless distribution shows monotonic increase from zero
    
    Args:
        Phi: Point cloud, shape (N, d)
        n_bins: Number of histogram bins
        max_samples: Maximum points to sample (for efficiency)
    
    Returns:
        (bin_edges, counts) tuple
    """
    from scipy.spatial.distance import pdist
    
    n = len(Phi)
    
    # Subsample if too many points
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        Phi_sample = Phi[indices]
    else:
        Phi_sample = Phi
    
    # Compute all pairwise distances
    distances = pdist(Phi_sample, metric='euclidean')
    
    # Histogram
    counts, bin_edges = np.histogram(distances, bins=n_bins)
    
    return bin_edges, counts.astype(np.float64)


def compute_radial_gap(
    Phi: NDArray[np.float64],
    cluster_labels: NDArray[np.int32],
) -> Optional[float]:
    """
    Compute the radial gap from the vacuum state.
    
    The vacuum is defined as the centroid of the largest cluster (or overall
    centroid if no clear clustering). We look for a "moat" around it.
    
    Args:
        Phi: Point cloud in (PCA) space
        cluster_labels: Cluster assignments
    
    Returns:
        R_gap: radius below which density drops to near zero
               Returns None if no clear gap detected
    """
    # Find vacuum state (centroid of largest cluster, or overall mean)
    unique_labels = [l for l in np.unique(cluster_labels) if l >= 0]
    
    if len(unique_labels) > 0:
        # Use largest cluster as vacuum
        cluster_sizes = [(l, (cluster_labels == l).sum()) for l in unique_labels]
        largest_cluster = max(cluster_sizes, key=lambda x: x[1])[0]
        vacuum = Phi[cluster_labels == largest_cluster].mean(axis=0)
    else:
        # Use overall mean
        vacuum = Phi.mean(axis=0)
    
    # Compute radial distances from vacuum
    radii = np.linalg.norm(Phi - vacuum, axis=1)
    
    # Compute radial density histogram
    n_bins = 50
    counts, bin_edges = np.histogram(radii, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Normalize by shell volume (2D: 2πr, 3D: 4πr², general: surface of sphere)
    # For high-D, approximate as power of r
    dim = Phi.shape[1]
    shell_volumes = bin_centers ** (dim - 1) * bin_widths
    shell_volumes = np.maximum(shell_volumes, 1e-10)  # Avoid division by zero
    density = counts / shell_volumes
    
    # Normalize
    if density.max() > 0:
        density = density / density.max()
    
    # Find gap: first bin where density exceeds 10% of max, after initial dip
    threshold = 0.1
    
    # Skip the very first bin (self-distance effects)
    for i in range(1, len(density)):
        if density[i] > threshold:
            # Found first significant density
            return float(bin_centers[i])
    
    return None


def estimate_void_density_from_histogram(
    pairwise_hist: Tuple[NDArray, NDArray],
) -> float:
    """
    Estimate void density from pairwise distance histogram.
    
    A gap manifests as low density at small (but non-zero) distances.
    
    Args:
        pairwise_hist: (bin_edges, counts) from compute_pairwise_distance_histogram
    
    Returns:
        Void density estimate (low = gap exists)
    """
    bin_edges, counts = pairwise_hist
    
    if len(counts) < 10:
        return 1.0
    
    # Normalize counts
    total = counts.sum()
    if total == 0:
        return 1.0
    
    density = counts / total
    
    # Look at the first 10% of distance range (excluding bin 0)
    # This is where the "gap" should appear
    n_void_bins = max(1, len(density) // 10)
    
    # Void density = average density in near-zero region
    void_density = density[1:n_void_bins+1].mean() if n_void_bins > 0 else density[1]
    
    # Compare to overall average
    avg_density = density.mean()
    
    if avg_density > 0:
        return void_density / avg_density
    return 1.0


def compute_combined_gap_visibility(
    min_cluster_distance: float,
    void_density: float,
    radial_gap: Optional[float],
    pairwise_hist: Optional[Tuple[NDArray, NDArray]],
) -> float:
    """
    Compute combined gap visibility metric from multiple diagnostics.
    
    Args:
        min_cluster_distance: Minimum distance between cluster centers
        void_density: Normalized void density (0-1, low = gap)
        radial_gap: Radial gap from vacuum (if detected)
        pairwise_hist: Pairwise distance histogram
    
    Returns:
        Combined gap visibility G (higher = stronger evidence for gap)
    """
    eps = 1e-10
    
    # Component 1: Cluster separation / void density
    # If no clusters found, this is 0
    G1 = min_cluster_distance / (void_density + eps)
    
    # Component 2: Histogram dip detection
    G2 = 0.0
    if pairwise_hist is not None:
        bin_edges, counts = pairwise_hist
        if len(counts) > 10:
            # Look for dip in first few bins (indicates void around vacuum)
            first_bins = counts[1:6]  # Skip bin 0 (self-distance)
            later_bins = counts[10:20] if len(counts) > 20 else counts[10:]
            
            if len(later_bins) > 0 and first_bins.mean() > 0:
                # Dip ratio: if first bins are empty relative to later bins, G2 is high
                dip_ratio = later_bins.mean() / (first_bins.mean() + eps)
                G2 = np.log1p(dip_ratio)  # Log scale
    
    # Component 3: Radial gap (most robust for single-cluster case)
    G3 = 0.0
    if radial_gap is not None and radial_gap > 0:
        G3 = radial_gap
    
    # Combine with weights that adapt to what data is available
    if min_cluster_distance > 0:
        # Multiple clusters found - use all metrics
        G = G1 + 0.5 * G2 + 0.3 * G3
    else:
        # No clusters found - rely on radial gap and histogram dip
        # These can still detect structure within a single cloud
        G = G2 + G3
    
    return float(G)


def estimate_void_density(
    embedding: NDArray[np.float64],
    cluster_labels: NDArray[np.int32],
    n_samples: int = 1000,
) -> float:
    """
    Estimate the density of points in void regions between clusters.
    
    A low void density indicates clear separation (gap).
    A high void density indicates continuous distribution (no gap).
    
    Args:
        embedding: Point positions in reduced space
        cluster_labels: Cluster assignments (-1 = noise)
        n_samples: Number of random samples to check
    
    Returns:
        Estimated density in void regions (relative to cluster density)
    
    Algorithm:
        1. Define "void" as points far from any cluster center
        2. Use KDE or histogram to estimate density
        3. Return ratio of void density to cluster density
    
    TODO: Implement proper void density estimation
    """
    # COPILOT: Implement void density estimation
    # Options:
    #   1. KDE-based: fit KDE, sample in void regions
    #   2. Grid-based: histogram, look at low-density cells between clusters
    #   3. Random sampling: sample points between centroids, count nearby configs
    
    from sklearn.neighbors import KernelDensity
    
    # Fit KDE
    kde = KernelDensity(bandwidth=0.5)
    kde.fit(embedding)
    
    # Find cluster centers
    unique_labels = set(cluster_labels) - {-1}
    if len(unique_labels) < 2:
        return 1.0
    
    centers = []
    for label in unique_labels:
        mask = cluster_labels == label
        center = embedding[mask].mean(axis=0)
        centers.append(center)
    centers = np.array(centers)
    
    # Sample points along lines between cluster centers (void region candidates)
    void_samples = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            # Sample along line between centers i and j
            for t in np.linspace(0.2, 0.8, 10):  # Avoid cluster cores
                point = centers[i] * (1 - t) + centers[j] * t
                void_samples.append(point)
    
    void_samples = np.array(void_samples)
    
    if len(void_samples) == 0:
        return 1.0
    
    # Evaluate density at void samples
    void_log_density = kde.score_samples(void_samples)
    void_density = np.exp(void_log_density).mean()
    
    # Evaluate density at cluster centers (reference)
    cluster_log_density = kde.score_samples(centers)
    cluster_density = np.exp(cluster_log_density).mean()
    
    # Return ratio
    if cluster_density > 0:
        return void_density / cluster_density
    else:
        return 1.0


def compute_gap_visibility(
    Phi: NDArray[np.float64],
    r: NDArray[np.int32],
) -> Dict[str, Any]:
    """
    Compute gap visibility for all topological sectors.
    
    This is the main entry point for gap detection.
    
    Args:
        Phi: Continuous cache vectors
        r: Topological charges
    
    Returns:
        Dictionary with overall results and per-sector breakdown
    """
    # Overall analysis (all sectors combined)
    overall = analyze_cache_space(Phi, r, sector=None)
    
    # Per-sector analysis
    sectors = {}
    for sector_val in np.unique(r):
        mask = r == sector_val
        n_in_sector = mask.sum()
        
        if n_in_sector >= 50:  # Minimum for meaningful analysis
            sector_result = analyze_cache_space(Phi, r, sector=sector_val)
            sectors[int(sector_val)] = {
                "n_points": int(n_in_sector),
                "n_clusters": sector_result.n_clusters,
                "gap_visibility": sector_result.gap_visibility,
                "silhouette_score": sector_result.silhouette_score,
            }
        else:
            sectors[int(sector_val)] = {
                "n_points": int(n_in_sector),
                "n_clusters": None,
                "gap_visibility": None,
                "silhouette_score": None,
            }
    
    return {
        "overall": overall.to_dict(),
        "sectors": sectors,
        "embedding": overall.embedding,
        "cluster_labels": overall.cluster_labels,
    }
