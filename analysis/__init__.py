"""
Analysis tools for the Davis-Wilson framework.

This package provides:
- The Davis-Wilson map Γ: A/G → C
- Clustering analysis for gap detection
- Gap visibility metrics
- Visualization tools
- Topological Data Analysis (TDA)
"""

from .davis_wilson import (
    davis_wilson_map,
    compute_cache_batch,
    CacheResult,
)
from .clustering import (
    analyze_cache_space,
    compute_gap_visibility,
    ClusteringResult,
)
from .visualization import (
    plot_cache_space_3d,
    plot_gap_visibility,
    plot_topological_sectors,
    plot_radial_density,
    plot_pairwise_distance_histogram,
    plot_null_hypothesis_test,
)
from .tda import (
    compute_persistent_homology,
    plot_persistence_diagram,
    plot_barcode,
    analyze_gap_with_tda,
    PersistenceResult,
)

__all__ = [
    "davis_wilson_map",
    "compute_cache_batch",
    "CacheResult",
    "analyze_cache_space",
    "compute_gap_visibility", 
    "ClusteringResult",
    "plot_cache_space_3d",
    "plot_gap_visibility",
    "plot_topological_sectors",
    "plot_radial_density",
    "plot_pairwise_distance_histogram",
    "plot_null_hypothesis_test",
    "compute_persistent_homology",
    "plot_persistence_diagram",
    "plot_barcode",
    "analyze_gap_with_tda",
    "PersistenceResult",
]
