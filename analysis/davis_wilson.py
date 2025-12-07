"""
The Davis-Wilson Map: Γ(A) = (Φ, r)

This is the core construction of the Davis-Wilson framework.
It maps gauge configurations to a "cache space" that captures
gauge-invariant information at a chosen resolution.

Components:
    Φ: Continuous cache (Wilson loop traces on skeleton)
    r: Discrete cache (topological charge / instanton number)

The key insight: Different bins in cache space must have different
integrated curvature, which costs energy → mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import h5py

from lattice import (
    GaugeConfig,
    load_config,
    compute_wilson_loop,
    compute_topological_charge,
    apply_smearing,
    apply_wilson_flow,
)
from lattice.skeleton import SkeletonSpec, build_skeleton


@dataclass
class CacheResult:
    """
    Result of the Davis-Wilson map for one configuration.
    
    Attributes:
        phi: Continuous cache vector (Wilson loop traces)
        r: Discrete cache (integer topological charge)
        q_raw: Raw (non-rounded) topological charge
        config_id: Identifier for the source configuration
    """
    phi: NDArray[np.float64]
    r: int
    q_raw: float
    config_id: Optional[str] = None
    
    @property
    def cache_dim(self) -> int:
        return len(self.phi)


def davis_wilson_map(
    config: GaugeConfig,
    skeleton: SkeletonSpec,
    flow_time: float = 1.0,
    use_flow: bool = True,
    smearing_steps: int = 10,
    smearing_rho: float = 0.1,
) -> CacheResult:
    """
    Compute the Davis-Wilson map Γ(A) = (Φ, r).
    
    CRITICAL: Both Φ and r MUST be computed at the same RG scale to avoid
    UV/IR mixing. We use Wilson flow (gradient flow) to define this scale.
    The resolution ε in the axioms corresponds to sqrt(8 * flow_time).
    
    Args:
        config: Gauge configuration
        skeleton: Skeleton specification defining which loops to measure
        flow_time: Wilson flow time (defines RG scale, ε ~ sqrt(8t))
        use_flow: If True, use Wilson flow. If False, fall back to APE smearing.
        smearing_steps: (legacy) APE smearing steps if use_flow=False
        smearing_rho: (legacy) APE smearing parameter if use_flow=False
    
    Returns:
        CacheResult containing Φ (continuous) and r (discrete) components
    
    Algorithm (with Wilson flow):
        1. Apply Wilson flow to configuration: U → V_t
        2. Compute Φ on flowed config V_t (consistent IR scale)
        3. Compute Q on flowed config V_t (same IR scale)
        4. r = round(Q)
    
    The mass gap is an IR phenomenon. By using Wilson flow, we:
        - Suppress UV fluctuations (lattice artifacts)
        - Define observables at a consistent physical scale
        - Ensure the "resolution ε" has a precise physical meaning
    """
    # Apply Wilson flow for consistent RG scale
    if use_flow and flow_time > 0:
        config_flowed = apply_wilson_flow(config, flow_time=flow_time)
    elif smearing_steps > 0:
        # Legacy fallback to APE smearing
        config_flowed = apply_smearing(config, n_steps=smearing_steps, rho=smearing_rho)
    else:
        config_flowed = config
    
    # Compute continuous cache Φ ON FLOWED CONFIG
    # Normalize by N=3 so |W| <= 1, making Φ comparable across volumes and β
    phi = np.zeros(skeleton.cache_dim, dtype=np.float64)
    
    for i, loop in enumerate(skeleton.loops):
        W = compute_wilson_loop(config_flowed, loop)
        phi[2 * i] = W.real / 3.0
        phi[2 * i + 1] = W.imag / 3.0
    
    # Compute discrete cache r ON SAME FLOWED CONFIG
    q_raw = compute_topological_charge(config_flowed)
    r = int(round(q_raw))
    
    return CacheResult(
        phi=phi,
        r=r,
        q_raw=q_raw,
        config_id=config.metadata.get("id", None)
    )


def compute_cache_batch(
    config_paths: List[Path],
    skeleton: SkeletonSpec,
    smearing_steps: int = 10,
    output_path: Optional[Path] = None,
    progress: bool = True,
) -> List[CacheResult]:
    """
    Compute Davis-Wilson map for a batch of configurations.
    
    Args:
        config_paths: Paths to configuration files
        skeleton: Skeleton specification
        smearing_steps: Smearing iterations
        output_path: If provided, save results to HDF5
        progress: Show progress bar
    
    Returns:
        List of CacheResult objects
    
    Output file format (if output_path provided):
        /Phi: (N, cache_dim) float64
        /r: (N,) int32
        /q_raw: (N,) float64
        /config_ids: (N,) string
    """
    results = []
    
    iterator = config_paths
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(config_paths, desc="Computing cache")
        except ImportError:
            pass
    
    for path in iterator:
        config = load_config(path)
        config.metadata["id"] = str(path)
        result = davis_wilson_map(config, skeleton, smearing_steps)
        results.append(result)
    
    # Save to HDF5 if requested
    if output_path is not None:
        save_cache_results(results, output_path)
    
    return results


def save_cache_results(results: List[CacheResult], path: Path) -> None:
    """Save cache results to HDF5 file."""
    n = len(results)
    cache_dim = results[0].cache_dim
    
    Phi = np.zeros((n, cache_dim), dtype=np.float64)
    r = np.zeros(n, dtype=np.int32)
    q_raw = np.zeros(n, dtype=np.float64)
    
    for i, res in enumerate(results):
        Phi[i] = res.phi
        r[i] = res.r
        q_raw[i] = res.q_raw
    
    with h5py.File(path, "w") as f:
        f.create_dataset("Phi", data=Phi, compression="gzip")
        f.create_dataset("r", data=r)
        f.create_dataset("q_raw", data=q_raw)
        f.attrs["n_configs"] = n
        f.attrs["cache_dim"] = cache_dim


def load_cache_results(path: Path) -> tuple[NDArray, NDArray, NDArray]:
    """
    Load cache results from HDF5 file.
    
    Returns:
        (Phi, r, q_raw) arrays
    """
    with h5py.File(path, "r") as f:
        Phi = f["Phi"][:]
        r = f["r"][:]
        q_raw = f["q_raw"][:]
    
    return Phi, r, q_raw


# ============================================================================
# Binning (discretization of continuous cache)
# ============================================================================

def compute_bins(
    Phi: NDArray[np.float64],
    r: NDArray[np.int32],
    epsilon: float = 0.1,
) -> NDArray[np.int32]:
    """
    Assign configurations to discrete bins based on cache values.
    
    Two configs are in the same bin iff:
        1. Same topological charge r
        2. ||Φ - Φ'|| < ε
    
    This implements the discretization map q_ε: C → B_ε
    
    Args:
        Phi: Continuous cache vectors, shape (N, d)
        r: Discrete cache (topological charge), shape (N,)
        epsilon: Binning resolution
    
    Returns:
        Bin labels, shape (N,)
    
    Algorithm:
        For each topological sector separately:
        - Use DBSCAN-like clustering with distance threshold ε
        - Or: quantize Φ to grid with spacing ε
    
    TODO: Implement proper binning
    """
    # COPILOT: Implement cache binning
    # Simple approach: hash-based binning
    # Better approach: hierarchical clustering with distance threshold
    
    from sklearn.cluster import DBSCAN
    
    n = len(r)
    bin_labels = np.zeros(n, dtype=np.int32)
    current_bin = 0
    
    # Process each topological sector separately
    for sector in np.unique(r):
        mask = r == sector
        Phi_sector = Phi[mask]
        
        if len(Phi_sector) == 0:
            continue
        
        # Cluster within sector
        clustering = DBSCAN(eps=epsilon, min_samples=1).fit(Phi_sector)
        sector_labels = clustering.labels_
        
        # Offset labels to be globally unique
        sector_labels = sector_labels + current_bin
        bin_labels[mask] = sector_labels
        current_bin = sector_labels.max() + 1
    
    return bin_labels
