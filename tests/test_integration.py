"""
Integration tests for the full Davis-Wilson pipeline.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path


class TestFullPipeline:
    """End-to-end pipeline tests on small lattices."""
    
    def test_cold_config_cache(self):
        """Cold (identity) config should have Q=0 and specific Φ."""
        from lattice import cold_start
        from lattice.skeleton import build_skeleton
        from analysis.davis_wilson import davis_wilson_map
        
        config = cold_start(L=4, beta=6.0)
        skeleton = build_skeleton(4, stride=1)
        
        result = davis_wilson_map(config, skeleton, smearing_steps=0)
        
        # Identity plaquettes have trace = 3
        # Wilson loop of identity links = identity = trace 3
        assert result.r == 0, f"Cold config should have r=0, got {result.r}"
        assert abs(result.q_raw) < 0.1, f"Cold config should have Q≈0, got {result.q_raw}"
        
        # Wilson loops are normalized by /3, so identity gives 1.0
        # Φ should be (1, 0, 1, 0, 1, 0, ...) for (Re, Im) pairs
        for i in range(0, len(result.phi), 2):
            assert abs(result.phi[i] - 1.0) < 0.1, f"Re[W]/3 should be 1, got {result.phi[i]}"
            assert abs(result.phi[i+1]) < 0.1, f"Im[W]/3 should be 0, got {result.phi[i+1]}"
    
    def test_skeleton_size_scaling(self):
        """Skeleton size should scale as expected with stride."""
        from lattice.skeleton import build_skeleton, estimate_skeleton_size
        
        L = 8
        
        for stride in [1, 2, 4]:
            est = estimate_skeleton_size(L, stride)
            skeleton = build_skeleton(L, stride)
            
            assert skeleton.n_loops == est["n_loops"]
            assert skeleton.cache_dim == est["cache_dim"]
    
    def test_hierarchical_skeleton(self):
        """Multi-level skeleton should have more loops than single level."""
        from lattice.skeleton import build_skeleton, build_hierarchical_skeleton
        
        L = 16
        
        simple = build_skeleton(L, stride=4)
        hierarchical = build_hierarchical_skeleton(L, base_stride=4, levels=2)
        
        assert hierarchical.n_loops > simple.n_loops
    
    def test_save_load_roundtrip(self):
        """Saving and loading config should preserve data."""
        from lattice import cold_start, save_config, load_config
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.h5"
            
            original = cold_start(L=4, beta=5.5)
            original.metadata["test_key"] = "test_value"
            
            save_config(original, path)
            loaded = load_config(path)
            
            assert loaded.L == original.L
            assert loaded.beta == original.beta
            assert np.allclose(loaded.U, original.U)
    
    def test_clustering_synthetic_clustered(self):
        """Synthetic clustered data should give G > 0."""
        pytest.importorskip("hdbscan")
        from analysis.clustering import analyze_cache_space
        
        # Create synthetic data with clear clusters
        np.random.seed(42)
        
        # Two well-separated clusters
        cluster1 = np.random.randn(100, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(100, 10) + np.array([-5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        Phi = np.vstack([cluster1, cluster2])
        r = np.array([0] * 100 + [0] * 100)  # Same sector
        
        result = analyze_cache_space(Phi, r)
        
        assert result.n_clusters >= 2, f"Should find at least 2 clusters, got {result.n_clusters}"
        assert result.gap_visibility > 0.5, f"Gap visibility should be high, got {result.gap_visibility}"
    
    def test_clustering_synthetic_continuous(self):
        """Synthetic uniform data should give G ≈ 0."""
        pytest.importorskip("hdbscan")
        from analysis.clustering import analyze_cache_space
        
        # Create synthetic uniform (no structure) data
        np.random.seed(42)
        Phi = np.random.randn(200, 10)
        r = np.zeros(200, dtype=np.int32)
        
        result = analyze_cache_space(
            Phi, r,
            hdbscan_min_cluster_size=20,
            hdbscan_min_samples=5,
        )
        
        # Uniform data should either have 1 cluster or very low gap visibility
        assert result.n_clusters <= 2 or result.gap_visibility < 1.0


class TestWilsonLoops:
    """Tests for Wilson loop computation."""
    
    def test_plaquette_trace_bounds(self):
        """Plaquette trace should be in valid range."""
        from lattice import cold_start
        from lattice.wilson_loops import plaquette_trace
        
        config = cold_start(L=4, beta=6.0)
        
        for t in range(4):
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        for mu in range(3):
                            for nu in range(mu+1, 4):
                                tr = plaquette_trace(config.U, t, x, y, z, mu, nu)
                                # For SU(3), trace should be real and in [-3, 3]
                                # For identity, should be exactly 3
                                assert -3.01 <= tr.real <= 3.01
    
    def test_average_plaquette_cold(self):
        """Cold config should have average plaquette = 3."""
        from lattice import cold_start
        from lattice.wilson_loops import average_plaquette
        
        config = cold_start(L=4, beta=6.0)
        avg = average_plaquette(config.U)
        
        assert abs(avg - 3.0) < 0.01, f"Cold config avg plaq should be 3, got {avg}"


class TestTopologicalCharge:
    """Tests for topological charge computation."""
    
    def test_trivial_config_zero_charge(self):
        """Cold (trivial) config should have Q = 0."""
        from lattice import cold_start
        from lattice.topological import compute_topological_charge
        
        config = cold_start(L=4, beta=6.0)
        Q = compute_topological_charge(config)
        
        assert abs(Q) < 0.1, f"Trivial config should have Q≈0, got {Q}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
