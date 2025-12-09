"""
Tests for analysis modules.
"""

import numpy as np
import pytest


class TestHeatKernel:
    """Tests for heat kernel analysis."""
    
    def test_laplacian_construction(self):
        """Graph Laplacian should be valid sparse matrix."""
        from scipy import sparse
        from analysis.unified_heat_kernel import build_laplacian
        
        np.random.seed(42)
        points = np.random.randn(100, 5)
        
        L = build_laplacian(points, k=10, t=1.0)
        
        # Should be sparse
        assert sparse.issparse(L)
        
        # Should be square
        assert L.shape[0] == L.shape[1] == 100
        
        # Normalized Laplacian has diagonal elements close to 1
        diag = L.diagonal()
        assert np.allclose(diag, 1.0, atol=0.01)
    
    def test_laplacian_symmetric(self):
        """Normalized graph Laplacian should be symmetric."""
        from analysis.unified_heat_kernel import build_laplacian
        
        np.random.seed(42)
        points = np.random.randn(50, 3)
        
        L = build_laplacian(points, k=10, t=1.0)
        
        # Check symmetry
        diff = (L - L.T).toarray()
        assert np.allclose(diff, 0, atol=1e-10)
    
    def test_spectrum_eigenvalues_nonnegative(self):
        """Laplacian eigenvalues should be non-negative."""
        from analysis.unified_heat_kernel import build_laplacian, compute_heat_kernel_spectrum
        
        np.random.seed(42)
        points = np.random.randn(100, 5)
        
        L = build_laplacian(points, k=10, t=1.0)
        eigenvalues, _ = compute_heat_kernel_spectrum(L, n_eigs=20)
        
        # All eigenvalues should be >= 0 (up to numerical tolerance)
        assert np.all(eigenvalues >= -1e-8)
    
    def test_smallest_eigenvalue_bounded(self):
        """Smallest eigenvalues should be bounded for normalized Laplacian."""
        from analysis.unified_heat_kernel import build_laplacian, compute_heat_kernel_spectrum
        
        np.random.seed(42)
        # Create well-connected cloud
        points = np.random.randn(100, 5)
        
        L = build_laplacian(points, k=20, t=1.0)
        eigenvalues, _ = compute_heat_kernel_spectrum(L, n_eigs=10)
        
        # Eigenvalues should be bounded in [0, 2] for normalized Laplacian
        assert np.all(eigenvalues >= -0.1)
        assert np.all(eigenvalues <= 2.1)


class TestClustering:
    """Tests for clustering analysis."""
    
    def test_synthetic_two_clusters(self):
        """Two well-separated clusters should be detected."""
        hdbscan = pytest.importorskip("hdbscan")
        from analysis.clustering import analyze_cache_space
        
        np.random.seed(42)
        
        # Two clusters 10 units apart
        cluster1 = np.random.randn(100, 5) + np.array([5, 0, 0, 0, 0])
        cluster2 = np.random.randn(100, 5) + np.array([-5, 0, 0, 0, 0])
        
        Phi = np.vstack([cluster1, cluster2])
        r = np.zeros(200, dtype=np.int32)
        
        result = analyze_cache_space(Phi, r)
        
        assert result.n_clusters >= 2
    
    def test_gap_visibility_range(self):
        """Gap visibility should be in [0, 1] or similar bounded range."""
        hdbscan = pytest.importorskip("hdbscan")
        from analysis.clustering import analyze_cache_space
        
        np.random.seed(42)
        Phi = np.random.randn(100, 5)
        r = np.zeros(100, dtype=np.int32)
        
        result = analyze_cache_space(Phi, r)
        
        # Gap visibility should be non-negative
        assert result.gap_visibility >= 0


class TestDavisWilson:
    """Tests for Davis-Wilson map."""
    
    def test_result_structure(self):
        """Davis-Wilson result should have expected fields."""
        from lattice import cold_start
        from lattice.skeleton import build_skeleton
        from analysis.davis_wilson import davis_wilson_map
        
        config = cold_start(L=4, beta=6.0)
        skeleton = build_skeleton(4, stride=1)
        
        result = davis_wilson_map(config, skeleton, smearing_steps=0)
        
        # Check required fields exist
        assert hasattr(result, 'phi')
        assert hasattr(result, 'r')
        assert hasattr(result, 'q_raw')
        
        # Phi should be a numpy array
        assert isinstance(result.phi, np.ndarray)
        assert len(result.phi) > 0


class TestHeatKernelAnalyzer:
    """Tests for the full heat kernel analyzer classes."""
    
    def test_heat_kernel_result_methods(self):
        """HeatKernelResult should have working methods."""
        from analysis.heat_kernel import HeatKernelResult
        
        # Create mock result
        eigenvalues = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
        eigenvectors = np.eye(5)
        
        result = HeatKernelResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            spectral_gap=0.1,
            spectral_gaps=np.diff(eigenvalues),
            heat_trace_values=np.array([5.0, 3.0, 1.0]),
            time_scales=np.array([0.0, 1.0, 10.0]),
            effective_dimensions=np.array([5.0, 3.0, 1.0]),
            n_eigenvalues=5,
            n_points=5,
        )
        
        # Test heat_trace
        Z_0 = result.heat_trace(0.0)
        assert np.isclose(Z_0, 5.0)  # All weights = 1
        
        Z_large = result.heat_trace(100.0)
        assert Z_large < 2.0  # Most modes decayed
        
        # Test effective_dimension
        d_0 = result.effective_dimension(0.0)
        assert d_0 > 1.0  # Multiple modes active at t=0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
