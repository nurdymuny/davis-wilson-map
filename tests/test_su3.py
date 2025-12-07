"""
Tests for SU(3) matrix operations.
"""

import numpy as np
import pytest

from lattice.su3 import (
    random_su3,
    is_su3,
    su3_dagger,
    su3_multiply,
    su3_trace,
    project_to_su3,
)


class TestSU3Basic:
    """Basic SU(3) matrix tests."""
    
    def test_identity_is_su3(self):
        """Identity matrix should be in SU(3)."""
        I = np.eye(3, dtype=np.complex128)
        assert is_su3(I)
    
    def test_random_su3_is_su3(self):
        """Random SU(3) should satisfy SU(3) properties."""
        for _ in range(10):
            U = random_su3()
            assert is_su3(U), "Random SU(3) failed validation"
    
    def test_su3_dagger_inverse(self):
        """U† should be the inverse of U for SU(3)."""
        U = random_su3()
        U_dag = su3_dagger(U)
        product = su3_multiply(U, U_dag)
        
        assert np.allclose(product, np.eye(3)), "U U† ≠ I"
    
    def test_su3_determinant_one(self):
        """SU(3) matrices should have determinant 1."""
        for _ in range(10):
            U = random_su3()
            det = np.linalg.det(U)
            assert np.isclose(det, 1.0), f"det(U) = {det} ≠ 1"
    
    def test_su3_trace_bounds(self):
        """Trace of SU(3) should have |Re[Tr]| ≤ 3."""
        for _ in range(100):
            U = random_su3()
            tr = su3_trace(U)
            assert -3 <= tr.real <= 3, f"Re[Tr(U)] = {tr.real} out of bounds"
    
    def test_projection_idempotent(self):
        """Projecting SU(3) should give back the same matrix."""
        U = random_su3()
        U_proj = project_to_su3(U)
        assert np.allclose(U, U_proj, atol=1e-10)
    
    def test_projection_fixes_non_su3(self):
        """Projection should convert arbitrary matrices to SU(3)."""
        M = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
        U = project_to_su3(M)
        assert is_su3(U, tol=1e-8)


class TestSU3Multiplication:
    """Tests for SU(3) group operations."""
    
    def test_closure(self):
        """Product of SU(3) matrices should be SU(3)."""
        U = random_su3()
        V = random_su3()
        W = su3_multiply(U, V)
        assert is_su3(W)
    
    def test_associativity(self):
        """Matrix multiplication should be associative."""
        U = random_su3()
        V = random_su3()
        W = random_su3()
        
        UV_W = su3_multiply(su3_multiply(U, V), W)
        U_VW = su3_multiply(U, su3_multiply(V, W))
        
        assert np.allclose(UV_W, U_VW)
    
    def test_identity_element(self):
        """Multiplying by identity should give original."""
        U = random_su3()
        I = np.eye(3, dtype=np.complex128)
        
        assert np.allclose(su3_multiply(U, I), U)
        assert np.allclose(su3_multiply(I, U), U)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
