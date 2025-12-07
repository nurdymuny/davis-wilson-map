"""
SU(3) matrix operations for lattice gauge theory.

SU(3) = Special Unitary group of 3×3 complex matrices
       { U ∈ ℂ^{3×3} : U†U = I, det(U) = 1 }

All functions are JIT-compiled with Numba for performance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numba import jit, prange

# Type alias for SU(3) matrix
SU3Matrix = NDArray[np.complex128]  # Shape: (3, 3)


@jit(nopython=True, cache=True)
def random_su3() -> SU3Matrix:
    """
    Generate a random SU(3) matrix using QR decomposition.
    
    Returns:
        Random 3×3 complex unitary matrix with determinant 1
    
    Algorithm:
        1. Generate random complex matrix with Gaussian entries
        2. QR decomposition gives unitary Q
        3. Fix determinant to be 1
    """
    # Generate random complex Gaussian matrix
    real_part = np.random.randn(3, 3)
    imag_part = np.random.randn(3, 3)
    Z = real_part + 1j * imag_part
    
    # QR decomposition - Q is unitary
    # Manual Gram-Schmidt for numba compatibility
    Q = np.zeros((3, 3), dtype=np.complex128)
    
    # First column
    v0 = Z[:, 0].copy()
    norm0 = np.sqrt(np.sum(np.abs(v0) ** 2))
    Q[:, 0] = v0 / norm0
    
    # Second column - orthogonalize against first
    v1 = Z[:, 1].copy()
    proj = np.sum(np.conj(Q[:, 0]) * v1)
    v1 = v1 - proj * Q[:, 0]
    norm1 = np.sqrt(np.sum(np.abs(v1) ** 2))
    Q[:, 1] = v1 / norm1
    
    # Third column - orthogonalize against first two
    v2 = Z[:, 2].copy()
    proj0 = np.sum(np.conj(Q[:, 0]) * v2)
    proj1 = np.sum(np.conj(Q[:, 1]) * v2)
    v2 = v2 - proj0 * Q[:, 0] - proj1 * Q[:, 1]
    norm2 = np.sqrt(np.sum(np.abs(v2) ** 2))
    Q[:, 2] = v2 / norm2
    
    # Fix determinant to be 1
    # det(Q) lies on unit circle; divide by cube root of det
    det_Q = (Q[0, 0] * (Q[1, 1] * Q[2, 2] - Q[1, 2] * Q[2, 1])
           - Q[0, 1] * (Q[1, 0] * Q[2, 2] - Q[1, 2] * Q[2, 0])
           + Q[0, 2] * (Q[1, 0] * Q[2, 1] - Q[1, 1] * Q[2, 0]))
    
    # Cube root of a complex number on unit circle
    phase = np.angle(det_Q) / 3.0
    correction = np.exp(-1j * phase)
    
    return Q * correction


@jit(nopython=True, cache=True)
def random_su3_near_identity(epsilon: float = 0.1) -> SU3Matrix:
    """
    Generate a random SU(3) matrix close to identity.
    
    Useful for Monte Carlo updates (Metropolis, HMC).
    
    Args:
        epsilon: Maximum deviation from identity (0 < epsilon < 1)
    
    Returns:
        SU(3) matrix U such that ||U - I|| < epsilon (approximately)
    
    Algorithm:
        1. Generate random Hermitian H with ||H|| < epsilon
        2. Return exp(iH) via Taylor series and project to SU(3)
    """
    # Generate random traceless anti-Hermitian matrix (su(3) algebra)
    # Use 8 Gell-Mann-like generators implicitly
    
    # Random real coefficients, scaled by epsilon
    h = epsilon * np.random.randn(8)
    
    # Construct anti-Hermitian traceless matrix iH
    # Using a simplified parameterization
    iH = np.zeros((3, 3), dtype=np.complex128)
    
    # Diagonal traceless part (2 generators)
    iH[0, 0] = 1j * h[0]
    iH[1, 1] = 1j * h[1]
    iH[2, 2] = -1j * (h[0] + h[1])  # Traceless constraint
    
    # Off-diagonal parts (6 generators)
    iH[0, 1] = h[2] + 1j * h[3]
    iH[1, 0] = -h[2] + 1j * h[3]
    iH[0, 2] = h[4] + 1j * h[5]
    iH[2, 0] = -h[4] + 1j * h[5]
    iH[1, 2] = h[6] + 1j * h[7]
    iH[2, 1] = -h[6] + 1j * h[7]
    
    # Compute exp(iH) via Taylor series
    result = np.eye(3, dtype=np.complex128)
    term = np.eye(3, dtype=np.complex128)
    
    for n in range(1, 15):  # Converges quickly for small epsilon
        term = np.dot(term, iH) / n
        result = result + term
        if np.max(np.abs(term)) < 1e-15:
            break
    
    # Project to exact SU(3)
    return project_to_su3(result)


@jit(nopython=True, cache=True)
def is_su3(U: SU3Matrix, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is in SU(3) within tolerance.
    
    Args:
        U: 3×3 complex matrix to check
        tol: Tolerance for unitarity and determinant checks
    
    Returns:
        True if U†U ≈ I and det(U) ≈ 1
    
    Checks:
        1. ||U†U - I||_F < tol (unitarity)
        2. |det(U) - 1| < tol (special)
    
    TODO: Implement
    """
    # COPILOT: Implement SU(3) validation
    # Check U†U = I
    UdU = np.dot(U.conj().T, U)
    unitarity_error = np.sqrt(np.sum(np.abs(UdU - np.eye(3)) ** 2))
    
    # Check det(U) = 1
    det_error = np.abs(np.linalg.det(U) - 1.0)
    
    return unitarity_error < tol and det_error < tol


@jit(nopython=True, cache=True)
def su3_dagger(U: SU3Matrix) -> SU3Matrix:
    """
    Compute the Hermitian conjugate U† of an SU(3) matrix.
    
    For SU(3), U† = U^{-1}, so this is also the inverse.
    
    Args:
        U: SU(3) matrix
    
    Returns:
        U† (conjugate transpose)
    """
    return U.conj().T


@jit(nopython=True, cache=True)
def su3_multiply(U: SU3Matrix, V: SU3Matrix) -> SU3Matrix:
    """
    Multiply two SU(3) matrices.
    
    Args:
        U, V: SU(3) matrices
    
    Returns:
        U · V (matrix product)
    
    Note: Result is exactly SU(3) if inputs are SU(3).
    """
    return np.dot(U, V)


@jit(nopython=True, cache=True)
def su3_trace(U: SU3Matrix) -> complex:
    """
    Compute the trace of an SU(3) matrix.
    
    Args:
        U: SU(3) matrix
    
    Returns:
        Tr(U) = U[0,0] + U[1,1] + U[2,2]
    
    Note: For SU(3), Re[Tr(U)] ∈ [-3, 3]
    """
    return U[0, 0] + U[1, 1] + U[2, 2]


@jit(nopython=True, cache=True)
def project_to_su3(M: SU3Matrix) -> SU3Matrix:
    """
    Project an arbitrary 3×3 matrix to SU(3).
    
    Useful after numerical operations that may break exact SU(3) structure.
    
    Args:
        M: Arbitrary 3×3 complex matrix
    
    Returns:
        Nearest SU(3) matrix in Frobenius norm
    
    Algorithm:
        1. SVD: M = U Σ V†
        2. Set W = U V†
        3. Fix determinant: W' = W / det(W)^{1/3}
    
    TODO: Implement
    """
    # COPILOT: Implement projection to SU(3)
    # Use SVD-based projection
    U_svd, S, Vh = np.linalg.svd(M)
    W = np.dot(U_svd, Vh)
    
    # Fix determinant
    det_W = np.linalg.det(W)
    phase = det_W / np.abs(det_W)
    W = W / (phase ** (1.0 / 3.0))
    
    return W


@jit(nopython=True, parallel=True, cache=True)
def project_config_to_su3(U: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Project all link variables in a gauge config to SU(3).
    
    Args:
        U: Gauge configuration, shape (4, L, L, L, L, 3, 3)
    
    Returns:
        Configuration with all links projected to exact SU(3)
    
    Use after HMC trajectory or loading from file to fix numerical drift.
    """
    L = U.shape[1]
    U_proj = np.empty_like(U)
    
    for mu in prange(4):
        for t in range(L):
            for x in range(L):
                for y in range(L):
                    for z in range(L):
                        U_proj[mu, t, x, y, z] = project_to_su3(U[mu, t, x, y, z])
    
    return U_proj


# ============================================================================
# SU(3) Lie Algebra (for HMC momentum)
# ============================================================================

@jit(nopython=True, cache=True)
def random_su3_algebra() -> SU3Matrix:
    """
    Generate a random element of the su(3) Lie algebra.
    
    su(3) = { H ∈ ℂ^{3×3} : H† = -H, Tr(H) = 0 }
    
    Returns:
        Random traceless anti-Hermitian 3×3 matrix
    
    Parameterization:
        H = i Σ_{a=1}^{8} h_a λ_a
    where λ_a are the Gell-Mann matrices and h_a ~ N(0,1)
    """
    # Random real coefficients
    h = np.random.randn(8)
    
    # Construct anti-Hermitian traceless matrix
    H = np.zeros((3, 3), dtype=np.complex128)
    
    # Diagonal traceless part (2 generators: λ_3, λ_8 style)
    H[0, 0] = 1j * h[0]
    H[1, 1] = 1j * h[1]
    H[2, 2] = -1j * (h[0] + h[1])  # Traceless constraint
    
    # Off-diagonal parts (6 generators)
    H[0, 1] = h[2] + 1j * h[3]
    H[1, 0] = -h[2] + 1j * h[3]
    H[0, 2] = h[4] + 1j * h[5]
    H[2, 0] = -h[4] + 1j * h[5]
    H[1, 2] = h[6] + 1j * h[7]
    H[2, 1] = -h[6] + 1j * h[7]
    
    return H


@jit(nopython=True, cache=True)
def su3_exp(H: SU3Matrix) -> SU3Matrix:
    """
    Compute the matrix exponential exp(H) where H ∈ su(3).
    
    Args:
        H: Traceless anti-Hermitian 3×3 matrix (su(3) algebra element)
    
    Returns:
        exp(H) ∈ SU(3)
    
    Algorithm options:
        1. Cayley-Hamilton: Explicit formula using characteristic polynomial
        2. Taylor series: Σ H^n / n! (truncate at machine precision)
        3. Scaling and squaring: exp(H) = exp(H/2^k)^{2^k}
    
    TODO: Implement Cayley-Hamilton for efficiency
    """
    # COPILOT: Implement SU(3) exponential
    # For now, use scipy-style Taylor series
    # Cayley-Hamilton is faster: see hep-lat/0311018
    
    result = np.eye(3, dtype=np.complex128)
    term = np.eye(3, dtype=np.complex128)
    
    for n in range(1, 20):  # Taylor series
        term = np.dot(term, H) / n
        result = result + term
        if np.max(np.abs(term)) < 1e-15:
            break
    
    return project_to_su3(result)  # Ensure exact SU(3)
