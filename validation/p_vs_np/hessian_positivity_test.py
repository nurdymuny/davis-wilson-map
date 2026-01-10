"""
Hessian Positivity Test for Kac-Rice Proof
===========================================

This script numerically tests the conditional step in Lemma C:
    At critical points with energy density e, is the spherical Hessian positive definite?
    
Specifically, we test whether:
    ∇²_S E ≈ GOE(σ²) + λ(e)·I  with λ(e) > 2σ

Author: Bee Rosa Davis
Date: January 2026
"""

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from contextlib import nullcontext
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# GPU support via PyTorch
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Optional: mixed precision can speed up HVPs on supported GPUs.
# Default OFF for numerical safety in eigenvalue tests.
USE_AMP = False
AMP_DTYPE = torch.bfloat16  # bfloat16 is usually safer than float16

def _amp_ctx():
    if (not USE_AMP) or (device.type != "cuda"):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)

if device.type == "cuda":
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


@dataclass
class SATInstance:
    """A random 3-SAT instance."""
    n: int                      # number of variables
    clauses: np.ndarray         # shape (m, 3) - variable indices per clause
    signs: np.ndarray           # shape (m, 3) - signs ±1 per literal
    
    @property
    def m(self) -> int:
        return len(self.clauses)
    
    @property
    def alpha(self) -> float:
        return self.m / self.n


def generate_random_3sat(n: int, alpha: float, seed: int = None) -> SATInstance:
    """Generate a random 3-SAT instance with m = alpha * n clauses."""
    if seed is not None:
        np.random.seed(seed)
    
    m = int(alpha * n)
    clauses = np.zeros((m, 3), dtype=int)
    signs = np.zeros((m, 3), dtype=int)
    
    for i in range(m):
        # Choose 3 distinct variables uniformly
        clauses[i] = np.random.choice(n, size=3, replace=False)
        # Choose signs uniformly ±1
        signs[i] = np.random.choice([-1, 1], size=3)
    
    return SATInstance(n=n, clauses=clauses, signs=signs)


class SATEnergy:
    """
    Unsquared SAT-consistent energy on the sphere.
    
    Energy model (LOCKED - unsquared):
        E_C(x) = P_C^{viol}(x) = ∏_{l ∈ C} (1 - σ_l x_l) / 2
        E(x) = Σ_C E_C(x)
    
    With tanh chart:
        x_i(s) = tanh(β s_i)
        E(s) = E(x(s))
    """
    
    def __init__(self, instance: SATInstance, beta: float = 1.0):
        self.instance = instance
        self.beta = beta
        self.n = instance.n
        self.m = instance.m
        self.clauses = instance.clauses
        self.signs = instance.signs
    
    def x_from_s(self, s: np.ndarray) -> np.ndarray:
        """Tanh chart: x_i = tanh(β s_i)"""
        return np.tanh(self.beta * s)
    
    def dx_ds(self, s: np.ndarray) -> np.ndarray:
        """Derivative of chart: dx/ds = β sech²(β s)"""
        return self.beta / np.cosh(self.beta * s)**2
    
    def d2x_ds2(self, s: np.ndarray) -> np.ndarray:
        """Second derivative: d²x/ds² = -2β² tanh(βs) sech²(βs)"""
        return -2 * self.beta**2 * np.tanh(self.beta * s) / np.cosh(self.beta * s)**2
    
    def clause_energy(self, x: np.ndarray, clause_idx: int) -> float:
        """Compute E_C = ∏_l (1 - σ_l x_l) / 2 for one clause."""
        c = self.clauses[clause_idx]
        sig = self.signs[clause_idx]
        u = (1 - sig * x[c]) / 2  # u_l for each literal
        return np.prod(u)
    
    def energy_x(self, x: np.ndarray) -> float:
        """Total energy in x-coordinates."""
        E = 0.0
        for i in range(self.m):
            E += self.clause_energy(x, i)
        return E
    
    def energy(self, s: np.ndarray) -> float:
        """Total energy on sphere (s-coordinates)."""
        x = self.x_from_s(s)
        return self.energy_x(x)
    
    def grad_x(self, x: np.ndarray) -> np.ndarray:
        """Gradient in x-coordinates: ∂E/∂x_a"""
        grad = np.zeros(self.n)
        
        for i in range(self.m):
            c = self.clauses[i]
            sig = self.signs[i]
            u = (1 - sig * x[c]) / 2
            
            # E_C = u_0 * u_1 * u_2
            # ∂E_C/∂x_{c[j]} = -σ_j/2 * ∏_{k≠j} u_k
            for j in range(3):
                other_u = np.prod([u[k] for k in range(3) if k != j])
                grad[c[j]] += (-sig[j] / 2) * other_u
        
        return grad
    
    def grad_s(self, s: np.ndarray) -> np.ndarray:
        """Euclidean gradient in s-coordinates (chain rule)."""
        x = self.x_from_s(s)
        grad_x = self.grad_x(x)
        dxds = self.dx_ds(s)
        return grad_x * dxds
    
    def spherical_grad(self, s: np.ndarray) -> np.ndarray:
        """Spherical gradient: P_s ∇E where P_s = I - ss^T/|s|²"""
        grad = self.grad_s(s)
        norm_sq = np.dot(s, s)
        proj = grad - s * (np.dot(s, grad) / norm_sq)
        return proj
    
    def hessian_x(self, x: np.ndarray) -> np.ndarray:
        """Hessian in x-coordinates: ∂²E/∂x_a ∂x_b"""
        H = np.zeros((self.n, self.n))
        
        for i in range(self.m):
            c = self.clauses[i]
            sig = self.signs[i]
            u = (1 - sig * x[c]) / 2
            
            # For a,b both in clause (a ≠ b):
            # ∂²E_C/∂x_a ∂x_b = (σ_a σ_b / 4) * u_c (the third variable)
            # Loop j < k to avoid double-counting, then fill symmetrically
            for j in range(3):
                for k in range(j + 1, 3):
                    third = 3 - j - k
                    val = (sig[j] * sig[k] / 4) * u[third]
                    H[c[j], c[k]] += val
                    H[c[k], c[j]] += val
        
        return H
    
    def hessian_s(self, s: np.ndarray) -> np.ndarray:
        """Euclidean Hessian in s-coordinates (chain rule)."""
        x = self.x_from_s(s)
        H_x = self.hessian_x(x)
        grad_x = self.grad_x(x)
        dxds = self.dx_ds(s)
        d2xds2 = self.d2x_ds2(s)
        
        # H_s[a,b] = H_x[a,b] * dx_a/ds_a * dx_b/ds_b  (for a ≠ b)
        # H_s[a,a] = H_x[a,a] * (dx_a/ds_a)² + grad_x[a] * d²x_a/ds²_a
        
        H_s = np.outer(dxds, dxds) * H_x
        H_s += np.diag(grad_x * d2xds2)
        
        return H_s
    
    def spherical_hessian(self, s: np.ndarray) -> np.ndarray:
        """
        Spherical (Riemannian) Hessian on tangent space.
        
        At a point s on the sphere, the Riemannian Hessian is:
            H_R = P_s H_s P_s - μ P_s
        where μ = (s · ∇E) / ||s||² is the Lagrange multiplier from ∇E = μs.
        
        Returns (H_proj, mu) where H_proj is n×n (has one zero eigenvalue in s direction).
        """
        H_s = self.hessian_s(s)
        grad_s = self.grad_s(s)
        norm_sq = np.dot(s, s)
        
        # Lagrange multiplier: at critical point, ∇E = μs
        mu = np.dot(s, grad_s) / norm_sq
        
        # Projector onto tangent space: P_s = I - ss^T/|s|²
        P_s = np.eye(self.n) - np.outer(s, s) / norm_sq
        
        # Riemannian Hessian: H_R = P H P - μ P
        H_proj = P_s @ H_s @ P_s - mu * P_s
        
        return H_proj, mu


class SATEnergyGPU:
    """GPU-accelerated SAT energy using PyTorch autograd."""
    
    def __init__(self, instance: SATInstance, beta: float = 1.0):
        self.n = instance.n
        self.m = instance.m
        self.beta = beta
        # Move clause data to GPU
        self.clauses = torch.tensor(instance.clauses, device=device, dtype=torch.long)
        self.signs = torch.tensor(instance.signs, device=device, dtype=torch.float32)
        
        # Try to compile energy function for faster HVP (PyTorch 2.x)
        # Skip on Windows where Triton is not available
        import sys
        if sys.platform != 'win32':
            try:
                self.energy_torch = torch.compile(self.energy_torch)
            except Exception:
                pass  # Fall back to eager mode if compile unavailable
    
    def energy_torch(self, s: torch.Tensor) -> torch.Tensor:
        """Compute energy with autograd support."""
        x = torch.tanh(self.beta * s)
        # Gather variables for each clause: shape (m, 3)
        x_clause = x[self.clauses]
        # u_l = (1 - σ_l x_l) / 2
        u = (1 - self.signs * x_clause) / 2
        # E_C = prod over literals
        E_clause = torch.prod(u, dim=1)
        return torch.sum(E_clause)
    
    def find_critical_point_gpu(self, s0: torch.Tensor = None, 
                                 lr: float = 0.5, max_iter: int = 5000,
                                 tol: float = 1e-4) -> Tuple[torch.Tensor, bool]:
        """Find critical point via projected gradient descent on GPU."""
        radius = torch.sqrt(torch.tensor(float(self.n), device=device))
        
        if s0 is None:
            s = torch.randn(self.n, device=device, dtype=torch.float32)
        else:
            s = s0.clone()
        s = s * (radius / torch.norm(s))
        
        best_s = s.clone()
        best_grad_norm = float('inf')
        
        for t in range(max_iter):
            s = s.requires_grad_(True)
            E = self.energy_torch(s)
            
            # Use autograd.grad instead of backward to avoid accumulation issues
            grad = torch.autograd.grad(E, s, create_graph=False)[0]
            
            # Spherical gradient
            norm_sq = torch.dot(s, s)
            grad_sph = grad - s * (torch.dot(s, grad) / norm_sq)
            grad_norm = torch.norm(grad_sph).item()
            
            # Track best point
            if grad_norm < best_grad_norm:
                best_grad_norm = grad_norm
                best_s = s.detach().clone()
            
            if grad_norm < tol:
                break
            
            # Gradient descent with LR decay
            lr_t = lr / (1 + 0.001 * t)
            with torch.no_grad():
                s = s - lr_t * grad_sph
                # Project back to sphere
                s = s * (radius / torch.norm(s))
        
        success = best_grad_norm < 1e-3
        return best_s, success
    
    def check_chart_saturation(self, s: torch.Tensor, threshold: float = 1e-3) -> Tuple[float, bool]:
        """
        Check for tanh chart saturation.
        
        Returns (mean_sech2, is_saturated) where is_saturated=True if 
        mean(sech²(βs)) < threshold (most variables near ±1).
        
        Threshold 1e-3 only rejects totally degenerate chart regimes.
        """
        sech2 = 1.0 / torch.cosh(self.beta * s) ** 2
        mean_sech2 = torch.mean(sech2).item()
        is_saturated = mean_sech2 < threshold
        return mean_sech2, is_saturated
    
    def spherical_hessian_gpu(self, s: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Compute spherical (Riemannian) Hessian via torch.func (vectorized).
        
        Returns (H_proj as torch tensor on GPU, mu, grad_norm).
        """
        s_det = s.detach().clone()
        
        # Use torch.func for vectorized Hessian (single GPU kernel)
        def energy_fn(x):
            return self.energy_torch(x)
        
        H = torch.func.hessian(energy_fn)(s_det)
        grad = torch.func.grad(energy_fn)(s_det)
        
        norm_sq = torch.dot(s_det, s_det)
        grad_sph = grad - s_det * (torch.dot(s_det, grad) / norm_sq)
        grad_norm = torch.norm(grad_sph).item()
        
        # Lagrange multiplier: μ = (s · ∇E) / ||s||²
        mu = torch.dot(s_det, grad) / norm_sq
        
        # Projector onto tangent space
        P_s = torch.eye(self.n, device=device) - torch.outer(s_det, s_det) / norm_sq
        
        # Riemannian Hessian: H_R = P H P - μ P
        H_proj = P_s @ H @ P_s - mu * P_s
        
        return H_proj, mu.item(), grad_norm
    
    def hessian_vector_product(self, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian-vector product H @ v via double autograd.
        
        This is O(1) backward passes instead of O(n) for full Hessian.
        """
        # Use torch.autograd.functional.hvp when available; it is often faster/leaner
        # than manual reverse-over-reverse for repeated HVP calls.
        s_in = s.detach().clone().requires_grad_(True)
        v_in = v.detach()

        def f(z):
            return self.energy_torch(z)

        with _amp_ctx():
            try:
                # Returns (f(s), H(s) @ v)
                _, hv = torch.autograd.functional.hvp(f, s_in, v_in, strict=True)
                return hv
            except Exception:
                # Fallback: manual reverse-over-reverse
                E = f(s_in)
                grad = torch.autograd.grad(E, s_in, create_graph=True)[0]
                hv = torch.autograd.grad(torch.dot(grad, v_in), s_in, retain_graph=False)[0]
                return hv
    
    def spherical_hvp(self, s: torch.Tensor, v: torch.Tensor, mu: float = None) -> torch.Tensor:
        """
        Spherical Hessian-vector product: H_R @ v = P(H @ P @ v) - μ * P @ v
        
        Where P is the tangent space projector and μ is the Lagrange multiplier.
        If mu is None, computes it from gradient.
        """
        # NOTE: callers in Lanczos always pass mu; keep mu=None path for completeness.
        norm_sq = torch.dot(s, s)
        
        # Project v onto tangent space: Pv = v - s(s·v)/||s||²
        Pv = v - s * (torch.dot(s, v) / norm_sq)
        
        # Compute H @ Pv
        with _amp_ctx():
            H_Pv = self.hessian_vector_product(s, Pv)
        
        # Project result: P @ H @ Pv
        P_H_Pv = H_Pv - s * (torch.dot(s, H_Pv) / norm_sq)
        
        # Get mu if not provided
        if mu is None:
            with _amp_ctx():
                s_grad = s.detach().clone().requires_grad_(True)
                E = self.energy_torch(s_grad)
                grad = torch.autograd.grad(E, s_grad)[0]
                mu = (torch.dot(s, grad) / norm_sq).item()
        
        # H_R @ v = P @ H @ Pv - μ * Pv
        return P_H_Pv - mu * Pv


def project_to_sphere(s: np.ndarray, radius: float) -> np.ndarray:
    """Project s onto the sphere of given radius."""
    return s * (radius / np.linalg.norm(s))


def build_tangent_basis(s: np.ndarray) -> np.ndarray:
    """
    Build orthonormal basis Q for tangent space at s.
    
    Returns Q of shape (n, n-1) such that Q^T s = 0 and Q^T Q = I.
    Uses projected-random-QR for numerical stability.
    """
    n = len(s)
    s_normalized = s / np.linalg.norm(s)
    
    # Generate random matrix and project columns to tangent space
    A = np.random.randn(n, n - 1)
    A -= np.outer(s_normalized, s_normalized @ A)  # project columns to tangent
    
    # QR gives orthonormal tangent basis
    Q, _ = np.linalg.qr(A)
    
    # Re-project to ensure orthogonality to s (numerical safety)
    Q -= np.outer(s_normalized, s_normalized @ Q)
    Q, _ = np.linalg.qr(Q)
    
    return Q


def tangent_eigenvalues(H_proj: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Get stable (n-1) eigenvalues of projected Hessian in tangent basis.
    """
    Q = build_tangent_basis(s)
    H_tangent = Q.T @ H_proj @ Q
    return np.linalg.eigvalsh(H_tangent)


def tangent_eigenvalues_gpu(H_proj: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Get stable (n-1) eigenvalues of projected Hessian in tangent basis (GPU).
    Uses projected-random-QR for numerical stability.
    """
    n = s.shape[0]
    s_normalized = s / torch.norm(s)
    
    # Generate random matrix and project columns to tangent space
    A = torch.randn(n, n - 1, device=device)
    A -= torch.outer(s_normalized, s_normalized @ A)  # project columns to tangent
    
    # QR gives orthonormal tangent basis
    Q_tangent, _ = torch.linalg.qr(A)
    
    # Re-project to ensure orthogonality to s (numerical safety)
    Q_tangent -= torch.outer(s_normalized, s_normalized @ Q_tangent)
    Q_tangent, _ = torch.linalg.qr(Q_tangent)
    
    H_tangent = Q_tangent.T @ H_proj @ Q_tangent
    return torch.linalg.eigvalsh(H_tangent)


def lanczos_extremal_eigenvalues(hvp_fn, n: int, s: torch.Tensor, 
                                  k: int = 30, tol: float = 1e-10,
                                  check_symmetry: bool = False) -> Tuple[float, float]:
    """
    Lanczos iteration to find extremal eigenvalues of spherical Hessian.
    
    Args:
        hvp_fn: function v -> H_R @ v (spherical Hessian-vector product)
        n: dimension
        s: point on sphere (for tangent space projection of initial vector)
        k: number of Lanczos iterations (typically 20-50 suffices)
        tol: absolute convergence tolerance for beta (early-stop heuristic; scale-dependent)
        check_symmetry: if True, verify operator symmetry (diagnostic)
    
    Returns:
        (min_eig, max_eig) estimates
    """
    norm_sq = torch.dot(s, s)
    
    # Optional symmetry check
    if check_symmetry:
        v1 = torch.randn(n, device=device)
        v2 = torch.randn(n, device=device)
        v1 = v1 - s * (torch.dot(s, v1) / norm_sq)
        v2 = v2 - s * (torch.dot(s, v2) / norm_sq)
        v1 = v1 / (torch.norm(v1) + 1e-12)
        v2 = v2 / (torch.norm(v2) + 1e-12)
        a = torch.dot(v1, hvp_fn(v2))
        b = torch.dot(v2, hvp_fn(v1))
        den = (a.abs() + b.abs() + 1e-12).item()
        rel = ((a - b).abs() / den).item()
        if rel > 1e-6:
            print(f"  WARNING: Operator symmetry rel error = {rel:.2e}")
    
    # Initialize with random tangent vector
    v = torch.randn(n, device=device)
    v = v - s * (torch.dot(s, v) / norm_sq)  # project to tangent
    v = v / torch.norm(v)
    
    # Lanczos vectors and tridiagonal matrix
    V = torch.zeros(n, k, device=device)
    alpha = torch.zeros(k, device=device)
    beta = torch.zeros(k - 1, device=device)
    
    V[:, 0] = v
    
    # Performance: full reorthogonalization is O(k^2 n).
    # Use cheap 3-term recurrence always, and do a full reorth only periodically.
    REORTH_EVERY = 5
    
    for j in range(k):
        # w = H @ v_j
        w = hvp_fn(V[:, j])
        
        # alpha_j = v_j · w
        alpha[j] = torch.dot(V[:, j], w)
        
        # Orthogonalize: w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        w = w - alpha[j] * V[:, j]
        if j > 0:
            w = w - beta[j - 1] * V[:, j - 1]
        
        # Periodic full reorthogonalization (much cheaper overall)
        if (REORTH_EVERY > 0) and (j % REORTH_EVERY == 0):
            # Full reorth against all previous vectors to control drift
            for i in range(j + 1):
                w = w - torch.dot(V[:, i], w) * V[:, i]
        else:
            # Skip full reorth; rely on 3-term recurrence
            # (still stable enough for k~30-50 in most cases)
            pass
        
        # CRITICAL: Project back to tangent space (Lanczos must stay in tangent)
        w = w - s * (torch.dot(s, w) / norm_sq)
        
        # beta_j = ||w||
        beta_j = torch.norm(w)
        
        if j < k - 1:
            if beta_j < tol:
                # Invariant subspace found, stop early
                k = j + 1
                break
            beta[j] = beta_j
            V[:, j + 1] = w / beta_j
            # Re-project to tangent and renormalize (numerical safety)
            V[:, j + 1] -= s * (torch.dot(s, V[:, j + 1]) / norm_sq)
            V[:, j + 1] /= torch.norm(V[:, j + 1])
    
    # Build tridiagonal matrix T and find its eigenvalues
    T = torch.diag(alpha[:k])
    if k > 1:
        T += torch.diag(beta[:k-1], 1) + torch.diag(beta[:k-1], -1)
    
    eigs = torch.linalg.eigvalsh(T)
    return eigs[0].item(), eigs[-1].item()


def fast_min_eigenvalue_gpu(energy_gpu: 'SATEnergyGPU', s: torch.Tensor, 
                            k: int = 30, mu: float = None, 
                            grad_norm: float = None,
                            check_symmetry: bool = False) -> Tuple[float, float, float, float]:
    """
    Fast check for Hessian positivity using Lanczos (no full Hessian).
    
    Args:
        energy_gpu: the energy object
        s: point on sphere
        k: Lanczos iterations
        mu: precomputed Lagrange multiplier (optional, avoids recomputation)
        grad_norm: precomputed spherical grad norm (optional)
        check_symmetry: if True, verify operator symmetry (diagnostic)
    
    Returns: (min_eig, max_eig, mu, grad_norm)
    """
    norm_sq = torch.dot(s, s)
    
    # Get mu and grad_norm if not provided
    if mu is None or grad_norm is None:
        s_grad = s.clone().requires_grad_(True)
        E = energy_gpu.energy_torch(s_grad)
        grad = torch.autograd.grad(E, s_grad)[0]
        grad_sph = grad - s * (torch.dot(s, grad) / norm_sq)
        if grad_norm is None:
            grad_norm = torch.norm(grad_sph).item()
        if mu is None:
            mu = (torch.dot(s, grad) / norm_sq).item()
    
    # Build HVP function with captured mu
    def hvp_fn(v):
        return energy_gpu.spherical_hvp(s, v, mu=mu)
    
    # Lanczos for extremal eigenvalues
    min_eig, max_eig = lanczos_extremal_eigenvalues(
        hvp_fn, energy_gpu.n, s, k=k, check_symmetry=check_symmetry
    )
    
    return min_eig, max_eig, mu, grad_norm


def find_critical_point(energy: SATEnergy, s0: np.ndarray = None, 
                        max_iter: int = 1000) -> Tuple[np.ndarray, bool]:
    """
    Find a critical point of E on the sphere using projected gradient descent.
    
    Returns (s, success) where s is on the sphere S^{n-1}(√n).
    """
    n = energy.n
    radius = np.sqrt(n)
    
    if s0 is None:
        s0 = np.random.randn(n)
    s0 = project_to_sphere(s0, radius)
    
    # Use scipy minimize with spherical constraint
    def objective(s):
        return energy.energy(s)
    
    def grad(s):
        return energy.grad_s(s)
    
    # Constraint: |s|² = n, with explicit Jacobian
    constraint = {
        'type': 'eq', 
        'fun': lambda s: np.dot(s, s) - n,
        'jac': lambda s: 2 * s
    }
    
    result = minimize(
        objective, s0, 
        method='SLSQP',
        jac=grad,
        constraints=constraint,
        options={'maxiter': max_iter, 'ftol': 1e-10}
    )
    
    s = project_to_sphere(result.x, radius)
    
    # Check if actually critical (spherical gradient ≈ 0)
    sph_grad = energy.spherical_grad(s)
    grad_norm = np.linalg.norm(sph_grad)
    success = grad_norm < 1e-5
    
    return s, success


def analyze_hessian_at_critical_point(energy: SATEnergy, s: np.ndarray) -> dict:
    """
    Analyze the Hessian at a critical point.
    
    Returns dict with:
        - eigenvalues: all (n-1) eigenvalues of tangent-space Hessian
        - lambda_shift: the Lagrange multiplier μ
        - min_eig: minimum eigenvalue
        - is_minimum: whether all eigenvalues > 0
        - energy_density: E(s) / n
    """
    H_proj, mu = energy.spherical_hessian(s)
    
    # Get stable (n-1) eigenvalues using tangent basis
    eigs_tangent = tangent_eigenvalues(H_proj, s)
    
    return {
        'eigenvalues': eigs_tangent,
        'lambda_shift': mu,
        'min_eig': np.min(eigs_tangent),
        'max_eig': np.max(eigs_tangent),
        'is_minimum': np.all(eigs_tangent > -1e-8),
        'energy_density': energy.energy(s) / energy.n,
        'grad_norm': np.linalg.norm(energy.spherical_grad(s))
    }


def run_experiment(n: int = 50, alpha: float = 4.0, num_instances: int = 20,
                   num_critical_per_instance: int = 10, beta: float = 1.0,
                   seed: int = 42, use_gpu: bool = True, use_lanczos: bool = True,
                   lanczos_k: int = 30):
    """
    Run the full experiment to test Hessian positivity.
    
    Parameters:
        n: number of variables
        alpha: clause density (m = alpha * n)
        num_instances: number of random SAT instances to generate
        num_critical_per_instance: critical points to find per instance
        beta: tanh chart parameter
        seed: random seed
        use_lanczos: if True, use fast Lanczos iteration (O(k) HVPs) instead of full Hessian (O(n²))
        lanczos_k: number of Lanczos iterations (20-50 usually sufficient)
    """
    np.random.seed(seed)
    
    all_results = []
    num_rejected_criticality = 0
    num_flagged_saturation = 0
    num_rejected_chart_degeneracy = 0
    
    print(f"Testing Hessian positivity for 3-SAT")
    print(f"  n = {n}, α = {alpha}, β = {beta}")
    print(f"  {num_instances} instances × {num_critical_per_instance} critical points each")
    print("-" * 60)
    
    for inst_idx in range(num_instances):
        instance = generate_random_3sat(n, alpha)
        
        if use_gpu and torch.cuda.is_available():
            energy_gpu = SATEnergyGPU(instance, beta=beta)
            
            for cp_idx in range(num_critical_per_instance):
                s, success = energy_gpu.find_critical_point_gpu()
                
                if not success:
                    continue
                
                # Tighter gate: verify criticality before expensive Hessian test
                s_check = s.clone().requires_grad_(True)
                E_check = energy_gpu.energy_torch(s_check)
                grad_check = torch.autograd.grad(E_check, s_check)[0]
                norm_sq_check = torch.dot(s, s)
                grad_sph_check = grad_check - s * (torch.dot(s, grad_check) / norm_sq_check)
                grad_norm_check = torch.norm(grad_sph_check).item()
                
                if grad_norm_check > 1e-3:
                    # Not critical enough for meaningful Hessian test
                    num_rejected_criticality += 1
                    continue
                
                # Compute chart-derivative diagnostic once
                sech2 = 1.0 / torch.cosh(energy_gpu.beta * s) ** 2
                sech2_mean = sech2.mean().item()

                # Detect fake stationarity from chart saturation:
                # If sech² is tiny AND spherical grad is tiny, likely chart-degeneracy artifact
                if sech2_mean < 1e-6 and grad_norm_check < 1e-7:
                    num_rejected_chart_degeneracy += 1
                    continue
                
                # Check saturation (store for analysis, don't drop)
                mean_sech2, is_saturated = sech2_mean, (sech2_mean < 1e-3)
                if is_saturated:
                    num_flagged_saturation += 1
                    # Don't continue - store saturated points for later analysis
                
                if use_lanczos:
                    # Fast path: Lanczos for extremal eigenvalues only
                    # Compute mu from already-computed grad_check to avoid duplicate work
                    mu_precomputed = (torch.dot(s, grad_check) / norm_sq_check).item()
                    
                    # Check symmetry on first CP per instance (diagnostic)
                    check_sym = (cp_idx == 0 and inst_idx == 0)
                    
                    # Lanczos-vs-full sanity check (once per run)
                    do_full_check = (inst_idx == 0 and cp_idx == 0)
                    
                    min_eig, max_eig, mu, grad_norm = fast_min_eigenvalue_gpu(
                        energy_gpu, s, k=lanczos_k, 
                        mu=mu_precomputed, grad_norm=grad_norm_check,
                        check_symmetry=check_sym
                    )
                    
                    # Lanczos-vs-full sanity check
                    if do_full_check:
                        H_proj_full, _, _ = energy_gpu.spherical_hessian_gpu(s)
                        eigs_full = torch.linalg.eigvalsh(H_proj_full)
                        # Drop the ~0 radial eigenvalue robustly (n×n projected matrix)
                        # Use a scale-aware epsilon based on the spectrum magnitude.
                        scale_full = torch.max(torch.abs(eigs_full)).item()
                        eps0 = max(1e-8, 1e-10 * max(1.0, scale_full))
                        eigs_tan = eigs_full[torch.abs(eigs_full) > eps0]
                        min_full = eigs_tan.min().item() if eigs_tan.numel() > 0 else float('nan')
                        max_full = eigs_tan.max().item() if eigs_tan.numel() > 0 else float('nan')
                        abs_diff_min = abs(min_eig - min_full)
                        rel_diff_min = abs_diff_min / (abs(min_full) + 1e-12)
                        print(f"  Lanczos-vs-full check:")
                        print(f"    min_lanczos={min_eig:.6f}, min_full={min_full:.6f}, rel_diff={rel_diff_min:.2e}")
                        print(f"    max_lanczos={max_eig:.6f}, max_full={max_full:.6f}")
                        if rel_diff_min > 0.1:
                            print(f"    WARNING: Large discrepancy - consider increasing lanczos_k")
                    
                    # Scale-adaptive margin: based on observed spectral scale, not mu
                    scale = max(abs(min_eig), abs(max_eig), 1.0)
                    margin = 1e-3 * scale
                    result = {
                        'eigenvalues': np.array([min_eig, max_eig]),  # only extremal
                        'lambda_shift': mu,
                        'min_eig': min_eig,
                        'max_eig': max_eig,
                        'is_minimum': min_eig > margin,  # scale-adaptive margin
                        'margin_used': margin,
                        'min_eig_raw': min_eig,  # record raw value
                        'energy_density': energy_gpu.energy_torch(s).item() / n,
                        'grad_norm': grad_norm,
                        'mean_sech2': mean_sech2,  # chart saturation metric
                        'is_saturated': is_saturated,  # flag for later analysis
                        'instance': inst_idx,
                        'cp_idx': cp_idx
                    }
                else:
                    # Full Hessian path (slower but gives all eigenvalues)
                    H_proj, mu, grad_norm = energy_gpu.spherical_hessian_gpu(s)
                    eigs = tangent_eigenvalues_gpu(H_proj, s)
                    eigs_np = eigs.cpu().numpy()
                    result = {
                        'eigenvalues': eigs_np,
                        'lambda_shift': mu,
                        'min_eig': np.min(eigs_np),
                        'max_eig': np.max(eigs_np),
                        'is_minimum': np.all(eigs_np > -1e-8),
                        'energy_density': energy_gpu.energy_torch(s).item() / n,
                        'grad_norm': grad_norm,
                        'instance': inst_idx,
                        'cp_idx': cp_idx
                    }
                all_results.append(result)
        else:
            energy = SATEnergy(instance, beta=beta)
            
            for cp_idx in range(num_critical_per_instance):
                s0 = np.random.randn(n)
                s, success = find_critical_point(energy, s0)
                
                if success:
                    result = analyze_hessian_at_critical_point(energy, s)
                    result['instance'] = inst_idx
                    result['cp_idx'] = cp_idx
                    all_results.append(result)
        
        if (inst_idx + 1) % 5 == 0:
            print(f"  Completed {inst_idx + 1}/{num_instances} instances...")
    
    print(f"\nFound {len(all_results)} critical points")
    if num_rejected_criticality > 0 or num_rejected_chart_degeneracy > 0 or num_flagged_saturation > 0:
        print(f"  Rejected for insufficient criticality: {num_rejected_criticality}")
        print(f"  Rejected for chart degeneracy (sech² tiny): {num_rejected_chart_degeneracy}")
        print(f"  Flagged as saturated (kept): {num_flagged_saturation}")
    
    if len(all_results) == 0:
        print("No critical points found!")
        return None
    
    # Analysis
    min_eigs = [r['min_eig'] for r in all_results]
    lambda_shifts = [r['lambda_shift'] for r in all_results]
    energy_densities = [r['energy_density'] for r in all_results]
    num_minima = sum(1 for r in all_results if r['is_minimum'])
    
    # Determine whether we ran full-spectrum eigenvalue analysis
    has_full_spectrum = len(all_results[0]['eigenvalues']) > 2
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total critical points found:     {len(all_results)}")
    if has_full_spectrum:
        print(f"Number that are local minima:    {num_minima} ({100*num_minima/len(all_results):.1f}%)")
    else:
        print(f"Estimated positive-definite:     {num_minima} ({100*num_minima/len(all_results):.1f}%)")
    print(f"")
    print(f"Minimum eigenvalue:")
    print(f"  Mean:   {np.mean(min_eigs):.6f}")
    print(f"  Std:    {np.std(min_eigs):.6f}")
    print(f"  Min:    {np.min(min_eigs):.6f}")
    print(f"  Max:    {np.max(min_eigs):.6f}")
    print(f"")
    print(f"Lagrange multiplier μ:")
    print(f"  Mean:   {np.mean(lambda_shifts):.6f}")
    print(f"  Std:    {np.std(lambda_shifts):.6f}")
    print(f"")
    print(f"Energy density E/n:")
    print(f"  Mean:   {np.mean(energy_densities):.6f}")
    print(f"  Std:    {np.std(energy_densities):.6f}")
    print(f"  Min:    {np.min(energy_densities):.6f}")
    print(f"  Max:    {np.max(energy_densities):.6f}")
    
    # GOE(+shift) analysis
    # IMPORTANT:
    #   Your tangent-space Riemannian Hessian is H_tan = P H P - μ P, so the identity shift is (-μ) on tangent.
    #   If we model the random part as a Wigner/GOE matrix with semicircle edge at ±2σ (no √n),
    #   then the appropriate comparison is:  (-μ) > 2σ.
    #
    # We therefore estimate σ from *centered* eigenvalues per critical point:
    #   eigs_centered = eigs - mean(eigs)
    #   σ_hat_i = std(eigs_centered)
    # Under the semicircle law, edge ≈ 2σ, and std ≈ σ (for the canonical scaling).
    #
    # This avoids mixing in an incorrect √n normalization when the matrix scaling is unknown.
    if has_full_spectrum:
        sigma_hats = []
        edge_hats = []
        edge_over_2sigma = []
        mean_shifts = []

        for r in all_results:
            eigs = np.asarray(r['eigenvalues'], dtype=float)
            # Center per-CP spectrum to isolate the "random" part shape
            ec = eigs - np.mean(eigs)
            sig = float(np.std(ec))
            edg = float(np.max(np.abs(ec)))
            sigma_hats.append(sig)
            edge_hats.append(edg)
            if sig > 0:
                edge_over_2sigma.append(edg / (2.0 * sig))
            mean_shifts.append(-float(r['lambda_shift']))  # shift magnitude on tangent

        sigma_hats = np.array(sigma_hats, dtype=float)
        edge_hats = np.array(edge_hats, dtype=float)
        edge_over_2sigma = np.array(edge_over_2sigma, dtype=float) if len(edge_over_2sigma) else np.array([], dtype=float)
        mean_shifts = np.array(mean_shifts, dtype=float)

        estimated_sigma = float(np.mean(sigma_hats))
        estimated_edge = 2.0 * estimated_sigma

        print(f"\nGOE(+shift) Analysis (full spectrum; semicircle-calibrated):")
        print(f"  σ_hat (mean std of centered eigs):     {estimated_sigma:.6f}")
        print(f"  2σ_hat (predicted semicircle edge):    {estimated_edge:.6f}")
        print(f"  empirical edge (mean max|centered|):   {float(np.mean(edge_hats)):.6f}")
        if edge_over_2sigma.size:
            print(f"  edge/(2σ) ratio: mean={float(np.mean(edge_over_2sigma)):.3f}, std={float(np.std(edge_over_2sigma)):.3f}")
        print(f"  Mean μ:                                 {float(np.mean(lambda_shifts)):.6f}")
        print(f"  Mean shift (-μ):                        {float(np.mean(mean_shifts)):.6f}")
        print(f"  (-μ) > 2σ ?                             {float(np.mean(mean_shifts)) > estimated_edge}")
    else:
        # Lanczos mode: GOE inference not valid, just report extremal stats
        max_eigs = [r['max_eig'] for r in all_results]
        estimated_edge = np.nan  # not computable
        estimated_sigma = np.nan
        
        print(f"\nExtremal Eigenvalue Statistics (Lanczos mode):")
        print(f"  Mean min eigenvalue: {np.mean(min_eigs):.6f}")
        print(f"  Mean max eigenvalue: {np.mean(max_eigs):.6f}")
        print(f"  Mean μ:              {np.mean(lambda_shifts):.6f}")
        print(f"  (GOE σ/edge inference disabled - not valid for extremal-only data)")
    
    return {
        'results': all_results,
        'min_eigs': min_eigs,
        'lambda_shifts': lambda_shifts,
        'energy_densities': energy_densities,
        'num_minima': num_minima,
        'estimated_sigma': estimated_sigma,
        'estimated_edge': estimated_edge,
        'n': n,
        'alpha': alpha,
        'beta': beta
    }


def plot_results(data: dict, save_path: str = None):
    """Generate plots of the results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Histogram of minimum eigenvalues
    ax = axes[0, 0]
    ax.hist(data['min_eigs'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='λ = 0')
    ax.set_xlabel('Minimum Eigenvalue')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Minimum Eigenvalues at Critical Points')
    ax.legend()
    
    # Plot 2: Energy density vs minimum eigenvalue
    ax = axes[0, 1]
    ax.scatter(data['energy_densities'], data['min_eigs'], alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Energy Density E/n')
    ax.set_ylabel('Minimum Eigenvalue')
    ax.set_title('Minimum Eigenvalue vs Energy Density')
    
    # Plot 3: Histogram of Lagrange multipliers
    ax = axes[1, 0]
    ax.hist(data['lambda_shifts'], bins=30, edgecolor='black', alpha=0.7)
    if not np.isnan(data['estimated_edge']):
        ax.axvline(x=data['estimated_edge'], color='green', linestyle='--', 
                   linewidth=2, label=f'2σ√n = {data["estimated_edge"]:.3f}')
        ax.legend()
    ax.set_xlabel('Lagrange Multiplier μ')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Lagrange Multiplier μ')
    
    # Plot 4: Eigenvalue spectrum (different for Lanczos vs full)
    ax = axes[1, 1]
    has_full_spectrum = len(data['results'][0]['eigenvalues']) > 2
    
    if has_full_spectrum:
        # Full spectrum mode: plot eigenvalue curves
        sample_results = data['results'][:min(5, len(data['results']))]
        for i, r in enumerate(sample_results):
            eigs = np.sort(r['eigenvalues'])
            ax.plot(eigs, 'o-', alpha=0.6, label=f'CP {i+1}')
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Eigenvalue Spectrum (Sample Critical Points)')
    else:
        # Lanczos mode: scatter plot of (min, max) pairs
        min_eigs = [r['min_eig'] for r in data['results']]
        max_eigs = [r['max_eig'] for r in data['results']]
        ax.scatter(range(len(min_eigs)), min_eigs, alpha=0.6, label='min eig', marker='v')
        ax.scatter(range(len(max_eigs)), max_eigs, alpha=0.6, label='max eig', marker='^')
        ax.set_xlabel('Critical Point Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Extremal Eigenvalues (Lanczos)')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.legend()
    
    plt.suptitle(f'Hessian Positivity Test: n={data["n"]}, α={data["alpha"]}, β={data["beta"]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    # Run with moderate size for testing
    # Increase n for more accurate results (but slower)
    
    print("=" * 60)
    print("HESSIAN POSITIVITY TEST FOR KAC-RICE LEMMA C")
    print("=" * 60)
    print()
    
    # Test parameters
    # α_d ≈ 3.86, α_c ≈ 4.27 for 3-SAT
    # We use α = 4.0 (clustered satisfiable regime)
    
    use_gpu = torch.cuda.is_available()
    print(f"GPU available: {use_gpu}")
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Use full Hessian (fast with torch.func vectorization)
    use_lanczos = False
    print(f"Using Lanczos: {use_lanczos} (full Hessian is faster with torch.func)")
    
    data = run_experiment(
        n=100,                      # variables (GPU can handle more)
        alpha=4.0,                  # clause density
        num_instances=10,           # SAT instances  
        num_critical_per_instance=10,  # critical points per instance
        beta=1.0,                   # tanh parameter
        seed=42,
        use_gpu=use_gpu,
        use_lanczos=use_lanczos,
        lanczos_k=40                # Lanczos iterations (30-50 typical)
    )
    
    if data:
        # Generate plots
        plot_results(data, save_path='hessian_positivity_results.png')
        
        print("\n" + "=" * 60)
        print("CONCLUSION")
        print("=" * 60)
        
        frac_minima = data['num_minima'] / len(data['results'])
        
        # Check if Lanczos mode (heuristic check)
        is_lanczos = len(data['results'][0]['eigenvalues']) <= 2
        
        if frac_minima > 0.5:
            if is_lanczos:
                print(f"✓ {100*frac_minima:.1f}% estimated positive-definite (min_eig > margin)")
            else:
                print(f"✓ {100*frac_minima:.1f}% of critical points are local minima")
            print("  This supports Lemma C (Hessian positivity).")
        else:
            if is_lanczos:
                print(f"✗ Only {100*frac_minima:.1f}% estimated positive-definite (min_eig > margin)")
            else:
                print(f"✗ Only {100*frac_minima:.1f}% of critical points are local minima")
            print("  Lemma C may need refinement (e.g., restrict energy band).")
        
        if not np.isnan(data['estimated_edge']):
            if np.mean(data['lambda_shifts']) > data['estimated_edge']:
                print(f"✓ Mean μ ({np.mean(data['lambda_shifts']):.4f}) > 2σ√n ({data['estimated_edge']:.4f})")
                print("  The GOE + shift model appears valid.")
            else:
                print(f"✗ Mean μ ({np.mean(data['lambda_shifts']):.4f}) < 2σ√n ({data['estimated_edge']:.4f})")
                print("  The shift may not exceed the spectral edge.")
        else:
            print(f"  Mean μ = {np.mean(data['lambda_shifts']):.4f}")
            print("  (Run with use_lanczos=False for GOE edge comparison)")
