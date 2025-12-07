"""
Gauge configuration I/O and generation.

A gauge configuration is the fundamental object in lattice QCD:
- 4D hypercubic lattice of size L^4
- SU(3) link variables U_μ(x) on each directed edge
- Total: 4 × L^4 matrices

Storage: U[μ][t][x][y][z] as complex 3×3 matrices
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from numpy.typing import NDArray
import h5py

from .su3 import random_su3, random_su3_near_identity, project_config_to_su3, is_su3, project_to_su3


@dataclass
class GaugeConfig:
    """
    Container for a lattice gauge configuration.
    
    Attributes:
        U: Link variables, shape (4, L, L, L, L, 3, 3) complex128
        L: Lattice size (assumed L^4 hypercubic)
        beta: Gauge coupling (for Wilson action: β = 6/g²)
        metadata: Additional info (trajectory number, action value, etc.)
    """
    U: NDArray[np.complex128]
    L: int
    beta: float = 6.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration shape."""
        expected_shape = (4, self.L, self.L, self.L, self.L, 3, 3)
        if self.U.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {self.U.shape}")
    
    @property
    def n_links(self) -> int:
        """Total number of link variables."""
        return 4 * self.L ** 4
    
    @property
    def memory_mb(self) -> float:
        """Memory usage in MB."""
        return self.U.nbytes / 1024 / 1024
    
    def validate(self, tol: float = 1e-10) -> bool:
        """Check that all links are valid SU(3) matrices."""
        for mu in range(4):
            for t in range(self.L):
                for x in range(self.L):
                    for y in range(self.L):
                        for z in range(self.L):
                            if not is_su3(self.U[mu, t, x, y, z], tol):
                                return False
        return True
    
    def project(self) -> "GaugeConfig":
        """Return a new config with all links projected to exact SU(3)."""
        return GaugeConfig(
            U=project_config_to_su3(self.U),
            L=self.L,
            beta=self.beta,
            metadata=self.metadata.copy()
        )


def cold_start(L: int, beta: float = 6.0) -> GaugeConfig:
    """
    Create a "cold" configuration with all links set to identity.
    
    This is the classical vacuum (zero field strength everywhere).
    
    Args:
        L: Lattice size
        beta: Gauge coupling
    
    Returns:
        Configuration with U_μ(x) = I for all (x, μ)
    """
    U = np.zeros((4, L, L, L, L, 3, 3), dtype=np.complex128)
    for mu in range(4):
        for t in range(L):
            for x in range(L):
                for y in range(L):
                    for z in range(L):
                        U[mu, t, x, y, z] = np.eye(3, dtype=np.complex128)
    
    return GaugeConfig(U=U, L=L, beta=beta, metadata={"start": "cold"})


def hot_start(L: int, beta: float = 6.0) -> GaugeConfig:
    """
    Create a "hot" configuration with random SU(3) links.
    
    This is a completely disordered configuration.
    
    Args:
        L: Lattice size
        beta: Gauge coupling
    
    Returns:
        Configuration with U_μ(x) = random SU(3) for all (x, μ)
    """
    U = np.zeros((4, L, L, L, L, 3, 3), dtype=np.complex128)
    for mu in range(4):
        for t in range(L):
            for x in range(L):
                for y in range(L):
                    for z in range(L):
                        U[mu, t, x, y, z] = random_su3()
    
    return GaugeConfig(U=U, L=L, beta=beta, metadata={"start": "hot"})


# ============================================================================
# SU(3) Heatbath Algorithm (Cabibbo-Marinari)
# ============================================================================

def _su2_heatbath_sample(a: float, beta: float) -> np.ndarray:
    """
    Sample SU(2) matrix from heatbath distribution.
    
    Samples from distribution ∝ exp(β/2 * a * Tr(X)) where X is SU(2).
    For SU(2), Tr(X) = 2*x0 where X = x0*I + i*x·σ.
    So we sample x0 from ∝ sqrt(1-x0²) * exp(β*a*x0).
    
    Uses Kennedy-Pendleton algorithm for efficient sampling.
    
    Args:
        a: Scale factor from staple (sqrt of |det(W)|)
        beta: Gauge coupling
    
    Returns:
        2x2 SU(2) matrix
    
    Reference: Kennedy, Pendleton, Phys. Lett. B 156 (1985) 393
    """
    if a < 1e-10:
        # Zero staple - return random SU(2)
        return _random_su2()
    
    # Effective coupling k = β * a
    k = beta * a
    
    # Kennedy-Pendleton algorithm for SU(2) heatbath
    # Sample x0 from P(x0) ∝ sqrt(1 - x0²) * exp(k * x0) on [-1, 1]
    max_iterations = 1000
    for _ in range(max_iterations):
        # Generate uniform random numbers
        r1 = np.random.random()
        r2 = np.random.random()
        r3 = np.random.random()
        r4 = np.random.random()
        
        # Avoid log(0)
        if r1 < 1e-100:
            r1 = 1e-100
        if r2 < 1e-100:
            r2 = 1e-100
        
        # Kennedy-Pendleton algorithm
        x = -np.log(r1) / k
        y = -np.log(r2) / k
        c = np.cos(2 * np.pi * r3) ** 2
        
        delta = x * c + y
        
        # Acceptance test: accept with probability sqrt(1 - delta/2)
        if delta < 2.0 and r4 ** 2 <= 1.0 - 0.5 * delta:
            x0 = 1.0 - delta
            break
    else:
        # Fallback: just use x0 = 1 (identity)
        x0 = 1.0
    
    # Clamp x0 to valid range
    if x0 > 1.0:
        x0 = 1.0
    if x0 < -1.0:
        x0 = -1.0
    
    # x0 is the real part; sample (x1, x2, x3) uniformly on sphere of radius r
    r = np.sqrt(max(0.0, 1.0 - x0 * x0))
    
    # Uniform point on 2-sphere for (x1, x2, x3)
    phi = 2 * np.pi * np.random.random()
    cos_theta = 2 * np.random.random() - 1
    sin_theta = np.sqrt(max(0.0, 1 - cos_theta ** 2))
    
    x1 = r * sin_theta * np.cos(phi)
    x2 = r * sin_theta * np.sin(phi)
    x3 = r * cos_theta
    
    # Construct SU(2) matrix: X = x0*I + i*(x1*σ1 + x2*σ2 + x3*σ3)
    # σ1 = [[0,1],[1,0]], σ2 = [[0,-i],[i,0]], σ3 = [[1,0],[0,-1]]
    # X = [[x0+i*x3, i*x1+x2], [i*x1-x2, x0-i*x3]]
    return np.array([
        [x0 + 1j * x3, x2 + 1j * x1],
        [-x2 + 1j * x1, x0 - 1j * x3]
    ], dtype=np.complex128)


def _random_su2() -> np.ndarray:
    """Generate a random SU(2) matrix (Haar measure)."""
    # Use QR on random complex 2x2
    Z = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    Q, R = np.linalg.qr(Z)
    # Fix det to +1
    det = np.linalg.det(Q)
    Q = Q / np.sqrt(det)
    return Q


def _embed_su2_in_su3(su2: np.ndarray, subgroup: int) -> np.ndarray:
    """
    Embed SU(2) matrix into SU(3).
    
    Args:
        su2: 2x2 SU(2) matrix
        subgroup: 0, 1, or 2 for the three SU(2) subgroups
            0: (12) - rows/cols 0,1
            1: (13) - rows/cols 0,2
            2: (23) - rows/cols 1,2
    
    Returns:
        3x3 SU(3) matrix
    """
    result = np.eye(3, dtype=np.complex128)
    
    if subgroup == 0:
        result[0, 0] = su2[0, 0]
        result[0, 1] = su2[0, 1]
        result[1, 0] = su2[1, 0]
        result[1, 1] = su2[1, 1]
    elif subgroup == 1:
        result[0, 0] = su2[0, 0]
        result[0, 2] = su2[0, 1]
        result[2, 0] = su2[1, 0]
        result[2, 2] = su2[1, 1]
    else:  # subgroup == 2
        result[1, 1] = su2[0, 0]
        result[1, 2] = su2[0, 1]
        result[2, 1] = su2[1, 0]
        result[2, 2] = su2[1, 1]
    
    return result


def _extract_su2_from_su3(m: np.ndarray, subgroup: int) -> np.ndarray:
    """Extract 2x2 submatrix from 3x3 for given subgroup."""
    if subgroup == 0:
        return m[np.ix_([0, 1], [0, 1])].copy()
    elif subgroup == 1:
        return m[np.ix_([0, 2], [0, 2])].copy()
    else:
        return m[np.ix_([1, 2], [1, 2])].copy()


def _heatbath_su3_update(U_link: np.ndarray, staple: np.ndarray, beta: float) -> np.ndarray:
    """
    Perform one SU(3) heatbath update using Cabibbo-Marinari algorithm.
    
    For Wilson action S = (β/3) Σ Re Tr(1 - U·staple^†), we want to sample U
    from P(U) ∝ exp((β/3) Re Tr(U·staple^†)) = exp((β/3) Re Tr(U·V))
    where V = staple^†.
    
    The Cabibbo-Marinari method updates U by sequential SU(2) subgroup updates.
    
    Args:
        U_link: Current SU(3) link variable
        staple: Staple sum for this link
        beta: Gauge coupling
    
    Returns:
        Updated SU(3) link variable
    """
    U_new = U_link.copy()
    
    # V = staple^† is what U couples to in the action
    V = staple.conj().T
    
    # Sweep over three SU(2) subgroups
    for subgroup in range(3):
        # W = U * V is the matrix whose trace we maximize
        W = np.dot(U_new, V)
        
        # Extract 2x2 subblock of W
        w_sub = _extract_su2_from_su3(W, subgroup)
        
        # Compute determinant and a = sqrt(|det|)
        det_w = w_sub[0, 0] * w_sub[1, 1] - w_sub[0, 1] * w_sub[1, 0]
        a = np.sqrt(np.abs(det_w))
        
        if a > 1e-10:
            # Normalize w_sub to get element of SU(2): v = w_sub / a
            # The normalized v satisfies det(v) = det_w / a² = det_w / |det_w| = phase
            # We want v ∈ SU(2), so we need to fix the phase
            v = w_sub / a
            
            # Sample X from P(X) ∝ exp((β/3) * a * Re Tr(X))
            # For SU(2), Tr(X) = 2*x0, so this is exp((2β/3) * a * x0)
            # The Kennedy-Pendleton uses k = (2β/3) * a
            x = _su2_heatbath_sample(a, 2.0 * beta / 3.0)
            
            # The update: new U_sub = X * v^† * old U_sub
            # where U_sub is the SU(2) subblock of U_new
            v_dag = v.conj().T
            update_su2 = np.dot(x, v_dag)
            
            # Embed in SU(3) and apply
            update_su3 = _embed_su2_in_su3(update_su2, subgroup)
            U_new = np.dot(update_su3, U_new)
    
    # Project to exact SU(3) to avoid numerical drift
    return project_to_su3(U_new)


def heatbath_sweep(config: GaugeConfig) -> GaugeConfig:
    """
    Perform one full heatbath sweep over all links.
    
    Args:
        config: Input gauge configuration
    
    Returns:
        Updated configuration after one sweep
    """
    from .topological import _compute_staple
    
    U = config.U.copy()
    L = config.L
    
    for mu in range(4):
        for t in range(L):
            for x in range(L):
                for y in range(L):
                    for z in range(L):
                        staple = _compute_staple(U, t, x, y, z, mu)
                        U[mu, t, x, y, z] = _heatbath_su3_update(
                            U[mu, t, x, y, z], staple, config.beta
                        )
    
    return GaugeConfig(U=U, L=config.L, beta=config.beta, metadata=config.metadata.copy())


def generate_heatbath_configs(
    L: int,
    beta: float,
    n_configs: int,
    thermalization: int = 500,
    separation: int = 10,
    start: str = "hot",
    verbose: bool = True,
) -> list[GaugeConfig]:
    """
    Generate gauge configurations using heatbath algorithm.
    
    The heatbath directly samples from the conditional distribution,
    thermalizing faster than Metropolis/HMC for pure gauge.
    
    Args:
        L: Lattice size (L^4)
        beta: Gauge coupling
        n_configs: Number of configurations to generate
        thermalization: Number of sweeps before first saved config
        separation: Number of sweeps between saved configs
        start: "hot" or "cold" initial configuration
        verbose: Print progress
    
    Returns:
        List of thermalized, decorrelated gauge configurations
    """
    from .wilson_loops import average_plaquette
    
    # Initialize
    if start == "cold":
        config = cold_start(L, beta)
    else:
        config = hot_start(L, beta)
    
    if verbose:
        plaq0 = average_plaquette(config.U)
        print(f"Heatbath: L={L}, β={beta}, start={start}")
        print(f"Initial plaquette: {plaq0:.6f}")
        print(f"Thermalizing ({thermalization} sweeps)...")
    
    # Thermalization
    for i in range(thermalization):
        config = heatbath_sweep(config)
        if verbose and (i + 1) % 50 == 0:
            plaq = average_plaquette(config.U)
            print(f"  Sweep {i+1}/{thermalization}: <P> = {plaq:.6f}")
    
    if verbose:
        plaq_therm = average_plaquette(config.U)
        print(f"After thermalization: <P> = {plaq_therm:.6f}")
        print(f"Generating {n_configs} configurations (separation={separation})...")
    
    # Generate configurations
    configs = []
    for i in range(n_configs):
        # Separation sweeps
        for _ in range(separation):
            config = heatbath_sweep(config)
        
        # Save a copy
        saved_config = GaugeConfig(
            U=config.U.copy(),
            L=L,
            beta=beta,
            metadata={
                "trajectory": i,
                "plaquette": average_plaquette(config.U),
                "thermalization": thermalization,
                "separation": separation,
            }
        )
        configs.append(saved_config)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{n_configs} configs, <P> = {saved_config.metadata['plaquette']:.6f}")
    
    if verbose:
        plaq_mean = np.mean([c.metadata['plaquette'] for c in configs])
        plaq_std = np.std([c.metadata['plaquette'] for c in configs])
        print(f"Done! Mean plaquette: {plaq_mean:.6f} ± {plaq_std:.6f}")
    
    return configs


def save_config(config: GaugeConfig, path: Path | str) -> None:
    """
    Save a gauge configuration to HDF5 file.
    
    File structure:
        /gauge_field: (4, L, L, L, L, 3, 3) complex128
        /metadata/lattice_size: int
        /metadata/beta: float
        /metadata/*: additional metadata
    
    Args:
        config: Configuration to save
        path: Output file path (should end in .h5)
    """
    path = Path(path)
    
    with h5py.File(path, "w") as f:
        # Main data
        f.create_dataset("gauge_field", data=config.U, compression="gzip")
        
        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["lattice_size"] = config.L
        meta.attrs["beta"] = config.beta
        
        for key, value in config.metadata.items():
            try:
                meta.attrs[key] = value
            except TypeError:
                # Skip non-serializable metadata
                pass


def load_config(path: Path | str) -> GaugeConfig:
    """
    Load a gauge configuration from HDF5 file.
    
    Args:
        path: Input file path
    
    Returns:
        Loaded configuration
    """
    path = Path(path)
    
    with h5py.File(path, "r") as f:
        U = f["gauge_field"][:]
        L = f["metadata"].attrs["lattice_size"]
        beta = f["metadata"].attrs["beta"]
        
        metadata = {}
        for key, value in f["metadata"].attrs.items():
            if key not in ("lattice_size", "beta"):
                metadata[key] = value
    
    return GaugeConfig(U=U, L=L, beta=beta, metadata=metadata)


# ============================================================================
# Configuration Generation (HMC)
# ============================================================================

def wilson_action(config: GaugeConfig) -> float:
    """
    Compute the Wilson gauge action.
    
    S_W = β Σ_{x,μ<ν} (1 - (1/3) Re Tr P_μν(x))
    
    where P_μν is the plaquette (see wilson_loops.py).
    
    Args:
        config: Gauge configuration
    
    Returns:
        Total Wilson action value
    
    TODO: Implement (use compute_all_plaquettes from wilson_loops.py)
    """
    # COPILOT: Implement Wilson action
    # Import compute_all_plaquettes, sum over (1 - ReTr/3)
    from .wilson_loops import average_plaquette
    
    plaq = average_plaquette(config.U)
    n_plaquettes = 6 * config.L ** 4  # 6 planes × L^4 sites
    
    return config.beta * n_plaquettes * (1 - plaq / 3)


def generate_config_hmc(
    L: int,
    beta: float,
    n_trajectories: int,
    trajectory_length: float = 1.0,
    n_steps: int = 20,
    start: str = "hot",
    thermalization: int = 100,
) -> list[GaugeConfig]:
    """
    Generate gauge configurations using Hybrid Monte Carlo (HMC).
    
    NOTE: For pure gauge SU(3), we now use the heatbath algorithm instead,
    which thermalizes faster by directly sampling the conditional distribution.
    
    Args:
        L: Lattice size
        beta: Gauge coupling
        n_trajectories: Number of configurations to generate
        trajectory_length: τ in leapfrog (ignored, using heatbath)
        n_steps: Number of leapfrog steps per trajectory (ignored)
        start: "cold" or "hot" (default "hot")
        thermalization: Number of initial sweeps to discard
    
    Returns:
        List of thermalized gauge configurations
    """
    # Use proper heatbath algorithm for pure gauge
    return generate_heatbath_configs(
        L=L,
        beta=beta,
        n_configs=n_trajectories,
        thermalization=thermalization,
        separation=10,  # 10 sweeps between configs for decorrelation
        start=start,
        verbose=True,
    )


def hmc_leapfrog_step(
    U: NDArray[np.complex128],
    P: NDArray[np.complex128],
    dt: float,
    beta: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Single leapfrog integration step for HMC.
    
    Leapfrog integrator (symplectic):
        P(dt/2) = P(0) - (dt/2) ∂S/∂U
        U(dt) = exp(dt P(dt/2)) U(0)
        P(dt) = P(dt/2) - (dt/2) ∂S/∂U
    
    Args:
        U: Link variables, shape (4, L, L, L, L, 3, 3)
        P: Conjugate momenta (su(3) algebra), same shape
        dt: Time step size
        beta: Gauge coupling
    
    Returns:
        Updated (U, P) after one leapfrog step
    
    TODO: Implement
    """
    # COPILOT: Implement leapfrog integrator
    # Need: compute_force (derivative of Wilson action)
    # Need: su3_exp for U update
    return U, P


def compute_hmc_force(U: NDArray[np.complex128], beta: float) -> NDArray[np.complex128]:
    """
    Compute the HMC force F = -∂S/∂U.
    
    For Wilson action:
        F_μ(x) = (β/3) Σ_{ν≠μ} [ staple_μν(x) - staple_μν(x)† ]_{TA}
    
    where staple is the sum of plaquettes sharing the link U_μ(x),
    and [...]_{TA} denotes the traceless anti-Hermitian part.
    
    Args:
        U: Link variables
        beta: Gauge coupling
    
    Returns:
        Force in su(3) algebra (traceless anti-Hermitian matrices)
    
    TODO: Implement
    """
    # COPILOT: Implement HMC force
    # This requires computing "staples" around each link
    return np.zeros_like(U)
