"""
Wilson loop computation for lattice gauge theory.

A Wilson loop W(C) is the trace of the path-ordered product of link variables
around a closed loop C:

    W(C) = Tr[ P ∏_{(x,μ)∈C} U_μ(x) ]

The simplest Wilson loop is the "plaquette" - a 1×1 loop in a single plane.

Wilson loops are:
- Gauge invariant (key property!)
- Related to flux of field strength through the loop
- The building blocks of the Davis-Wilson map
"""

from __future__ import annotations

from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from numba import jit, prange

from .gauge_config import GaugeConfig


# Type aliases
Site = Tuple[int, int, int, int]  # (t, x, y, z)
Direction = int  # 0-3 for forward, -1 to -4 for backward


@dataclass
class Loop:
    """
    Specification of a Wilson loop path.
    
    Attributes:
        start: Starting site (t, x, y, z)
        steps: List of directions to traverse
               Positive: 0=t, 1=x, 2=y, 3=z (forward)
               Negative: -1=-t, -2=-x, -3=-y, -4=-z (backward, uses U†)
    """
    start: Site
    steps: List[Direction]
    
    def is_closed(self, L: int) -> bool:
        """Check if path returns to starting point (with periodic BC)."""
        pos = list(self.start)
        for d in self.steps:
            if d >= 0:
                pos[d] = (pos[d] + 1) % L
            else:
                pos[-d - 1] = (pos[-d - 1] - 1) % L
        return tuple(pos) == self.start


@jit(nopython=True, cache=True)
def _shift_site(site: Tuple[int, int, int, int], mu: int, L: int) -> Tuple[int, int, int, int]:
    """Shift site by one step in direction mu with periodic BC."""
    t, x, y, z = site
    if mu == 0:
        return ((t + 1) % L, x, y, z)
    elif mu == 1:
        return (t, (x + 1) % L, y, z)
    elif mu == 2:
        return (t, x, (y + 1) % L, z)
    elif mu == 3:
        return (t, x, y, (z + 1) % L)
    else:
        raise ValueError(f"Invalid direction {mu}")


@jit(nopython=True, cache=True)
def _shift_site_back(site: Tuple[int, int, int, int], mu: int, L: int) -> Tuple[int, int, int, int]:
    """Shift site by one step backward in direction mu with periodic BC."""
    t, x, y, z = site
    if mu == 0:
        return ((t - 1) % L, x, y, z)
    elif mu == 1:
        return (t, (x - 1) % L, y, z)
    elif mu == 2:
        return (t, x, (y - 1) % L, z)
    elif mu == 3:
        return (t, x, y, (z - 1) % L)
    else:
        raise ValueError(f"Invalid direction {mu}")


@jit(nopython=True, cache=True)
def compute_plaquette(
    U: NDArray[np.complex128],
    t: int, x: int, y: int, z: int,
    mu: int, nu: int
) -> NDArray[np.complex128]:
    """
    Compute the plaquette P_μν(x) in the μν plane at site x.
    
    P_μν(x) = U_μ(x) · U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)
    
    This is a 1×1 Wilson loop, the elementary gauge-invariant observable.
    
    Args:
        U: Gauge configuration, shape (4, L, L, L, L, 3, 3)
        t, x, y, z: Site coordinates
        mu, nu: Plane directions (0 ≤ mu < nu ≤ 3)
    
    Returns:
        3×3 complex matrix (the plaquette, should be in SU(3))
    
    Diagram:
           U_ν(x+μ̂)
        x+μ̂ ----→ x+μ̂+ν̂
          ↑         ↓
    U_μ(x)|         |U_μ†(x+ν̂)
          |         |
          x ←---- x+ν̂
           U_ν†(x)
    """
    L = int(U.shape[1])
    
    # Initialize all shifted coordinates (ensures consistent types)
    t_mu = t
    x_mu = x
    y_mu = y
    z_mu = z
    t_nu = t
    x_nu = x
    y_nu = y
    z_nu = z
    
    # Apply shift in μ direction
    if mu == 0:
        t_mu = (t + 1) % L
    elif mu == 1:
        x_mu = (x + 1) % L
    elif mu == 2:
        y_mu = (y + 1) % L
    else:  # mu == 3
        z_mu = (z + 1) % L
    
    # Apply shift in ν direction
    if nu == 0:
        t_nu = (t + 1) % L
    elif nu == 1:
        x_nu = (x + 1) % L
    elif nu == 2:
        y_nu = (y + 1) % L
    else:  # nu == 3
        z_nu = (z + 1) % L
    
    # Get the four link matrices
    U_mu_x = U[mu, t, x, y, z]  # U_μ(x)
    U_nu_xmu = U[nu, t_mu, x_mu, y_mu, z_mu]  # U_ν(x+μ̂)
    U_mu_xnu = U[mu, t_nu, x_nu, y_nu, z_nu]  # U_μ(x+ν̂)
    U_nu_x = U[nu, t, x, y, z]  # U_ν(x)
    
    # Compute plaquette: U_μ(x) · U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)
    temp1 = np.dot(U_mu_x, U_nu_xmu)
    temp2 = np.dot(U_mu_xnu.conj().T, U_nu_x.conj().T)
    plaq = np.dot(temp1, temp2)
    
    return plaq


@jit(nopython=True, cache=True)
def plaquette_trace(
    U: NDArray[np.complex128],
    t: int, x: int, y: int, z: int,
    mu: int, nu: int
) -> complex:
    """
    Compute the trace of a plaquette.
    
    Returns:
        Tr(P_μν(x)) - complex number with Re ∈ [-3, 3] for SU(3)
    """
    P = compute_plaquette(U, t, x, y, z, mu, nu)
    return P[0, 0] + P[1, 1] + P[2, 2]


@jit(nopython=True, cache=True)
def compute_all_plaquettes(U: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Compute all plaquettes on the lattice.
    
    Args:
        U: Gauge configuration, shape (4, L, L, L, L, 3, 3)
    
    Returns:
        Array of plaquettes, shape (6, L, L, L, L, 3, 3)
        Index 0-5 corresponds to planes: 01, 02, 03, 12, 13, 23
    
    This is parallelized over lattice sites.
    """
    L = int(U.shape[1])
    plaquettes = np.zeros((6, L, L, L, L, 3, 3), dtype=np.complex128)
    
    # Plane index mapping
    planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for plane_idx in range(6):
        mu, nu = planes[plane_idx]
        for t in range(L):  # Changed from prange to range
            for x in range(L):
                for y in range(L):
                    for z in range(L):
                        plaquettes[plane_idx, t, x, y, z] = compute_plaquette(
                            U, t, x, y, z, mu, nu
                        )
    
    return plaquettes


@jit(nopython=True, cache=True)
def average_plaquette(config_U: NDArray[np.complex128]) -> float:
    """
    Compute the average plaquette value.
    
    <P> = (1 / 6L⁴) Σ_{x,μ<ν} Re[Tr(P_μν(x))]
    
    This is the most basic observable in lattice QCD.
    For a cold (identity) config, <P> = 3.
    For a hot (random) config, <P> ≈ 0.
    
    Args:
        config_U: Gauge field array, shape (4, L, L, L, L, 3, 3)
    
    Returns:
        Average plaquette value (should be in [0, 3] for thermalized configs)
    """
    L = int(config_U.shape[1])
    total = 0.0
    count = 0
    
    planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for plane_idx in range(6):
        mu, nu = planes[plane_idx]
        for t in range(L):  # Changed from prange
            for x in range(L):
                for y in range(L):
                    for z in range(L):
                        tr = plaquette_trace(config_U, t, x, y, z, mu, nu)
                        total += tr.real
                        count += 1
    
    return total / count


def compute_wilson_loop(
    config: GaugeConfig,
    loop: Loop
) -> complex:
    """
    Compute a general Wilson loop W(C) = Tr[∏ U].
    
    Args:
        config: Gauge configuration
        loop: Loop specification (start site and path)
    
    Returns:
        Complex trace of the path-ordered product
    
    TODO: Implement general Wilson loop
    """
    # COPILOT: Implement general Wilson loop computation
    # Start with identity, multiply links along path
    # For backward direction d<0, use U†(x-d̂) 
    
    U = config.U
    L = config.L
    
    result = np.eye(3, dtype=np.complex128)
    pos = list(loop.start)
    
    for d in loop.steps:
        if d >= 0:
            # Forward link U_μ(x)
            mu = d
            link = U[mu, pos[0], pos[1], pos[2], pos[3]]
            result = np.dot(result, link)
            # Move forward
            pos[mu] = (pos[mu] + 1) % L
        else:
            # Backward link U_μ†(x-μ̂)
            mu = -d - 1
            # First move backward
            pos[mu] = (pos[mu] - 1) % L
            # Then get the daggered link
            link = U[mu, pos[0], pos[1], pos[2], pos[3]].conj().T
            result = np.dot(result, link)
    
    return np.trace(result)


def compute_rectangular_wilson_loop(
    config: GaugeConfig,
    site: Site,
    mu: int,
    nu: int,
    m: int,
    n: int
) -> complex:
    """
    Compute an m×n rectangular Wilson loop in the μν plane.
    
    Args:
        config: Gauge configuration
        site: Starting corner
        mu, nu: Plane directions
        m: Size in μ direction
        n: Size in ν direction
    
    Returns:
        Tr(W_{m×n})
    
    For m=n=1, this is the plaquette.
    Larger loops probe longer-distance correlations.
    
    TODO: Implement
    """
    # COPILOT: Implement rectangular Wilson loop
    # Build the path: m steps in μ, n steps in ν, m steps in -μ, n steps in -ν
    
    steps = []
    steps.extend([mu] * m)      # m steps forward in μ
    steps.extend([nu] * n)      # n steps forward in ν  
    steps.extend([-mu - 1] * m) # m steps backward in μ
    steps.extend([-nu - 1] * n) # n steps backward in ν
    
    loop = Loop(start=site, steps=steps)
    return compute_wilson_loop(config, loop)
